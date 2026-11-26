"""Measure what optimal refitting contributes beyond two explicit pair-aligning maps."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import random

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT, runtime
from .depiction_path_diagnostic import class_interval
from .encoding import load_frozen_clip, save_tensor
from .forced_pair_alignment import AnchorGeometry, QUADRATURE_POINTS, profile_terms
from .forced_pair_study import FAMILY, adjacent_pairs, pair_summary
from .gallery_convex_support import SEED
from .gallery_convex_support_audit import matched_counts, newcomer_hits, prefix, summarize
from .locked_geometry_words import score_rows
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import BATCH, SKETCHY, Split, mean_percent, posteriors
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt
from .semantic_boxable_sparse import image_inputs


CONTROLS = ("best_feasible", "weighted_displacement")
TOLERANCE = 2e-4


def feasible_costs(x, y, s):
    c = (x @ y.T).clamp(-1, 1)
    b = (x * s) @ y.T
    n = (x.square() @ s)[:, None] + (y.square() @ s)[None, :]
    displacement = (n - 2 * b).clamp_min(0)
    rotation = (n - 2 * c * b).clamp_min(0) / (1 + c).clamp_min(1e-7)
    reflection = displacement / (1 - c).clamp_min(1e-7)
    best = torch.minimum(rotation, reflection)

    # Recover the bisector directions explicitly for numerically parallel pairs.
    # The plane is undefined at exact antipodes. There a reflection always costs
    # at most a half-turn in any plane containing the common axis.
    for pairs in (c.abs() > .999).nonzero().split(4096):
        if not len(pairs):
            continue
        i, j = pairs.unbind(1)
        u, v = F.normalize(x[i].double()), F.normalize(y[j].double())
        plus, minus = u + v, u - v
        plus2 = plus.square().sum(1)
        minus2 = minus.square().sum(1)
        energy_minus = minus.square() @ s.double()
        rayleigh_minus = energy_minus / minus2.clamp_min(1e-30)
        rayleigh_plus = (plus.square() @ s.double()) / plus2.clamp_min(1e-30)
        r = .5 * minus2 * (rayleigh_plus + rayleigh_minus)
        h = 2 * rayleigh_minus
        exact = torch.where(plus2 < 1e-24, h, torch.minimum(r, h))
        best[i, j] = torch.where(minus2 < 1e-24, 0., exact).to(best.dtype)
        displacement[i, j] = energy_minus.to(displacement.dtype)
    return best, displacement


def check_maps(device):
    """Check closed-form costs against actual orthogonal matrices, including endpoints."""
    gen = torch.Generator().manual_seed(SEED)
    x, near = F.normalize(torch.randn((2, 5), generator=gen, dtype=torch.float64)).to(device)
    candidates = F.normalize(torch.stack((x, -x, x + 1e-4*near, -x + 1e-4*near, near)))
    s = torch.tensor([1., .8, .5, .2, .1], dtype=torch.float64, device=device)
    got, displacement = feasible_costs(x[None, :], candidates, s)
    eye = torch.eye(x.numel(), device=device, dtype=x.dtype)
    expected = []
    for index, y in enumerate(candidates):
        if index == 0:
            expected.append(0.)
            continue
        diff = F.normalize((x - y), dim=0)
        h = eye - 2 * torch.outer(diff, diff)
        torch.testing.assert_close(h @ y, x)
        h_cost = (s * (1 - h.diag())).sum()
        if index == 1:
            expected.append(h_cost.item())
            continue
        across = F.normalize(x + y, dim=0)
        # Two consecutive Householder reflections give the in-plane rotation.
        r = (eye - 2 * torch.outer(across, across)) @ (eye - 2 * torch.outer(y, y))
        torch.testing.assert_close(r @ y, x)
        expected.append(min(float((s * (1 - r.diag())).sum()), float(h_cost)))
    error = (got[0] - got.new_tensor(expected)).abs().max().item()
    torch.testing.assert_close(displacement[0], (x - candidates).square() @ s)
    if error > 2e-10:
        raise ValueError("Feasible cost failed direct map check")
    return error


def evaluate(root, main, split, text, geometry, kd, pool, lock, source_lock):
    result = root / f"{split.name}.json"
    artifact = result.with_suffix(".pt")
    if result.is_file():
        report = json.loads(result.read_bytes())
        if report["protocol"] != lock or report["per_query_sha256"] != checksum(artifact):
            raise ValueError("Invalid previous feasible-map audit")
        print(split.name, "verified previous audit", flush=True)
        return
    parent_score = main / result.name
    parent_pt = parent_score.with_suffix(".pt")
    parent = json.loads(parent_score.read_bytes())
    if parent["protocol"] != source_lock or checksum(parent_pt) != parent["per_query_sha256"]:
        raise ValueError("Primary run provenance mismatch")
    saved = torch.load(parent_pt, weights_only=True)
    if any(not torch.equal(saved[key], expected) for key, expected in (
        ("qi", split.query_ids), ("pi", split.gallery_ids), ("labels", split.sketch_labels),
    )):
        raise ValueError("Primary run sample ordering mismatch")
    qbank, pbank = posteriors(split, text, [DEFAULT_PROMPTS])
    q, p = qbank[DEFAULT_PROMPTS[0]], pbank[DEFAULT_PROMPTS[1]]
    x, y = geometry.project(q, p.T.contiguous())
    replay = PrefixReplay(kd, split.photo_labels, split.name)
    gallery, truth = split.photo_labels.cuda(), split.sketch_labels.cuda()
    rows = {arm: [] for arm in CONTROLS}
    matched = {arm: [] for arm in CONTROLS}
    pairs = {arm: [] for arm in CONTROLS}
    maximum_violation, slack_sum, slack_max = 0., 0., 0.
    total = len(x) * len(y)
    for start in range(0, len(x), BATCH):
        sl = slice(start, min(start + BATCH, len(x)))
        dot = q[sl] @ p
        order = prefix(dot, *replay.cutoffs)
        in_dot = torch.zeros_like(dot, dtype=torch.bool).scatter_(1, order, True)
        newcomer_hits(order, in_dot, gallery, truth[sl], saved["rows"]["dot"][sl, 1])
        profile = profile_terms(x[sl], y, geometry.spectrum, QUADRATURE_POINTS)[2]
        profile_top = prefix(profile, *replay.cutoffs)
        prof = newcomer_hits(profile_top, in_dot, gallery, truth[sl], saved["rows"]["profile"][sl, 1])
        if not torch.equal(prof[1][:, 0].cpu(), saved["rows"]["profile"][sl, 3].bool()):
            raise ValueError("Could not reproduce profile top-1")
        best, displacement = feasible_costs(x[sl], y, geometry.spectrum)
        slack = best + profile
        if not torch.isfinite(slack).all():
            raise ValueError("Non-finite profile slack")
        maximum_violation = max(maximum_violation, -slack.min().item())
        slack_max = max(slack_max, slack.max().item())
        slack_sum += slack.sum(dtype=torch.float64).item()
        if maximum_violation > TOLERANCE:
            raise ValueError(f"Optimal fit costs more than a feasible map by {maximum_violation}")
        for name, cost in zip(CONTROLS, (best, displacement), strict=True):
            scores = -cost
            value = score_rows(scores, split.sketch_labels[sl], gallery, split.name, replay, pool)
            rows[name].append(value)
            check = newcomer_hits(prefix(scores, *replay.cutoffs), in_dot, gallery, truth[sl], value[:, 1])
            matched[name].append(matched_counts(prof, check))
            pairs[name].append(adjacent_pairs(order, gallery, truth[sl], {"profile": profile, "separable": scores}))
        if start == 0 or sl.stop == len(x) or (start // BATCH + 1) % 10 == 0:
            print(split.name, sl.stop, "/", len(x), flush=True)
    rows = {arm: torch.cat(parts).double() for arm, parts in rows.items()}
    rows.update({key: saved["rows"][key] for key in ("profile", "dot")})
    matched = {arm: torch.cat(parts) for arm, parts in matched.items()}
    pairs = {arm: torch.cat(parts) for arm, parts in pairs.items()}
    save_tensor({"qi": split.query_ids, "pi": split.gallery_ids, "labels": split.sketch_labels,
                 "rows": rows, "matched": matched, "adjacent_pairs": pairs}, artifact)
    report = {
        "protocol": lock, "source_protocol": source_lock, "source_per_query_sha256": parent["per_query_sha256"],
        "queries": len(x), "gallery": len(y), "classes": len(truth.unique()), "cutoffs": list(replay.cutoffs),
        "metrics": {arm: mean_percent(value[:, :2]) for arm, value in rows.items()},
        "profile_minus_control_macro_ap": {arm: class_interval(
            100 * (rows["profile"][:, 0] - rows[arm][:, 0]), split.sketch_labels, seed=SEED,
        ) for arm in CONTROLS},
        "matched_promotions": {arm: summarize(val, split.sketch_labels) for arm, val in matched.items()},
        "adjacent_pair_reversals": {arm: pair_summary(val, split.sketch_labels) for arm, val in pairs.items()},
        "cost_slack": {"pairs": total, "mean": slack_sum / total, "max": slack_max,
                       "max_violation": maximum_violation, "tolerance": TOLERANCE},
        "per_query_sha256": checksum(artifact),
    }
    atomic_json(result, report)
    print(split.name, "verified", report["metrics"], flush=True)


@torch.inference_mode()
def run():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    workspace = Path("/content/sbir")
    main = workspace / "results" / FAMILY
    root = main / "feasible-audit"
    frozen = json.loads((main / "freeze.json").read_bytes())
    source_lock, source = frozen["sha256"], frozen["config"]
    if source_lock != digest(source) or source["seed"] != SEED or tuple(source["prompts"]) != DEFAULT_PROMPTS:
        raise ValueError("Primary protocol mismatch")
    cfg_vocab = json.loads((DOCS / VOCABULARY_STUDY / "freeze.json").read_bytes())
    vocab_sha = cfg_vocab.pop("sha256")
    words = cfg_vocab["subsets"]["boxable"]["words"]
    if vocab_sha != source["vocabulary_sha256"] or len(words) != 601 or digest(cfg_vocab) != vocab_sha:
        raise ValueError("Vocabulary mismatch")
    kd, _ = load_kd(workspace)
    if (source["clip_sha256"] != CHECKPOINT or source["evaluator_sha256"] != KD_FUNCTION_SHA or
            hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA):
        raise ValueError("Inconsistent evaluator or backbone")
    workers = min(4, len(os.sched_getaffinity(0)))
    cfg = runtime(workspace, "sketchy", batch=BATCH, workers=workers)
    lock = save_lock(root / "freeze.json", {
        "commit": revision(), "source_commit": source["commit"], "source_protocol": source_lock,
        "clip_sha256": CHECKPOINT, "evaluator_sha256": KD_FUNCTION_SHA, "vocabulary_sha256": vocab_sha,
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "seed": SEED,
        "controls": list(CONTROLS), "selection": "no tuning", "max_map_fixture_error": check_maps("cuda"),
    })
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = {prompt: encode_prompt(model, tokenizer, words, prompt) for prompt in DEFAULT_PROMPTS}
    geometry = AnchorGeometry(*(text[prompt] for prompt in DEFAULT_PROMPTS))
    data = image_inputs(cfg, SKETCHY, model)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name in SKETCHY:
            evaluate(root, main, Split.from_inputs(data, name), text, geometry, kd, pool, lock, source_lock)
        del data
        cfg = runtime(workspace, "tuberlin", batch=BATCH, workers=workers)
        split = Split.from_inputs(image_inputs(cfg, ("tuberlin",), model), "tuberlin")
        evaluate(root, main, split, text, geometry, kd, pool, lock, source_lock)


if __name__ == "__main__":
    run()
