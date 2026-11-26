"""Locate ranking changes along the representation used for compulsory alignment."""

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
from .gallery_convex_support_audit import newcomer_hits, prefix
from .locked_geometry_words import score_rows
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import BATCH, SKETCHY, Split, mean_percent, posteriors
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt
from .semantic_boxable_sparse import image_inputs


CONTROLS = ("posterior_cosine", "centered_posterior_cosine", "anchor_dot", "fixed_map_posterior_radii")
CONTRASTS = (
    ("posterior_cosine", "dot"), ("centered_posterior_cosine", "posterior_cosine"),
    ("anchor_dot", "dot"), ("fixed_map", "centered_posterior_cosine"),
    ("fixed_map_posterior_radii", "dot"), ("fixed_map_posterior_radii", "posterior_cosine"),
    ("fixed_map_posterior_radii", "anchor_dot"), ("fixed_map_posterior_radii", "fixed_map"),
    ("fixed_map", "anchor_dot"), ("profile", "fixed_map"),
)


def evaluate(root, main, split, text, geometry, kd, pool, lock, source_lock):
    result = root / f"{split.name}.json"
    artifact = result.with_suffix(".pt")
    if result.is_file():
        report = json.loads(result.read_bytes())
        if report["protocol"] != lock or report["per_query_sha256"] != checksum(artifact):
            raise ValueError("Inconsistent previous representation audit")
        print(split.name, "verified prior audit", flush=True)
        return
    source_path = main / result.name
    primary = json.loads(source_path.read_bytes())
    parent_pt = source_path.with_suffix(".pt")
    if primary["protocol"] != source_lock or checksum(parent_pt) != primary["per_query_sha256"]:
        raise ValueError("Primary run failed verification")
    saved = torch.load(parent_pt, weights_only=True)
    if any(not torch.equal(saved[key], expected) for key, expected in (
        ("qi", split.query_ids), ("pi", split.gallery_ids), ("labels", split.sketch_labels),
    )):
        raise ValueError("Primary sample order changed")
    qbank, pbank = posteriors(split, text, [DEFAULT_PROMPTS])
    q, p = qbank[DEFAULT_PROMPTS[0]], pbank[DEFAULT_PROMPTS[1]]
    pr = p.T.contiguous()
    u, v = q @ geometry.source, pr @ geometry.target
    x, y = F.normalize(u), F.normalize(v)
    qr, yr = q.norm(dim=1), pr.norm(dim=1)
    uniform = 1. / q.shape[1]
    cq, cp = q - uniform, pr - uniform
    qc, pc = F.normalize(q), F.normalize(pr)
    qcc, pcc = F.normalize(cq), F.normalize(cp)
    replay = PrefixReplay(kd, split.photo_labels, split.name)
    gallery, truth = split.photo_labels.cuda(), split.sketch_labels.cuda()
    parts = {arm: [] for arm in CONTROLS}
    pairs, center_error = [], 0.
    for start in range(0, len(q), BATCH):
        sl = slice(start, min(start + BATCH, len(q)))
        dot = q[sl] @ p
        fixed = x[sl] @ y.T
        profile = profile_terms(x[sl], y, geometry.spectrum, QUADRATURE_POINTS)[2]
        order = prefix(dot, *replay.cutoffs)
        base_mask = torch.zeros_like(dot, dtype=torch.bool).scatter_(1, order, True)
        for name, values in (("dot", dot), ("fixed_map", fixed), ("profile", profile)):
            ranked = order if name == "dot" else prefix(values, *replay.cutoffs)
            _, hits = newcomer_hits(ranked, base_mask, gallery, truth[sl], saved["rows"][name][sl, 1])
            if not torch.equal(hits[:, 0].cpu(), saved["rows"][name][sl, 3].bool()):
                raise ValueError("Primary top-1 mismatch for " + name)
        pairs.append(adjacent_pairs(order, gallery, truth[sl], {"profile": profile, "separable": fixed}))
        error = (cq[sl] @ cp.T - (dot - uniform)).abs().max().item()
        center_error = max(center_error, error)
        if center_error > 2e-5:
            raise ValueError("Centering changed Dot beyond its constant shift")
        scores = {
            "posterior_cosine": qc[sl] @ pc.T,
            "centered_posterior_cosine": qcc[sl] @ pcc.T,
            "anchor_dot": u[sl] @ v.T,
            "fixed_map_posterior_radii": (fixed * qr[sl, None]) * yr[None, :],
        }
        for arm, score in scores.items():
            parts[arm].append(score_rows(score, split.sketch_labels[sl], gallery, split.name, replay, pool))
        del scores, score
        if start == 0 or sl.stop == len(q) or (start // BATCH + 1) % 10 == 0:
            print(split.name, sl.stop, "/", len(q), flush=True)
    rows = {arm: torch.cat(blocks).double() for arm, blocks in parts.items()}
    rows.update({arm: saved["rows"][arm] for arm in ("dot", "fixed_map", "profile")})
    paired = torch.cat(pairs)
    save_tensor({
        "qi": split.query_ids, "pi": split.gallery_ids, "labels": split.sketch_labels,
        "rows": rows, "profile_vs_fixed_adjacent_pairs": paired,
    }, artifact)
    report = {
        "protocol": lock, "source_protocol": source_lock, "source_summary_sha256": checksum(source_path),
        "source_per_query_sha256": primary["per_query_sha256"],
        "queries": len(q), "gallery": len(y), "classes": len(truth.unique()), "cutoffs": list(replay.cutoffs),
        "metrics": {arm: mean_percent(row[:, :2]) for arm, row in rows.items()},
        "left_minus_right_macro_ap": {
            left + "_minus_" + right: class_interval(
                100 * (rows[left][:, 0] - rows[right][:, 0]), split.sketch_labels, seed=SEED,
            ) for left, right in CONTRASTS
        },
        "centered_dot_max_error_from_constant_shift": center_error,
        "profile_vs_fixed_adjacent_pairs": pair_summary(paired, split.sketch_labels),
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
    root = main / "representation-audit"
    frozen = json.loads((main / "freeze.json").read_bytes())
    source_lock, source = frozen["sha256"], frozen["config"]
    if source_lock != digest(source) or source["seed"] != SEED or tuple(source["prompts"]) != DEFAULT_PROMPTS:
        raise ValueError("Primary protocol mismatch")
    cfg_vocab = json.loads((DOCS / VOCABULARY_STUDY / "freeze.json").read_bytes())
    vocab_sha = cfg_vocab.pop("sha256")
    words = cfg_vocab["subsets"]["boxable"]["words"]
    if len(words) != 601 or digest(cfg_vocab) != vocab_sha or source["vocabulary_sha256"] != vocab_sha:
        raise ValueError("Unexpected vocabulary")
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
        "controls": list(CONTROLS), "selection": "diagnostic only, no tuning",
        "radius": "uncentered L2 norm of each original 601-word posterior",
        "pairs": "disjoint adjacent fixed-map cosine within Dot prefix, strict reversals",
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
