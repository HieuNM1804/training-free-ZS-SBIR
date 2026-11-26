"""Attribute substitution rankings to refitting, gallery pair identity, and omission."""

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

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT, runtime
from .depiction_path_diagnostic import class_interval
from .encoding import load_frozen_clip, save_tensor
from .forced_pair_alignment import QUADRATURE_POINTS, profile_terms
from .gallery_convex_support import SEED
from .gallery_convex_support_audit import matched_counts, newcomer_hits, prefix, summarize
from .gallery_substitution_alignment import GallerySubstitution, paired_profiles, sphere
from .gallery_substitution_study import FAMILY
from .locked_geometry_words import score_rows
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import BATCH, SKETCHY, Split, mean_percent, posteriors
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt
from .semantic_boxable_sparse import image_inputs
from .semantic_disputed_word_risk import word_bank


CONTROLS = ("centered_fixed_map", "undeleted", "permuted_estimator")
BLOCK_SIZE = 32


def permuted_estimator(source, target):
    order = target.square().sum(1).argsort(stable=True)
    generator = torch.Generator().manual_seed(SEED)
    perm = torch.arange(len(target), device=order.device)
    for group in order.split(BLOCK_SIZE):
        perm[group] = group[torch.randperm(len(group), generator=generator).to(group.device)]
    chart = GallerySubstitution(source[perm], target, seed=SEED)
    # Undo the shuffle only for pricing each candidate's natural source reading.
    a = chart.a[perm.argsort()].contiguous()
    natural = paired_profiles(a, chart.b, chart.s)
    propensity = target.square().sum(1)
    return chart, a, natural, {
        "block": BLOCK_SIZE, "fixed_point_count": int((perm == torch.arange(len(perm), device=perm.device)).sum()),
        "mean_absolute_propensity_change": float((propensity - propensity[perm]).abs().mean()),
        "permutation_sha256": hashlib.sha256(perm.cpu().numpy().tobytes()).hexdigest(),
    }


def deletion_audit(raw, delta, deleted, raw_top, deleted_top):
    limit = delta.amax(1) - delta.amin(1)
    was_in = torch.zeros_like(raw, dtype=torch.bool).scatter_(1, raw_top, True)
    actual = (~was_in.gather(1, deleted_top)).sum(1)
    threshold = raw.gather(1, raw_top[:, -1:])
    # An outsider below the kth score by more than the span of all corrections
    # cannot enter the top K. Float32 addition at the threshold also admits ties.
    rounding = 4 * torch.finfo(raw.dtype).eps * (1 + deleted.abs().amax(1))
    contenders = ((~was_in) & (raw >= threshold - (limit + rounding)[:, None])).sum(1)
    bound = contenders.clamp_max(raw_top.size(1))
    if (actual > bound).any():
        raise ArithmeticError("Top-K change exceeded its perturbation bound")
    top1_change = raw_top[:, 0] != deleted_top[:, 0]
    margin = (raw.gather(1, raw_top[:, :1]) - raw.gather(1, raw_top[:, 1:2])).flatten()
    possible = margin <= limit + rounding
    if (top1_change & ~possible).any():
        raise ArithmeticError("Top-1 changed beyond its perturbation bound")
    return torch.stack((actual, bound, top1_change, possible), 1).cpu(), limit.max().item()


def evaluate(root, main, split, text, kd, pool, lock, source_lock):
    result = root / f"{split.name}.json"
    artifact = result.with_suffix(".pt")
    if result.is_file():
        prior = json.loads(result.read_bytes())
        if prior["protocol"] != lock or checksum(artifact) != prior["per_query_sha256"]:
            raise ValueError("Invalid previous audit")
        print(split.name, "verified existing audit", flush=True)
        return
    source_pt = main / artifact.name
    source_report = json.loads((main / result.name).read_bytes())
    if source_report["protocol"] != source_lock or checksum(source_pt) != source_report["per_query_sha256"]:
        raise ValueError("Main experiment verification failed")
    saved = torch.load(source_pt, weights_only=True)
    if any(not torch.equal(saved[key], value) for key, value in (
        ("qi", split.query_ids), ("pi", split.gallery_ids), ("labels", split.sketch_labels),
    )):
        raise ValueError("Main experiment and audit have different sample order")
    qbank, pbank = posteriors(split, text, [DEFAULT_PROMPTS])
    q, p = qbank[DEFAULT_PROMPTS[0]], pbank[DEFAULT_PROMPTS[1]]
    source = word_bank(split.photo, text[DEFAULT_PROMPTS[0]], split.scale).T.contiguous()
    target = p.T.contiguous()
    chart = GallerySubstitution(source, target, seed=SEED)
    shuffled, a_null, observed_null, perm_info = permuted_estimator(source, target)
    x, x_null = chart.project_queries(q), shuffled.project_queries(q)
    numerical = chart.check(x[:2])
    fixed_self = (chart.a * chart.b).sum(1)
    replay = PrefixReplay(kd, split.photo_labels, split.name)
    gallery, truth = split.photo_labels.cuda(), split.sketch_labels.cuda()
    rows = {name: [] for name in CONTROLS}
    matches = {name: [] for name in CONTROLS}
    deletion_rows, span_max = [], 0.
    for start in range(0, len(q), BATCH):
        sl = slice(start, min(start + BATCH, len(q)))
        raw = profile_terms(x[sl], chart.b, chart.s, QUADRATURE_POINTS)[2]
        undeleted = raw - chart.observed
        correction = (1 - x[sl] @ chart.a.T) / (chart.n * chart.unit)
        method = undeleted + correction
        null_raw = profile_terms(x_null[sl], shuffled.b, shuffled.s, QUADRATURE_POINTS)[2]
        null_scores = null_raw - observed_null + (1 - x_null[sl] @ a_null.T) / (shuffled.n * shuffled.unit)
        fixed_scores = x[sl] @ chart.b.T - fixed_self
        dot = q[sl] @ p
        base_top = prefix(dot, *replay.cutoffs)
        base_mask = torch.zeros_like(dot, dtype=torch.bool).scatter_(1, base_top, True)
        method_top = prefix(method, *replay.cutoffs)
        newcomer_hits(base_top, base_mask, gallery, truth[sl], saved["rows"]["dot"][sl, 1])
        main_hits = newcomer_hits(method_top, base_mask, gallery, truth[sl], saved["rows"]["substitution"][sl, 1])
        if not torch.equal(main_hits[1][:, 0].cpu(), saved["rows"]["substitution"][sl, 3].bool()):
            raise ValueError("Failed to reproduce original substitution top-1")
        for name, score in zip(CONTROLS, (fixed_scores, undeleted, null_scores), strict=True):
            if not torch.isfinite(score).all():
                raise ValueError("Non-finite " + name)
            identified = score_rows(score, split.sketch_labels[sl], gallery, split.name, replay, pool)
            rows[name].append(identified)
            top = prefix(score, *replay.cutoffs)
            new = newcomer_hits(top, base_mask, gallery, truth[sl], identified[:, 1])
            matches[name].append(matched_counts(main_hits, new))
            if name == "undeleted":
                changes, span = deletion_audit(undeleted, correction, method, top, method_top)
                deletion_rows.append(changes)
                span_max = max(span_max, span)
        if start == 0 or sl.stop == len(q) or (start // BATCH + 1) % 10 == 0:
            print(split.name, sl.stop, "/", len(q), flush=True)
    rows = {name: torch.cat(blocks).double() for name, blocks in rows.items()}
    rows.update({name: saved["rows"][name] for name in ("dot", "substitution")})
    matches = {name: torch.cat(blocks) for name, blocks in matches.items()}
    changes = torch.cat(deletion_rows)
    save_tensor({"qi": split.query_ids, "pi": split.gallery_ids, "labels": split.sketch_labels,
                 "rows": rows, "matched": matches, "deletion_changes": changes}, artifact)
    table = {
        "protocol": lock, "source_protocol": source_lock, "source_per_query_sha256": source_report["per_query_sha256"],
        "classes": len(truth.unique()), "queries": len(q), "gallery": len(target), "cutoffs": list(replay.cutoffs),
        "metrics": {name: mean_percent(value[:, :2]) for name, value in rows.items()},
        "primary_minus_control_macro_ap": {name: class_interval(
            100 * (rows["substitution"][:, 0] - rows[name][:, 0]), split.sketch_labels, seed=SEED,
        ) for name in CONTROLS},
        "matched_promotions": {name: summarize(values, split.sketch_labels) for name, values in matches.items()},
        "numerics": numerical, "estimator_shuffle": perm_info,
        "deletion_order": {
            "observed_new_top_k_items": int(changes[:, 0].sum()), "bounded_new_top_k_items": int(changes[:, 1].sum()),
            "queries_with_top_k_set_change": int((changes[:, 0] > 0).sum()),
            "observed_top1_id_changes": int(changes[:, 2].sum()), "bounded_top1_id_changes": int(changes[:, 3].sum()),
            "maximum_query_correction_span": span_max, "analytic_span_upper_bound": 1 / (chart.n * chart.unit),
        },
        "per_query_sha256": checksum(artifact),
    }
    atomic_json(result, table)
    print(split.name, "verified", table["metrics"], flush=True)


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
    root = main / "attribution"
    frozen = json.loads((main / "freeze.json").read_bytes())
    source_lock, cfg_main = frozen["sha256"], frozen["config"]
    if digest(cfg_main) != source_lock or cfg_main["seed"] != SEED or tuple(cfg_main["prompts"]) != DEFAULT_PROMPTS:
        raise ValueError("Invalid primary protocol")
    cfg_vocab = json.loads((DOCS / VOCABULARY_STUDY / "freeze.json").read_bytes())
    word_sha = cfg_vocab.pop("sha256")
    words = cfg_vocab["subsets"]["boxable"]["words"]
    if len(words) != 601 or digest(cfg_vocab) != word_sha or cfg_main["vocabulary_sha256"] != word_sha:
        raise ValueError("Unexpected 601-word vocabulary")
    kd, _ = load_kd(workspace)
    if (cfg_main["clip_sha256"] != CHECKPOINT or cfg_main["evaluator_sha256"] != KD_FUNCTION_SHA or
            hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA):
        raise ValueError("Primary backbone or exact evaluator changed")
    workers = min(4, len(os.sched_getaffinity(0)))
    cfg = runtime(workspace, "sketchy", batch=BATCH, workers=workers)
    lock = save_lock(root / "freeze.json", {
        "commit": revision(), "source_commit": cfg_main["commit"], "source_protocol": source_lock,
        "seed": SEED, "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "arms": list(CONTROLS), "selection": "fixed, same three controls on each split",
        "permuted_estimator": "shuffle source RGB heads within target concentration blocks of 32, retain natural reference",
        "deletion_bound": "undeleted Kth threshold minus range of query-specific corrections with roundoff allowance",
    })
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    tok = open_clip.get_tokenizer("ViT-B-32")
    text = {prompt: encode_prompt(model, tok, words, prompt) for prompt in DEFAULT_PROMPTS}
    data = image_inputs(cfg, SKETCHY, model)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name in SKETCHY:
            evaluate(root, main, Split.from_inputs(data, name), text, kd, pool, lock, source_lock)
        del data
        cfg = runtime(workspace, "tuberlin", batch=BATCH, workers=workers)
        split = Split.from_inputs(image_inputs(cfg, ("tuberlin",), model), "tuberlin")
        evaluate(root, main, split, text, kd, pool, lock, source_lock)


if __name__ == "__main__":
    run()
