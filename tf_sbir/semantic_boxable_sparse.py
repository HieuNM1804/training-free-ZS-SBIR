"""Compare dot, cosine and sketch-selected top-K terms on the locked S baseline."""

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
from torch.nn import functional as F

from .benchmark import merge_inputs
from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT, runtime
from .depiction_path_diagnostic import class_interval
from .encoding import encode_clip, global_cache_name, load_frozen_clip, load_global_inputs, save_tensor
from .locked_geometry_words import score_rows
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import BATCH, SKETCHY, Split, mean_percent, metric_rows, posteriors
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt
from .semantic_disputed_word_risk import SEED


FAMILY = "semantic-boxable-sparse"
TOP_K = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
SPARSE = ("dot", "cosine", "text")
GRID = (("dot", 601), ("cosine", 601),
        *((method, k) for k in TOP_K for method in SPARSE))
BASE = ("dot", 601)


def key(arm):
    return f"{arm[0]}_{arm[1]}"


def image_inputs(cfg, names, model):
    q_paths, qs = merge_inputs(cfg.root, names, "sketch")
    p_paths, gs = merge_inputs(cfg.root, names, "photo")
    q, p, _ = load_global_inputs(cfg, q_paths, p_paths, False)
    view = cfg.cache / global_cache_name(cfg.root, q_paths, "sketch", "reflected", "fp16", cfg.clip_checkpoint)
    if not view.is_file():
        mirror = encode_clip(model, q_paths, "rgb", cfg.batch_size, cfg.workers, torch.device("cuda"), reflected=True)
        save_tensor(mirror.half(), view)
    mirror = torch.load(view, weights_only=True, mmap=True)
    scale = float(model.logit_scale.exp())
    return qs, gs, q, p, mirror, None, None, scale, None


def score_blocks(q, p, text, photos, arms):
    photo_norm = p.norm(dim=0)
    sizes = sorted({k for _, k in arms})
    need_sparse = any(k < q.size(1) for k in sizes)
    for start in range(0, len(q), BATCH):
        sl = slice(start, min(start + BATCH, len(q)))
        source = q[sl]
        if need_sparse:
            order = source.argsort(dim=1, descending=True, stable=True)
        for k in sizes:
            if k == q.size(1):
                query = source
            else:
                indices = order[:, :k]
                query = torch.zeros_like(source).scatter_(1, indices, source.gather(1, indices))
            if ("dot", k) in arms or ("cosine", k) in arms:
                overlap = query @ p
                if ("dot", k) in arms:
                    yield sl, ("dot", k), overlap
                if ("cosine", k) in arms:
                    norms = query.norm(dim=1, keepdim=True) * photo_norm
                    yield sl, ("cosine", k), overlap / norms.clamp_min(1e-30)
            if ("text", k) in arms:
                yield sl, ("text", k), F.normalize(query @ text, dim=1) @ photos.T


def evaluate(split, text, arms, kd, pool, *, exact):
    query, gallery = posteriors(split, text, [DEFAULT_PROMPTS])
    q, p = query[DEFAULT_PROMPTS[0]], gallery[DEFAULT_PROMPTS[1]]
    label, photo_label = split.sketch_labels.cuda(), split.photo_labels.cuda()
    replay = PrefixReplay(kd, split.photo_labels, split.name)
    map_k, precision_k = replay.cutoffs
    positions = torch.arange(1, (map_k or len(photo_label)) + 1, device="cuda")
    rows = {arm: [] for arm in arms}
    audit_error = 0.
    audit = torch.randperm(len(label), generator=torch.Generator().manual_seed(SEED))[:24].sort().values
    for sl, arm, scores in score_blocks(q, p, text[DEFAULT_PROMPTS[1]], split.photo, set(arms)):
        if exact:
            values = score_rows(scores, split.sketch_labels[sl], photo_label, split.name, replay, pool)[:, :2]
        else:
            values = metric_rows(scores, label[sl], photo_label, positions, precision_k).cpu()
            selected = audit[(audit >= sl.start) & (audit < sl.stop)]
            if len(selected):
                indices = selected - sl.start
                checked = score_rows(scores[indices], split.sketch_labels[selected], photo_label,
                                     split.name, replay, pool)[:, :2]
                audit_error = max(audit_error, (checked - values[indices]).abs().max().item())
        rows[arm].append(values)
    if audit_error > 2e-6:
        raise ValueError(f"CUDA sweep differs from original KD-SBIR by {audit_error}")
    rows = {key(arm): torch.cat(parts).double() for arm, parts in rows.items()}
    if any(value.shape != (len(label), 2) or not value.isfinite().all() for value in rows.values()):
        raise ValueError(f"Incomplete or nonfinite measurements on {split.name}")
    return rows, audit_error, replay.cutoffs


def select(tables):
    def mean(arm):
        return tuple((tables[0][key(arm)][i] + tables[1][key(arm)][i]) / 2 for i in range(2))
    return {method: max((arm for arm in GRID if arm[0] == method and arm[1] in TOP_K), key=mean)
            for method in SPARSE}


def sweep(root, split, text, kd, pool, protocol):
    result_path = root / f"grid-{split.name}.json"
    if result_path.is_file():
        report = json.loads(result_path.read_bytes())
        if report["protocol"] != protocol:
            raise ValueError(f"Changed protocol for {split.name}")
        return report["metrics"]
    print(split.name, "sweeping", len(GRID), "configurations", flush=True)
    rows, error, cutoffs = evaluate(split, text, GRID, kd, pool, exact=False)
    report = {
        "protocol": protocol, "queries": len(split.query_ids), "gallery": len(split.gallery_ids),
        "classes": len(split.sketch_labels.unique()), "cutoffs": list(cutoffs),
        "metrics": {name: mean_percent(value) for name, value in rows.items()}, "max_audit_error": error,
    }
    atomic_json(result_path, report)
    print(split.name, "grid ready", report["metrics"][key(BASE)], flush=True)
    return report["metrics"]


def official(root, split, text, arms, kd, pool, protocol, locked, grid):
    rows, _, cutoffs = evaluate(split, text, arms, kd, pool, exact=True)
    table = {name: mean_percent(value) for name, value in rows.items()}
    if grid is not None:
        for name, scores in table.items():
            if max(abs(a-b) for a, b in zip(scores, grid[name], strict=True)) > 2e-4:
                raise ValueError(f"Final replay differs from the sweep for {split.name}/{name}")
    baseline = rows[key(BASE)]
    contrasts = {}
    for name, values in rows.items():
        difference = 100 * (values - baseline)
        contrasts[name] = {
            "points": difference.mean(0).tolist(),
            "macro_ap": class_interval(difference[:, 0], split.sketch_labels, seed=SEED),
        }
    artifact = root / f"{split.name}.pt"
    save_tensor({"qi": split.query_ids, "pi": split.gallery_ids, "labels": split.sketch_labels, "rows": rows}, artifact)
    report = {
        "protocol": protocol, "selection": locked, "queries": len(split.query_ids), "gallery": len(split.gallery_ids),
        "classes": len(split.sketch_labels.unique()), "cutoffs": list(cutoffs),
        "metrics": table, "delta_vs_dot_full": contrasts, "per_query_sha256": checksum(artifact),
    }
    atomic_json(artifact.with_suffix(".json"), report)
    print(split.name, "original KD-SBIR", table, flush=True)


@torch.inference_mode()
def run():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    workspace = Path("/content/sbir")
    root = workspace / "results" / FAMILY
    config = json.loads((DOCS / VOCABULARY_STUDY / "freeze.json").read_bytes())
    vocab_sha = config.pop("sha256")
    words = config["subsets"]["boxable"]["words"]
    if len(words) != 601 or digest(config) != vocab_sha:
        raise ValueError("Expected the locked 601 boxable concepts")
    kd, _ = load_kd(workspace)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("KD-SBIR source has changed")
    workers = min(4, len(os.sched_getaffinity(0)))
    cfg = runtime(workspace, "sketchy", batch=384, workers=workers)
    spec = {
        "commit": revision(), "clip_sha256": CHECKPOINT, "evaluator_sha256": KD_FUNCTION_SHA,
        "vocabulary_sha256": vocab_sha, "prompts": list(DEFAULT_PROMPTS), "seed": SEED,
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "arms": [list(arm) for arm in GRID],
        "selection": "Per scorer, maximize mean full unseen Sketchy mAP, then mean precision",
    }
    protocol = save_lock(root / "protocol.json", spec)
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = {prompt: encode_prompt(model, tokenizer, words, prompt) for prompt in DEFAULT_PROMPTS}
    data = image_inputs(cfg, SKETCHY, model)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tables = [sweep(root, Split.from_inputs(data, name), text, kd, pool, protocol) for name in SKETCHY]
        winners = select(tables)
        arms = [BASE, ("cosine", len(words)), *winners.values()]
        locked = save_lock(root / "freeze.json", {
            "protocol": protocol, "choices": {name: list(arm) for name, arm in winners.items()},
            "grid_sha256": {name: checksum(root / f"grid-{name}.json") for name in SKETCHY},
        })
        print("Frozen before TU-Berlin", winners, flush=True)
        for name, grid in zip(SKETCHY, tables, strict=True):
            official(root, Split.from_inputs(data, name), text, arms, kd, pool, protocol, locked, grid)
        del data
        cfg = runtime(workspace, "tuberlin", batch=384, workers=workers)
        split = Split.from_inputs(image_inputs(cfg, ("tuberlin",), model), "tuberlin")
        official(root, split, text, arms, kd, pool, protocol, locked, None)


if __name__ == "__main__":
    run()
