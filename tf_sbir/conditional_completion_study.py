"""Full unseen, two-split selection and frozen transfer for conditional completion."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .conditional_completion import complete, counterexample, full_donor_controls, random_frames
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT, runtime
from .depiction_path_diagnostic import class_interval
from .encoding import load_frozen_clip, save_tensor
from .gallery_diffusion import nearest_gallery
from .joint_gallery_witnesses import load_reference
from .locked_geometry_words import score_rows
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import BATCH, SKETCHY, Split, mean_percent, metric_rows, posteriors
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt
from .semantic_boxable_sparse import image_inputs


FAMILY = "conditional-sphere-completion"
SEED = 42
RANKS = (64, 128, 256, 384)
NEIGHBORS = (8, 16, 32, 64, 128, 256)
FRAMES = 4


def evaluate(split, query, index, replay, pool=None):
    gallery = split.photo_labels.cuda()
    labels = split.sketch_labels.cuda()
    cutoff, p_at = replay.cutoffs
    positions = torch.arange(1, (cutoff or len(gallery)) + 1, device=gallery.device)
    bank = index.T.contiguous()
    rows = []
    for start in range(0, len(query), BATCH):
        sl = slice(start, min(start + BATCH, len(query)))
        scores = query[sl] @ bank
        if pool is None:
            rows.append(metric_rows(scores, labels[sl], gallery, positions, p_at).cpu())
        else:
            rows.append(score_rows(scores, split.sketch_labels[sl], gallery, split.name, replay, pool)[:, :2])
    return torch.cat(rows)


def context(split, text, kd):
    qb, pb = posteriors(split, text, [DEFAULT_PROMPTS])
    photos = F.normalize(split.photo.float(), dim=1)
    return (qb[DEFAULT_PROMPTS[0]], pb[DEFAULT_PROMPTS[1]].T, photos, text[DEFAULT_PROMPTS[1]],
            PrefixReplay(kd, split.photo_labels, split.name))


def sweep(root, split, text, kd, frames, protocol):
    target = root / f"grid-{split.name}.json"
    if target.exists():
        table = json.loads(target.read_bytes())
        if table["protocol"] != protocol:
            raise ValueError("Conflicting saved grid")
        return table["metrics"]
    query, _, photos, photo_text, replay = context(split, text, kd)
    result = {}
    full, _ = nearest_gallery(photos, max(NEIGHBORS))
    for arm, centroid in (("donor", False), ("centroid", True)):
        controls = full_donor_controls(photos, photo_text, split.scale, full, NEIGHBORS, centroid=centroid)
        for k, bank in controls.items():
            result[f"{arm}-{k}"] = mean_percent(evaluate(split, query, bank, replay))
        del controls, bank
    for rank in RANKS:
        start = time.monotonic()
        banks, _ = complete(photos, photo_text, split.scale, frames, rank, NEIGHBORS)
        for k, bank in banks.items():
            result[f"completion-{rank}-{k}"] = mean_percent(evaluate(split, query, bank, replay))
        print(split.name, "visible dimensions", rank, {k: [round(v, 3) for v in result[f"completion-{rank}-{k}"]]
              for k in NEIGHBORS}, "seconds", round(time.monotonic()-start, 1), flush=True)
        del banks, bank
    atomic_json(target, {"protocol": protocol, "queries": len(query), "gallery": len(photos), "metrics": result})
    return result


def select(tables, prefix):
    def key(name):
        return tuple(sum(table[name][col] for table in tables) / len(tables) for col in (0, 1))
    return max((n for n in tables[0] if n.startswith(prefix + "-")), key=key)


def official(root, split, text, kd, pool, frames, lock, selection, sweep_metrics=None):
    target = root / f"{split.name}.json"
    artifact = target.with_suffix(".pt")
    if target.exists():
        previous = json.loads(target.read_bytes())
        if previous["selection"] != selection or previous["per_query_sha256"] != checksum(artifact):
            raise ValueError("Conflicting saved original KD results")
        return
    q, native, photos, t, replay = context(split, text, kd)
    rows = {}
    diagnostics = {}
    def record(name, bank, grid_key=None):
        rows[name] = evaluate(split, q, bank, replay, pool)
        values = mean_percent(rows[name])
        if sweep_metrics is not None and grid_key is not None:
            if max(abs(a-b) for a,b in zip(values, sweep_metrics[grid_key], strict=True)) > .00025:
                raise ValueError(f"Full original KD replay differs from selection: {name}")
        print(split.name, name, values, flush=True)

    start = time.monotonic()
    full, _ = nearest_gallery(photos, max(lock["k"], lock["donor_k"], lock["centroid_k"]))
    record("dot", native)
    saved, _ = load_reference(split)
    drift = [a-b for a,b in zip(mean_percent(rows["dot"]), mean_percent(saved["dot"][:, :2]), strict=True)]
    if max(map(abs, drift)) > .01:
        raise ValueError(f"Reference ranking drift in percentage points: {drift}")
    k, r = lock["k"], lock["rank"]
    for name, readout, selection_mode in (
        ("completion", "posterior", "visible"), ("same_visible_donors", "donor", "visible"),
        ("mean_completed_logits", "mean_logits", "visible"), ("full_selection", "posterior", "full"),
        ("random_selection", "posterior", "random"),
    ):
        torch.cuda.synchronize()
        before = time.monotonic()
        banks, info = complete(photos, t, split.scale, frames, r, [k], selection=selection_mode,
                               full_indices=full, readout=readout, seed=SEED)
        torch.cuda.synchronize()
        info["build_seconds"] = time.monotonic() - before
        diagnostics[name] = info
        record(name, banks[k], lock["winner"] if name == "completion" else None)
        del banks
    for arm, param, centroid in (("donor", "donor_k", False), ("centroid", "centroid_k", True)):
        tuned = lock[param]
        controls = full_donor_controls(photos, t, split.scale, full, sorted({k, tuned}), centroid=centroid)
        record(f"full_{arm}_matched_k", controls[k])
        if k == tuned:
            rows[f"full_{arm}_best"] = rows[f"full_{arm}_matched_k"]
        else:
            record(f"full_{arm}_best", controls[tuned], f"{arm}-{tuned}")
    deltas = {arm: class_interval(100 * (rows["completion"][:, 0] - values[:, 0]),
                                  split.sketch_labels, seed=SEED) for arm, values in rows.items()}
    save_tensor({"qi": split.query_ids, "pi": split.gallery_ids, "labels": split.sketch_labels,
                 "rows": rows}, artifact)
    atomic_json(target, {"commit": revision(), "selection": selection, "queries": len(q), "gallery": len(photos),
                         "classes": len(split.sketch_labels.unique()), "device": torch.cuda.get_device_name(),
                         "cutoffs": replay.cutoffs, "metrics": {name: mean_percent(v) for name,v in rows.items()},
                         "completion_minus_control_class_macro_ap": deltas, "diagnostics": diagnostics,
                         "baseline_drift_points": drift, "seconds": time.monotonic() - start,
                         "per_query_sha256": checksum(artifact)})


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
    vocabulary = json.loads((DOCS / VOCABULARY_STUDY / "freeze.json").read_bytes())
    vocab_sha = vocabulary.pop("sha256")
    words = vocabulary["subsets"]["boxable"]["words"]
    if len(words) != 601 or digest(vocabulary) != vocab_sha:
        raise ValueError("Unexpected vocabulary")
    kd, _ = load_kd(workspace)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("Original KD-SBIR evaluator changed")
    workers = min(4, len(os.sched_getaffinity(0)))
    protocol = save_lock(root / "protocol.json", {
        "commit": revision(), "seed": SEED, "ranks": RANKS, "counts": NEIGHBORS, "frames": FRAMES,
        "checkpoint": CHECKPOINT, "vocabulary": vocab_sha, "prompts": DEFAULT_PROMPTS, "evaluator": KD_FUNCTION_SHA,
        "selection": "Maximize mean full-unseen Sketchy mAP, mean precision breaks ties. Lock before TU.",
        "control_selection": "Optimize both ordinary donor-posterior mean and unit full-feature centroid separately.",
        "toy": counterexample(), "gpu": torch.cuda.get_device_name(), "pytorch": str(torch.__version__),
    })
    cfg = runtime(workspace, "sketchy", batch=BATCH, workers=workers)
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = {p: encode_prompt(model, tokenizer, words, p) for p in DEFAULT_PROMPTS}
    frames = random_frames(text[DEFAULT_PROMPTS[1]].size(1), FRAMES, SEED, torch.device("cuda"))
    data = image_inputs(cfg, SKETCHY, model)
    grids = [sweep(root, Split.from_inputs(data, name), text, kd, frames, protocol) for name in SKETCHY]
    winner = select(grids, "completion")
    _, r, k = winner.split("-")
    lock = {"commit": revision(), "protocol": protocol, "rank": int(r), "k": int(k), "winner": winner,
            "donor_k": int(select(grids, "donor").split("-")[-1]),
            "centroid_k": int(select(grids, "centroid").split("-")[-1]),
            "grids": {name: checksum(root / f"grid-{name}.json") for name in SKETCHY},
            "metrics_selected_on_sketchy": [grid[winner] for grid in grids]}
    selection = save_lock(root / "freeze.json", lock)
    print("FROZEN before transfer", lock, flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name, grid in zip(SKETCHY, grids, strict=True):
            official(root, Split.from_inputs(data, name), text, kd, pool, frames, lock, selection, grid)
        del data
        cfg = runtime(workspace, "tuberlin", batch=BATCH, workers=workers)
        split = Split.from_inputs(image_inputs(cfg, ("tuberlin",), model), "tuberlin")
        official(root, split, text, kd, pool, frames, lock, selection)


if __name__ == "__main__":
    run()
