"""Post-hoc exchange of class means and subject deviations between two fixed indices."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json

import torch
from torch.nn.functional import normalize

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .conditional_completion_study import evaluate
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE, load_split
from .native_block_probe import SEED, deterministic
from .replicated_simplex_study import evaluator
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_disputed_word_risk import query_posteriors, word_bank
from .semantic_donor_specificity import OUT, average_donors, boot
from .semantic_operator_audit import SPLITS


def split_moments(p, b, labels):
    mean_only = torch.empty_like(p)
    within_only = torch.empty_like(p)
    scaled = torch.empty_like(p)
    measures = []
    for c in labels.unique().tolist():
        ids = labels == c
        old, new = p[ids], b[ids]
        old_mean, new_mean = old.mean(0), new.mean(0)
        old_center = old - old_mean
        new_center = new - new_mean
        alpha = new_center.norm() / old_center.norm()
        if not torch.isfinite(alpha) or alpha <= 0:
            raise ValueError("The positive classwise scale is not defined")
        mean_only[ids] = new_mean + old_center
        within_only[ids] = old_mean + new_center
        replacement = new_mean + alpha * old_center
        scaled[ids] = replacement
        mean_error = (replacement.mean(0) - new_mean).abs().max().item()
        variance_error = abs(((replacement - replacement.mean(0)).norm()/new_center.norm()).item() - 1)
        if max(mean_error, variance_error) > 1e-9:
            raise ValueError("Class moments were not preserved")
        length = old_center.norm(dim=1) * new_center.norm(dim=1)
        valid = length > 0
        angles = (old_center[valid] * new_center[valid]).sum(1) / length[valid]
        measures.append({"class_id": c, "photos": int(ids.sum()), "rms_scale": alpha.item(),
                         "mean_shift_norm": (new_mean - old_mean).norm().item(),
                         "mean_residual_cosine": angles.mean().item(),
                         "positive_residual_cosine_fraction": (angles > 0).double().mean().item(),
                         "undefined_cosines": int((~valid).sum()), "moment_max_error": max(mean_error, variance_error)})
    if (mean_only + within_only - p - b).abs().max() > 1e-9:
        raise ValueError("Crossed indices do not add to the observed endpoints")
    return {"mean_only": mean_only, "within_only": within_only, "matched_rms": scaled}, measures


@torch.inference_mode()
def run(split_name, revision):
    split, text, _ = load_split(DEFAULT_CACHE, split_name)
    native_file = OUT / f"{split_name}-semantic.pt"
    native_meta = json.loads(native_file.with_suffix(".json").read_bytes())
    if checksum(native_file) != native_meta["per_query_sha256"]:
        raise ValueError("Missing validated category endpoint scores")
    previous = torch.load(native_file, mmap=True, weights_only=True)
    inputs = OUT / f"input-{split_name}.pt"
    if checksum(inputs) != native_meta["input_sha256"]:
        raise ValueError("Mismatched nearest-neighbor index")
    cache = torch.load(inputs, mmap=True, weights_only=True)
    if (not torch.equal(previous["query_ids"], split.query_ids) or
        any(not torch.equal(x["gallery_ids"], split.gallery_ids) for x in (cache, previous))):
        raise ValueError("Mismatched dataset order")
    p = word_bank(normalize(split.photo.float(), dim=1), text[DEFAULT_PROMPTS[1]], split.scale).T.contiguous().double()
    p = (p / p.sum(1, keepdim=True)).float()
    b = average_donors(p, cache["neighbors"]["full64"].cuda())
    banks, measures = split_moments(p.double(), b.double(), split.photo_labels.cuda())
    q = torch.cat([query_posteriors(a, f, text[DEFAULT_PROMPTS[0]], split.scale)
                   for a, f in zip(split.sketch.split(384), split.mirror.split(384), strict=True)])
    rows = {k: previous["category"][k] for k in ("native", "full64")}
    kd = evaluator(DEFAULT_CACHE.parent / "real-sketchy-smoke/evaluator.py")
    replay = PrefixReplay(kd, split.photo_labels, split.name)
    with ThreadPoolExecutor(max_workers=4) as workers:
        for key, bank in banks.items():
            rows[key] = evaluate(split, q, bank.float().contiguous(), replay, workers)[:, :2].double()
    avg = {k: (100*v.mean(0)).tolist() for k, v in rows.items()}
    for endpoint in ("native", "full64"):
        if max(abs(x-y) for x, y in zip(avg[endpoint], native_meta["category_mAP_P"][endpoint], strict=True)) > 1e-9:
            raise ValueError("Archived original endpoint changed")
    base, full = rows["native"], rows["full64"]
    mean, within = rows["mean_only"], rows["within_only"]
    differences = {"mean_first": mean-base, "within_first": within-base, "interaction": full-mean-within+base,
                   "mean_average_of_orders": ((mean-base) + (full-within)) / 2,
                   "within_average_of_orders": ((within-base) + (full-mean)) / 2,
                   "full_minus_matched_rms": full - rows["matched_rms"]}
    if ((differences["mean_average_of_orders"] + differences["within_average_of_orders"]) - (full-base)).abs().max() > 1e-12:
        raise ValueError("Averaged marginal effects do not sum to the full gain")
    effects = {}
    classes = split.sketch_labels.unique()
    for name, delta in differences.items():
        cls = [100 * delta[split.sketch_labels == c].mean(0).numpy() for c in classes]
        effects[name] = {"mean_query_points": (100*delta.mean(0)).tolist(), "class_balanced": boot(cls)}
    path = OUT / f"moments-{split_name}.pt"
    save_tensor({"query_ids": split.query_ids, "gallery_ids": split.gallery_ids, "rows": rows}, path)
    atomic_json(path.with_suffix(".json"), {"revision": revision, "endpoint_revision": native_meta["revision"],
                "seed": SEED, "split": split_name, "queries": len(q), "gallery": len(b),
                "classes": list(UNSEEN_CLASSES[split_name]), "category_mAP_P": avg, "effects": effects,
                "gallery_moments": measures, "device": torch.cuda.get_device_name(),
                "input_sha256": native_meta["input_sha256"], "per_query_sha256": checksum(path)})
    print(split_name, avg, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    deterministic()
    run(args.split, args.revision)
