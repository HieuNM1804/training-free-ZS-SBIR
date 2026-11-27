"""Aggregate paired spatial-response measurements and evaluate the registered decision."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .concept_benchmark import atomic_json, checksum
from .data import SKETCHY_SPLITS
from .native_block_geometry import READOUTS, UPDATES
from .native_block_probe import DST, SEED


def class_interval(values, indices):
    return {"mean": float(values.mean()),
            "ci95": np.quantile(values[indices].mean(1), [.025, .975]).tolist(),
            "positive_classes": int((values > 0).sum()), "classes": len(values)}


def paired_effects(class_means, classes):
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(classes), (10_000, len(classes)))
    source, same_class, other_class = class_means.transpose(1, 0, 2, 3)
    margin = same_class - other_class
    measures = {}
    for col, update in enumerate(UPDATES):
        differences = {
            "source_aligned_minus_identity": source[:, 0, col] - source[:, 1, col],
            "source_aligned_minus_permuted": source[:, 0, col] - source[:, 2, col],
            "category_margin_aligned_minus_identity": margin[:, 0, col] - margin[:, 1, col],
            "category_margin_aligned_minus_permuted": margin[:, 0, col] - margin[:, 2, col],
        }
        measures[update] = {key: class_interval(value, indices) for key, value in differences.items()}
    gate_names = list(measures[UPDATES[0]])[:3]
    return {"effects": measures, "required": gate_names,
            "passed": all(measures[UPDATES[0]][key]["ci95"][0] > 0 for key in gate_names)}


def split_report(name, root):
    scores, links, hashes = [], [], []
    classes = None
    for part in (0, 1):
        path = root / f"{name}-{part}-heldout.pt"
        meta = json.loads(path.with_suffix(".json").read_bytes())
        fingerprint = checksum(path)
        if fingerprint != meta["sha256"]:
            raise ValueError(f"Score hash mismatch for {path}")
        hashes.append(fingerprint)
        item = torch.load(path, weights_only=True, map_location="cpu")
        if classes is not None and classes != item["classes"]:
            raise ValueError("Split shards have different class order")
        classes = item["classes"]
        if tuple(item["updates"]) != UPDATES or tuple(item["readouts"]) != READOUTS:
            raise ValueError("Unexpected readout order")
        links.append(item["links"])
        scores.append(item["scores"])
    links = torch.cat(links)
    scores = torch.cat(scores)
    if scores.shape != (len(classes) * 32 * 2 * 6, 2, 3, 2):
        raise ValueError(f"Unexpected measurement shape {scores.shape}")
    per_class = []
    relatives = {0: "source", 1: "other_subject_same_class", 2: "different_classes"}
    for cls in range(len(classes)):
        kinds = []
        for kind, pairs in enumerate((1, 1, 4)):
            select = (links[:, 0] == cls) & (links[:, 3] == kind)
            subjects, draws = links[select, 1], links[select, 2]
            counts = torch.bincount(subjects * 2 + draws - 1, minlength=64)
            if not torch.all(counts == pairs):
                raise ValueError("Unbalanced redraw panel")
            is_same = links[select, 4] == cls
            is_source = links[select, 5] == subjects
            if not (torch.all(is_same == (kind != 2)) and torch.all(is_source == (kind == 0))):
                raise ValueError("Malformed same-subject or same-class control")
            kinds.append(scores[select].mean((0, 1)))
        per_class.append(torch.stack(kinds))
    per_class = torch.stack(per_class).numpy()
    means = per_class.mean(0)
    table = {update: {label: {readout: float(means[k, i, col]) for i, readout in enumerate(READOUTS)}
                      for k, label in relatives.items()} for col, update in enumerate(UPDATES)}
    return {"split": name, "classes": classes, "subjects": len(classes) * 32,
            "sketches": len(classes) * 64, "query_photo_pairs": len(links),
            "heldout_folds": scores.shape[1], "shard_hashes": hashes,
            "means": table, "class_means": per_class.tolist(),
            "axis_order": [list(relatives.values()), list(READOUTS), list(UPDATES)],
            "decision": paired_effects(per_class, classes)}, per_class


def report(args):
    records = {name: split_report(name, args.output) for name in SKETCHY_SPLITS}
    shared = sorted(set.intersection(*(set(r["classes"]) for r, _ in records.values())))
    summary = {"revision": args.revision, "seed": SEED, "shared_classes": shared,
               "decision_metric": "cosine in [-1,1], spatially centered, paired 95% class bootstrap",
               "splits": {}}
    for name, (record, per_class) in records.items():
        keep = np.array([c not in shared for c in record["classes"]])
        if not keep.any():
            raise ValueError("No split-specific classes")
        record["excluding_shared_query_classes"] = paired_effects(per_class[keep], np.array(record["classes"])[keep])
        summary["splits"][name] = record
    summary["advance_to_retrieval"] = all(r["decision"]["passed"] for r in summary["splits"].values())
    atomic_json(args.output / "summary.json", summary)
    concise = {name: (rec["means"]["spatially_centered"], rec["decision"]["effects"]["spatially_centered"])
               for name, rec in summary["splits"].items()}
    print(json.dumps({"advance": summary["advance_to_retrieval"], "results": concise}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, default=DST)
    torch.set_num_threads(1)
    report(parser.parse_args())
