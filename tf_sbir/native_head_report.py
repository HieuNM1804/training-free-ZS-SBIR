"""Summarize the fixed pairs, preserving query and class weights as specified in the audit."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .concept_benchmark import atomic_json, checksum
from .data import SKETCHY_SPLITS
from .native_block_probe import DST, SEED
from .native_head_evidence import READOUTS


def summarize(root, split):
    docs, parts = [], []
    for part in (0, 1):
        path = root / f"{split}-{part}-head.pt"
        meta = json.loads(path.with_suffix(".json").read_bytes())
        art = torch.load(path, weights_only=True)
        if meta["sha256"] != checksum(path) or tuple(art["readouts"]) != READOUTS:
            raise ValueError("Wrong artifact hash or readout order")
        parts.append(art)
        docs.append(meta)
    expected = {"sketchy_1": 14179, "sketchy_2": 12694}[split]
    rows = torch.cat([p["rows"] for p in parts])
    if not torch.equal(rows.sort().values, torch.arange(expected)):
        raise ValueError("Missing or duplicated query rows")
    labels = torch.cat([p["labels"] for p in parts]).numpy()
    sem = torch.cat([p["semantic_margin"] for p in parts]).numpy()
    margin = torch.cat([p["margins"] for p in parts]).numpy()
    rates = (margin > 0).astype(float) + 0.5 * (margin == 0)
    cls = np.unique(labels)
    bootstrap = np.random.default_rng(SEED).integers(0, len(cls), size=(10000, len(cls)))
    result = {}
    for group, select in {"all": np.ones(expected, dtype=bool),
                          "semantic_wrong": sem < 0, "semantic_correct": sem > 0}.items():
        by_class = []
        for label in cls:
            keep = select & (labels == label)
            if not keep.any():
                raise ValueError("A class has no eligible observations")
            by_class.append(rates[keep].mean(0))
        by_class = np.array(by_class)
        stats = {}
        for name in ("random_retained_256", "retained", "clip"):
            delta = 100 * (by_class[:, READOUTS.index("discarded")] - by_class[:, READOUTS.index(name)])
            stats["discarded_minus_"+name] = {"class_mean_points": float(delta.mean()),
                        "ci95": np.quantile(delta[bootstrap].mean(1), [.025, .975]).tolist()}
        success = {k: float(100 * v) for k, v in zip(READOUTS, rates[select].mean(0), strict=True)}
        q, r = margin[select, READOUTS.index("discarded")], margin[select, READOUTS.index("retained")]
        result[group] = {"pairs": int(select.sum()), "correct_percent": success, "paired_differences": stats,
                         "discarded_correct_retained_wrong": int(((q > 0) & (r < 0)).sum()),
                         "retained_correct_discarded_wrong": int(((r > 0) & (q < 0)).sum())}
    return {"split": split, "queries": expected, "classes": len(cls), "semantic_ties": int((sem == 0).sum()),
            "groups": result, "source_shards": docs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DST)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    payload = {"revision": args.revision, "seed": SEED,
               "splits": {name: summarize(args.output, name) for name in SKETCHY_SPLITS}}
    atomic_json(args.output / "projection-evidence.json", payload)
    print(json.dumps({s: {group: r["correct_percent"] for group, r in a["groups"].items()}
                      for s, a in payload["splits"].items()}))
