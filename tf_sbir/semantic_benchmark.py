"""Evaluate public-vocabulary transfer, without using benchmark class names."""

import argparse
import json
from pathlib import Path
import time

import torch

from .benchmark import merge_inputs
from .data import UNSEEN_CLASSES
from .foundation import BACKBONES, load_vision, load_words
from .manifold import fused_features
from .manifold_benchmark import dot_rows
from .metrics import evaluate_rows
from .semantic import (
    agreement_rows, agreement_settings, read_words, semantic_manifold_settings, semantic_settings,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--study", choices=("core", "manifold", "agreement"), default="core")
    parser.add_argument("--backbone", choices=BACKBONES, default="b32")
    parser.add_argument("--clip-checkpoint")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--metric-workers", type=int, default=4)
    parser.add_argument("--score-batch", type=int, default=256)
    parser.add_argument("--exploratory-unseen", action="store_true", required=True)
    return parser.parse_args()


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    splits = tuple(UNSEEN_CLASSES)
    query_paths, queries = merge_inputs(args.root, splits, "sketch")
    photo_paths, photos = merge_inputs(args.root, splits, "photo")
    all_q, all_p, contours = load_vision(args, query_paths, photo_paths, args.study == "agreement")
    words = read_words(args.vocabulary)
    sketch_text, photo_text = [view.to(device) for view in load_words(args, words)]
    args.output.mkdir(parents=True, exist_ok=True)
    for split in splits:
        q_ids, q_labels = queries[split]
        p_ids, p_labels = photos[split]
        q, p = all_q[q_ids].to(device), all_p[p_ids].to(device)
        if args.study == "core":
            settings = {"rgb-cosine": (q, p)}
            settings.update(semantic_settings(q, p, sketch_text, photo_text))
            sq, sp = settings["visual-prototypes-0.3"]
            for blend in (0.15, 0.3):
                settings[f"semantic-plus-rgb-{blend:g}"] = fused_features(sq, sp, q, p, blend)
            variants = ((name, *pair) for name, pair in settings.items())
        elif args.study == "manifold":
            variants = semantic_manifold_settings(q, p, sketch_text, photo_text)
        else:
            variants = agreement_settings(q, p, contours[p_ids].to(device), sketch_text, photo_text)
        for name, target_q, target_p in variants:
            filename = args.output / f"{split}-{name}.json"
            if filename.is_file():
                continue
            start = time.perf_counter()
            rows = (
                agreement_rows(target_q, target_p, args.score_batch)
                if name.startswith("agreement-") else dot_rows(target_q, target_p, args.score_batch)
            )
            ap, precision, map_k, p_k = evaluate_rows(
                rows, q_labels, p_labels, split, f"{split} {name}", workers=args.metric_workers,
            )
            result = {
                "split": split, "method": name, "backbone": BACKBONES[args.backbone],
                "mAP": float(ap), "precision": float(precision), "map_k": map_k, "p_k": p_k,
                "queries": len(q), "gallery": len(p), "external_vocabulary": len(words), "study": args.study,
                "exploratory_unseen": True, "evaluation_seconds": round(time.perf_counter() - start, 2),
            }
            filename.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    run(arguments())
