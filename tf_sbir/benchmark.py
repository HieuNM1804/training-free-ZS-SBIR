"""Run training-free Sketchy retrieval on the two official unseen splits."""

import argparse
import json
from pathlib import Path
import time

import torch

from .data import UNSEEN_CLASSES, collect_paths
from .encoding import load_global_inputs
from .global_methods import GLOBAL_METHODS, prepare_methods
from .metrics import cosine_rows, evaluate_rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", choices=UNSEEN_CLASSES, nargs="+", default=list(UNSEEN_CLASSES))
    parser.add_argument("--methods", choices=GLOBAL_METHODS, nargs="+", default=list(GLOBAL_METHODS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--clip-checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def merge_inputs(root, splits, mode):
    entries = {split: collect_paths(root, UNSEEN_CLASSES[split], mode) for split in splits}
    paths = list(dict.fromkeys(path for members, _ in entries.values() for path in members))
    index = {path: i for i, path in enumerate(paths)}
    views = {
        split: (torch.tensor([index[path] for path in members]), labels)
        for split, (members, labels) in entries.items()
    }
    return paths, views


def explicit_rows(method, queries):
    for batch in queries.split(128):
        yield from method(batch).cpu()


@torch.inference_mode()
def run(args):
    torch.set_num_threads(2)
    device = torch.device(args.device)
    query_paths, query_views = merge_inputs(args.root, args.splits, "sketch")
    photo_paths, photo_views = merge_inputs(args.root, args.splits, "photo")
    print(f"Encoding {len(query_paths)} sketches and {len(photo_paths)} unique photos", flush=True)
    need_contours = any(method != "rgb" for method in args.methods)
    all_q, all_p, all_z = load_global_inputs(args, query_paths, photo_paths, need_contours)
    args.output.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        query_ids, q_labels = query_views[split]
        photo_ids, p_labels = photo_views[split]
        q, p = all_q[query_ids].to(device), all_p[photo_ids].to(device)
        z = all_z[photo_ids].to(device) if all_z is not None else None
        methods = prepare_methods(q, p, z, args.methods)
        for name in args.methods:
            start = time.perf_counter()
            method = methods[name]
            rows = cosine_rows(*method) if isinstance(method, tuple) else explicit_rows(method, q)
            ap, precision, map_k, p_k = evaluate_rows(rows, q_labels, p_labels, split, f"{split} {name}")
            result = {
                "split": split, "method": name, "backbone": "OpenAI CLIP ViT-B/32",
                "queries": len(q), "gallery": len(p), "classes": len(UNSEEN_CLASSES[split]),
                "mAP": float(ap), "precision": float(precision),
                "map_k": map_k, "p_k": p_k,
                "evaluation_seconds": round(time.perf_counter() - start, 2),
                "torch": torch.__version__,
            }
            filename = args.output / f"{split}-{name}.json"
            filename.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    run(parse_args())
