"""Compare fixed and exploratory patch rerankers on the Sketchy unseen sets."""

import argparse
import itertools
import json
from pathlib import Path
import time

import torch

from .benchmark import merge_inputs
from .data import UNSEEN_CLASSES
from .encoding import global_cache_name, load_global_inputs, save_tensor
from .global_methods import prepare_methods
from .metrics import evaluate_rows
from .patches import load_patch_inputs
from .reranking import (
    PATCH_METHODS, RankingConfig, cache_base_scores, cache_shortlist_scores, reranked_rows,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", choices=UNSEEN_CLASSES, nargs="+", default=list(UNSEEN_CLASSES))
    parser.add_argument("--methods", choices=PATCH_METHODS, nargs="+", default=list(PATCH_METHODS))
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=400)
    parser.add_argument("--shortlist-method", choices=("centered", "query_expansion", "local_residual"), default="centered")
    parser.add_argument("--shortlist-weight", type=float)
    parser.add_argument("--top-k", type=int, nargs="+", default=[200])
    parser.add_argument("--weights", type=float, nargs="+", default=[0.5])
    parser.add_argument("--exploratory-unseen", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--clip-checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patch-batch-size", type=int, default=64)
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--metric-batch-size", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    if args.metric_workers < 1:
        parser.error("--metric-workers must be positive")
    if min(args.batch_size, args.patch_batch_size, args.score_batch_size, args.metric_batch_size) < 1:
        parser.error("batch sizes must be positive")
    if min(args.top_k) < 1 or min(args.weights) < 0:
        parser.error("shortlist sizes must be positive and weights nonnegative")
    if args.max_candidates < max(args.top_k):
        parser.error("--max-candidates must cover every evaluated --top-k")
    if args.shortlist_weight is not None:
        if args.shortlist_method == "centered" or not 0 <= args.shortlist_weight <= 2:
            parser.error("an expansion or residual shortlist accepts a weight from 0 to 2")
        if not args.exploratory_unseen:
            parser.error("custom shortlist weights require --exploratory-unseen")
    if (len(args.top_k) > 1 or len(args.weights) > 1) and not args.exploratory_unseen:
        parser.error("pass --exploratory-unseen when selecting settings on unseen classes")
    return args


def shortlist_cache_path(args, split, query_paths, photo_paths, query_ids, photo_ids):
    photos = [photo_paths[i] for i in photo_ids.tolist()]
    queries = [query_paths[i] for i in query_ids.tolist()]
    precision = "fp16" if torch.device(args.device).type == "cuda" else "fp32"
    query_key = global_cache_name(args.root, queries, "sketch", "rgb", precision, args.clip_checkpoint)
    photo_key = global_cache_name(args.root, photos, "photo", "rgb", precision, args.clip_checkpoint)
    method = args.shortlist_method
    if args.shortlist_weight is not None:
        method += f"-w{args.shortlist_weight:g}"
    return args.cache / f"clip-shortlist-v2-{split}-{method}-{query_key}-{photo_key}-{args.max_candidates}.pt"



@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    device = torch.device(args.device)
    query_paths, query_views = merge_inputs(args.root, args.splits, "sketch")
    photo_paths, photo_views = merge_inputs(args.root, args.splits, "photo")
    if args.encode_only:
        load_patch_inputs(args, query_paths, photo_paths)
        return
    all_q, all_p, all_z = load_global_inputs(args, query_paths, photo_paths, need_contours=True)
    all_patch_p, all_patch_q = load_patch_inputs(args, query_paths, photo_paths)
    args.output.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        query_ids, query_labels = query_views[split]
        photo_ids, photo_labels = photo_views[split]
        q, p = all_q[query_ids].to(device), all_p[photo_ids].to(device)
        q, p = prepare_methods(
            q, p, all_z[photo_ids].to(device), [args.shortlist_method], args.shortlist_weight,
        )[args.shortlist_method]
        if args.max_candidates > len(p):
            raise ValueError(f"Shortlist cannot exceed gallery size for {split}")
        cache = shortlist_cache_path(args, split, query_paths, photo_paths, query_ids, photo_ids)
        full_cache = cache.with_name(f"{cache.stem.rsplit('-', 1)[0]}.f32")
        base_scores = cache_base_scores(q, p, full_cache, args.metric_batch_size)
        saved = torch.load(cache, weights_only=True) if cache.is_file() else {}
        local = saved.get("scores", {})
        missing = [method for method in args.methods if method not in local]
        if missing:
            ids, patch_scores = cache_shortlist_scores(
                base_scores, all_patch_q, all_patch_p, query_ids, photo_ids,
                args.max_candidates, missing, args.score_batch_size, device,
            )
            local.update(patch_scores)
            saved = {"ids": ids, "scores": local}
            save_tensor(saved, cache)
        for method, top_k, weight in itertools.product(args.methods, args.top_k, args.weights):
            filename = args.output / f"{split}-{method}-k{top_k}-w{weight:g}.json"
            if args.resume and filename.is_file():
                print(f"Already evaluated {filename}", flush=True)
                continue
            start = time.perf_counter()
            config = RankingConfig(method, top_k, weight)
            rows = reranked_rows(base_scores, saved["ids"], local[method], config)
            ap, precision, map_k, p_k = evaluate_rows(
                rows, query_labels, photo_labels, split, f"{split} {method} k={top_k} w={weight}",
                workers=args.metric_workers,
            )
            result = {
                "split": split, "method": method, "backbone": "OpenAI CLIP ViT-B/32",
                "patch_backbone": "OpenAI CLIP ViT-B/32 spatial tokens",
                "shortlist": args.shortlist_method, "shortlist_weight": args.shortlist_weight,
                "top_k": top_k, "weight": weight, "exploratory_unseen": args.exploratory_unseen,
                "queries": len(q), "gallery": len(p), "classes": len(UNSEEN_CLASSES[split]),
                "mAP": float(ap), "precision": float(precision), "map_k": map_k, "p_k": p_k,
                "evaluation_seconds": round(time.perf_counter() - start, 2),
                "torch": torch.__version__,
            }
            filename.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    run(parse_args())
