"""Ablate query feedback and photo-manifold propagation on Sketchy."""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .benchmark import merge_inputs
from .data import UNSEEN_CLASSES
from .encoding import load_global_inputs
from .global_methods import centered, prepare_methods
from .manifold import add_gallery_feedback, build_graph, fused_features
from .metrics import cosine_rows, evaluate_rows


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-checkpoint")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--metric-workers", type=int, default=4)
    parser.add_argument("--score-batch", type=int, default=256)
    parser.add_argument("--exploratory-unseen", action="store_true", required=True)
    return parser.parse_args()


def settings(rgb_q, rgb_p, contour_views):
    proxies = F.normalize(contour_views.mean(dim=1), dim=-1)
    shape_q = centered(rgb_q, proxies.mean(dim=0))
    shape_p = centered(proxies, proxies.mean(dim=0))
    pairs = prepare_methods(rgb_q, rgb_p, contour_views, ["local_residual", "query_expansion"])
    q, p = pairs["local_residual"]
    qe, _ = pairs["query_expansion"]
    yield "rgb-cosine", rgb_q, rgb_p
    yield "local-residual", q, p
    yield "query-expansion", qe, p
    yield "rgb-gallery-feedback", add_gallery_feedback(q, p, 32, 2.0), p
    yield "rgb-gallery-feedback-wide", add_gallery_feedback(q, p, 128, 2.0), p
    for k in (48, 160):
        for support_weight in (0.0, 0.2):
            tag = f"k{k}-shape{support_weight:g}"
            graph = build_graph(p, shape_p, k, support_weight, reciprocal=support_weight > 0)
            smooth = F.normalize(graph.diffuse(p, restart=0.1, steps=4), dim=-1)
            yield f"walk-{tag}", q, smooth
            feedback = add_gallery_feedback(q, smooth, 64, 2.0)
            yield f"feedback-walk-{tag}", feedback, smooth
            if support_weight:
                shapes = F.normalize(graph.diffuse(shape_p, restart=0.1, steps=4), dim=-1)
                q_fused, p_fused = fused_features(feedback, smooth, shape_q, shapes, 0.15)
                yield f"dual-seed-walk-{tag}", q_fused, p_fused


def dot_rows(queries, gallery, batch_size):
    # Candidate scores use FP32 dot products between L2-normalized embeddings.
    # The TorchMetrics transformation, ranking and aggregation are unchanged.
    for batch in queries.split(batch_size):
        yield from (batch @ gallery.T).cpu()


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("highest")
    splits = tuple(UNSEEN_CLASSES)
    query_paths, queries = merge_inputs(args.root, splits, "sketch")
    photo_paths, photos = merge_inputs(args.root, splits, "photo")
    all_q, all_p, proxies = load_global_inputs(args, query_paths, photo_paths, True)
    args.output.mkdir(parents=True, exist_ok=True)
    for split in splits:
        ids_q, labels_q = queries[split]
        ids_p, labels_p = photos[split]
        current = (all_q[ids_q].to(device), all_p[ids_p].to(device), proxies[ids_p].to(device))
        for method, q, p in settings(*current):
            start = time.perf_counter()
            scores = (
                cosine_rows(q, p, batch_size=8)
                if method in {"rgb-cosine", "local-residual", "query-expansion"}
                else dot_rows(q, p, args.score_batch)
            )
            ap, precision, map_k, p_k = evaluate_rows(
                scores, labels_q, labels_p, split, f"{split} {method}", workers=args.metric_workers,
            )
            result = {
                "method": method, "split": split, "backbone": getattr(args, "backbone_name", "OpenAI CLIP ViT-B/32"),
                "queries": len(q), "gallery": len(p), "map_k": map_k, "p_k": p_k,
                "mAP": float(ap), "precision": float(precision), "exploratory_unseen": True,
                "evaluation_seconds": round(time.perf_counter() - start, 2),
            }
            filename = args.output / f"{split}-{method}.json"
            filename.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    run(arguments())
