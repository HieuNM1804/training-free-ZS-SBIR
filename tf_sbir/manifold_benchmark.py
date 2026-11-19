"""Ablate query feedback and photo-manifold propagation on Sketchy."""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .asif import paired_anchor_settings
from .benchmark import merge_inputs
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .encoding import load_global_inputs
from .gallery_context import contextual_graph, spectral_coordinates, spectral_query
from .global_methods import centered, prepare_methods
from .layer_views import LAYERS, load_layer_inputs
from .manifold import (
    GalleryGraph, add_gallery_feedback, build_graph, fused_features, iterative_feedback, nonlinear_walk_rows,
)
from .metrics import cosine_rows, evaluate_rows


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", choices=UNSEEN_CLASSES, nargs="+", default=SKETCHY_SPLITS)
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-checkpoint")
    parser.add_argument("--study", choices=("ablation", "diffusion", "scales", "context", "layers", "asif", "coral"), default="ablation")
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


def diffusion_settings(rgb_q, rgb_p, contour_views):
    proxies = F.normalize(contour_views.mean(dim=1), dim=-1)
    shape_q = centered(rgb_q, proxies.mean(dim=0))
    shape_p = centered(proxies, proxies.mean(dim=0))
    q, p = prepare_methods(rgb_q, rgb_p, contour_views, ["local_residual"])["local_residual"]
    for k in (160, 320):
        graph = build_graph(p, shape_p, k, shape_weight=0.0, reciprocal=False)
        p_walk = F.normalize(graph.diffuse(p, restart=0.10, steps=4), dim=-1)
        transition = graph.transition()
        tag = f"k{k}"
        q_first = add_gallery_feedback(q, p_walk, 64, 2.0)
        yield f"first-feedback-{tag}", q_first, p_walk, None
        for rounds in (2, 4):
            feedback = iterative_feedback(q, p_walk, 64, rounds, 2.0)
            yield f"anchored-feedback-{tag}-r{rounds}", feedback, p_walk, None
        contour_walk = F.normalize(graph.diffuse(shape_p, restart=0.10, steps=4), dim=-1)
        q_dual, p_dual = fused_features(q_first, p_walk, shape_q, contour_walk, 0.20)
        yield f"two-view-seeds-{tag}", q_dual, p_dual, None
        for seed_size, weight in ((48, 0.15), (128, 0.15), (128, 0.35)):
            name = f"personalized-walk-{tag}-s{seed_size}-w{weight:g}"
            yield name, q_dual, p_dual, (transition, seed_size, weight)


def scale_settings(rgb_q, rgb_p, contour_views):
    standard, photos = prepare_methods(rgb_q, rgb_p, contour_views, ["local_residual"])["local_residual"]
    stronger, _ = prepare_methods(rgb_q, rgb_p, contour_views, ["local_residual"], 0.9)["local_residual"]
    whole = build_graph(photos, photos, 640)
    banks, standard_queries, stronger_queries = {}, {}, {}
    for k in (160, 320, 640):
        weights = whole.weights[:, :k].contiguous()
        graph = GalleryGraph(whole.indices[:, :k].contiguous(), weights / weights.sum(dim=1, keepdim=True))
        bank = F.normalize(graph.diffuse(photos, restart=0.10, steps=4), dim=-1)
        banks[k] = bank
        standard_queries[k] = iterative_feedback(standard, bank, 64, 2, 2.0)
        stronger_queries[k] = iterative_feedback(stronger, bank, 64, 2, 2.0)
        yield f"scale-k{k}", standard_queries[k], bank
        yield f"scale-k{k}-strong", stronger_queries[k], bank
    for k in (160, 320):
        for weight in (0.35, 0.65):
            pair = fused_features(standard_queries[k], banks[k], stronger_queries[640], banks[640], weight)
            yield f"combine-k{k}-640-w{weight:g}", *pair
    for feedback_size in (192, 384):
        query = iterative_feedback(standard, banks[320], feedback_size, 2, 2.0)
        yield f"scale-k320-feedback{feedback_size}", query, banks[320]


def covariance_roots(source, target, regularization):
    """Estimate CORAL's whiten-and-recolor map from mean-centered features."""
    source_covariance = source.T @ source / (len(source) - 1)
    target_covariance = target.T @ target / (len(target) - 1)
    ridge = regularization * (source_covariance.diag().mean() + target_covariance.diag().mean()) / 2
    source_values, source_axes = torch.linalg.eigh(source_covariance)
    target_values, target_axes = torch.linalg.eigh(target_covariance)
    white = (source_axes * (source_values.clamp_min(0) + ridge).rsqrt()) @ source_axes.T
    color = (target_axes * (target_values.clamp_min(0) + ridge).sqrt()) @ target_axes.T
    return white @ color


def coral_settings(rgb_q, rgb_p, contour_views):
    """Use only synthetic contours and RGB photos to estimate domain statistics."""
    proxies = F.normalize(contour_views.mean(dim=1), dim=-1)
    proxy_mean, photo_mean = proxies.mean(dim=0), rgb_p.mean(dim=0)
    q = centered(rgb_q, proxy_mean)
    p = centered(rgb_p, photo_mean)
    graph = build_graph(p, p, 640)
    bank = F.normalize(graph.diffuse(p, restart=0.1, steps=4), dim=-1)
    unaligned = iterative_feedback(q, bank, 64, 2, 2.0)
    yield "centered-walk-k640", unaligned, bank
    local, _ = prepare_methods(rgb_q, rgb_p, contour_views, ["local_residual"])["local_residual"]
    reference = iterative_feedback(local, bank, 64, 2, 2.0)
    yield "local-residual-walk-k640", reference, bank
    for ridge in (0.1, 0.5, 2.0):
        transform = covariance_roots(proxies - proxy_mean, rgb_p - photo_mean, ridge)
        aligned = F.normalize(q @ transform, dim=-1)
        yield f"coral-r{ridge:g}", aligned, p
        expanded = iterative_feedback(aligned, bank, 64, 2, 2.0)
        yield f"coral-walk-r{ridge:g}", expanded, bank
        mixed = F.normalize(reference.lerp(expanded, 0.5), dim=-1)
        yield f"coral-walk-mixed-r{ridge:g}", mixed, bank


def context_settings(rgb_q, rgb_p, contour_views):
    query, photos = prepare_methods(rgb_q, rgb_p, contour_views, ["local_residual"])["local_residual"]
    initial = build_graph(photos, photos, 320)
    reference_bank = F.normalize(initial.diffuse(photos, restart=0.1, steps=4), dim=-1)
    reference_query = add_gallery_feedback(query, reference_bank, 64, 2.0)
    yield "rgb-walk", reference_query, reference_bank
    for neighborhood, k in ((64, 160), (128, 320)):
        for overlap_weight in (0.35, 0.65):
            name = f"n{neighborhood}-k{k}-w{overlap_weight:g}"
            graph = contextual_graph(photos, initial, neighborhood, k, overlap_weight)
            gallery = F.normalize(graph.diffuse(photos, restart=0.1, steps=4), dim=-1)
            expanded = add_gallery_feedback(query, gallery, 64, 2.0)
            yield f"context-{name}", expanded, gallery
            if overlap_weight == 0.65:
                coordinates = spectral_coordinates(graph, rank=56)
                projected = spectral_query(reference_query, reference_bank, coordinates)
                for weight in (0.25, 0.5):
                    pair = fused_features(expanded, gallery, projected, coordinates, weight)
                    yield f"spectral-{name}-w{weight:g}", *pair


def layer_settings(rgb_q, rgb_p, contour_views, layers):
    query, photo = prepare_methods(rgb_q, rgb_p, contour_views, ["local_residual"])["local_residual"]
    graph = build_graph(photo, photo, 320)
    gallery = F.normalize(graph.diffuse(photo, restart=0.1, steps=4), dim=-1)
    expanded = add_gallery_feedback(query, gallery, 64, 2.0)
    yield "rgb-walk", expanded, gallery
    query_layers, photo_layers, contour_layers = layers
    features = []
    for index, layer in enumerate(LAYERS):
        q, p = prepare_methods(
            query_layers[:, index], photo_layers[:, index], contour_layers[:, index, None], ["local_residual"],
        )["local_residual"]
        p = F.normalize(graph.diffuse(p, restart=0.1, steps=4), dim=-1)
        q = add_gallery_feedback(q, p, 64, 2.0)
        features.append((q, p))
        yield f"layer-{layer + 1}", q, p
        yield f"rgb-layer-{layer + 1}-w0.25", *fused_features(expanded, gallery, q, p, 0.25)
    q = torch.cat([q for q, _ in features], dim=-1) / len(features) ** 0.5
    p = torch.cat([p for _, p in features], dim=-1) / len(features) ** 0.5
    for weight in (0.15, 0.30, 0.50):
        yield f"rgb-multilayer-w{weight:g}", *fused_features(expanded, gallery, q, p, weight)


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
    splits = args.splits
    query_paths, queries = merge_inputs(args.root, splits, "sketch")
    photo_paths, photos = merge_inputs(args.root, splits, "photo")
    all_q, all_p, proxies = load_global_inputs(args, query_paths, photo_paths, True)
    layer_inputs = load_layer_inputs(args, query_paths, photo_paths) if args.study == "layers" else None
    args.output.mkdir(parents=True, exist_ok=True)
    for split in splits:
        ids_q, labels_q = queries[split]
        ids_p, labels_p = photos[split]
        current = (all_q[ids_q].to(device), all_p[ids_p].to(device), proxies[ids_p].to(device))
        if args.study == "diffusion":
            variants = diffusion_settings(*current)
        else:
            if layer_inputs is None:
                feature_settings = {
                    "ablation": settings, "scales": scale_settings, "context": context_settings,
                    "asif": paired_anchor_settings, "coral": coral_settings,
                }[args.study]
                pairs = feature_settings(*current)
            else:
                q_layers, p_layers, c_layers = layer_inputs
                layers = (q_layers[ids_q].to(device), p_layers[ids_p].to(device), c_layers[ids_p].to(device))
                pairs = layer_settings(*current, layers)
            variants = ((name, q, p, None) for name, q, p in pairs)
        for method, q, p, walk in variants:
            filename = args.output / f"{split}-{method}.json"
            if filename.is_file():
                continue
            start = time.perf_counter()
            scores = (
                nonlinear_walk_rows(q, p, *walk, batch_size=min(args.score_batch, 96))
                if walk is not None else cosine_rows(q, p, batch_size=8)
                if method in {"rgb-cosine", "local-residual", "query-expansion"}
                else dot_rows(q, p, args.score_batch)
            )
            ap, precision, map_k, p_k = evaluate_rows(
                scores, labels_q, labels_p, split, f"{split} {method}", workers=args.metric_workers,
            )
            result = {
                "method": method, "split": split, "backbone": "OpenAI CLIP ViT-B/32",
                "queries": len(q), "gallery": len(p), "map_k": map_k, "p_k": p_k,
                "mAP": float(ap), "precision": float(precision), "exploratory_unseen": True,
                "evaluation_seconds": round(time.perf_counter() - start, 2),
            }
            filename.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    run(arguments())
