"""Transfer NeurIPS 2023 ASIF using paired RGB and contour gallery anchors."""

import torch
from torch.nn import functional as F

from .global_methods import centered, prepare_methods
from .manifold import add_gallery_feedback, build_graph, fused_features


def sparse_relative_features(samples, anchors, neighbors, exponent):
    """Use ASIF's top-k anchor coordinates, power sharpening and cosine scoring."""
    scores = samples @ anchors.T
    values, indices = scores.topk(neighbors, dim=1)
    values = values.sign() * values.abs().pow(exponent)
    return F.normalize(torch.zeros_like(scores).scatter_(1, indices, values), dim=-1)


def paired_anchor_settings(queries, photos, contours):
    proxies = F.normalize(contours.mean(dim=1), dim=-1)
    center = proxies.mean(dim=0)
    drawing = centered(queries, center)
    outline = centered(proxies, center)
    visual_query, rgb = prepare_methods(queries, photos, contours, ["local_residual"])["local_residual"]
    graph = build_graph(rgb, rgb, 320)
    smooth = F.normalize(graph.diffuse(rgb, restart=0.1, steps=4), dim=-1)
    feedback = add_gallery_feedback(visual_query, smooth, 64, 2.0)
    yield "rgb-walk", feedback, smooth

    generator = torch.Generator(device=rgb.device).manual_seed(0)
    anchor_ids = torch.randperm(len(rgb), generator=generator, device=rgb.device)[:2048]
    sketch_anchors, photo_anchors = outline[anchor_ids], rgb[anchor_ids]
    for neighbors, exponent in ((64, 1), (128, 2), (256, 4)):
        suffix = f"k{neighbors}-p{exponent}"
        query = sparse_relative_features(drawing, sketch_anchors, neighbors, exponent)
        raw = sparse_relative_features(rgb, photo_anchors, neighbors, exponent)
        smoothed = sparse_relative_features(smooth, photo_anchors, neighbors, exponent)
        yield f"asif-{suffix}", query, raw
        yield f"asif-gallery-walk-{suffix}", query, smoothed
        yield f"asif-rgb-fusion-{suffix}", *fused_features(feedback, smooth, query, smoothed, 0.35)
