"""Apply IsoCLIP's middle singular band to cached ViT-B/32 embeddings."""

from functools import partial

import torch
from torch.nn import functional as F

from .encoding import load_frozen_clip
from .gallery_diffusion import diffusion_rows, full_gallery_inputs, linear_fusion_rows, mutual_transition
from .global_methods import prepare_methods
from .manifold import build_graph, fused_features, iterative_feedback
from .reranking import standardize


def output_space_maps(vision_proj, text_proj, bands):
    """Map already-projected vectors without replacing the frozen CLIP projector."""
    vision, text = vision_proj.float(), text_proj.float()
    left, _, _ = torch.linalg.svd(vision @ text.T, full_matrices=False)
    preimage = torch.linalg.pinv(vision)
    maps = {}
    for top, bottom in bands:
        middle = left[:, top:left.size(1) - bottom]
        maps[(top, bottom)] = (preimage @ middle) @ (middle.T @ vision)
    return maps


def isoclip_maps(args, device):
    model = load_frozen_clip(args.clip_checkpoint, device)
    return output_space_maps(model.visual.proj, model.text_projection, ((64, 32), (150, 50)))


def local_feedback(queries, photos, contours):
    query, gallery = prepare_methods(queries, photos, contours, ("local_residual",))["local_residual"]
    graph = build_graph(gallery, gallery, 640)
    bank = F.normalize(graph.diffuse(gallery, restart=0.1, steps=4), dim=-1)
    return iterative_feedback(query, bank, 64, 2, 2.0), bank


def projected_feedback(queries, photos, contours, transform):
    q, p, c = (F.normalize(vectors @ transform, dim=-1) for vectors in (queries, photos, contours))
    return local_feedback(q, p, c)


def iso_image_settings(queries, photos, contours, maps):
    reference = local_feedback(queries, photos, contours)
    yield "isoclip-image-control", *reference
    for (top, bottom), transform in maps.items():
        projected = projected_feedback(queries, photos, contours, transform)
        band = f"top{top}-bottom{bottom}"
        yield f"isoclip-walk-{band}", *projected
        yield f"isoclip-fusion-{band}", *fused_features(*reference, *projected, 0.35)


def iso_transport_settings(queries, photos, contours, sketch_words, photo_words, maps, score_batch):
    base, region, gallery, _, _ = full_gallery_inputs(queries, photos, contours, sketch_words, photo_words, score_batch)
    transition = mutual_transition(gallery, 160)
    yield "isoclip-diffusion-control", partial(diffusion_rows, base, region, transition, score_batch, 0.9, 320, "regional")
    baseline = standardize(base)
    for (top, bottom), transform in maps.items():
        query, photos_iso = projected_feedback(queries, photos, contours, transform)
        evidence = baseline + 0.35 * standardize(query @ photos_iso.T)
        band = f"top{top}-bottom{bottom}"
        yield f"isoclip-transport-{band}", partial(linear_fusion_rows, evidence, region, score_batch, 0.25)
        # Concatenated unit vectors produce a weighted sum of the two view cosines.
        geometry = torch.cat((gallery * (0.65 ** 0.5), photos_iso * (0.35 ** 0.5)), dim=1)
        iso_graph = mutual_transition(geometry, 160)
        yield f"isoclip-diffusion-{band}", partial(diffusion_rows, evidence, region, iso_graph, score_batch, 0.9, 320, "regional")
