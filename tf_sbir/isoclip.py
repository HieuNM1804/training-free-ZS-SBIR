"""Apply IsoCLIP's middle singular band to cached ViT-B/32 embeddings."""

from functools import partial

import torch
from torch.nn import functional as F

from .contextual_rank import contextual_propagation
from .encoding import load_frozen_clip
from .gallery_diffusion import diffusion_rows, full_gallery_inputs, linear_fusion_rows, mutual_transition
from .global_methods import prepare_methods, reciprocal_ranks
from .manifold import build_graph, fused_features, iterative_feedback
from .reranking import standardize


def output_space_maps(vision_proj, text_proj, bands, paired=False):
    """Map already-projected vectors without replacing the frozen CLIP projector."""
    vision, text = vision_proj.float(), text_proj.float()
    left, _, right = torch.linalg.svd(vision @ text.T, full_matrices=False)
    image_preimage = torch.linalg.pinv(vision)
    text_preimage = torch.linalg.pinv(text) if paired else None
    maps = {}
    for top, bottom in bands:
        band = slice(top, left.size(1) - bottom)
        middle = left[:, band]
        image_map = (image_preimage @ middle) @ (middle.T @ vision)
        if paired:
            text_middle = right[band].T
            text_map = (text_preimage @ text_middle) @ (text_middle.T @ text)
            maps[(top, bottom)] = (image_map, text_map)
        else:
            maps[(top, bottom)] = image_map
    return maps


def isoclip_maps(args, device, bands=((64, 32), (150, 50)), paired=False):
    model = load_frozen_clip(args.clip_checkpoint, device)
    return output_space_maps(model.visual.proj, model.text_projection, bands, paired)


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
        q, p, c = (F.normalize(vectors @ transform, dim=-1) for vectors in (queries, photos, contours))
        band = f"top{top}-bottom{bottom}"
        yield f"isoclip-cosine-{band}", q, p
        projected = local_feedback(q, p, c)
        yield f"isoclip-walk-{band}", *projected
        yield f"isoclip-fusion-{band}", *fused_features(*reference, *projected, 0.35)


def iso_band_settings(queries, photos, contours, rendered, maps):
    combined = torch.cat((contours, rendered), dim=1)
    for (top, bottom), transform in maps.items():
        band = f"top{top}-bottom{bottom}"
        yield f"iso-band-canny-{band}", *projected_feedback(queries, photos, contours, transform)
        if top in (64, 150):
            yield f"iso-band-xdog-mix-{band}", *projected_feedback(queries, photos, combined, transform)


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


def iso_paired_settings(queries, photos, contours, sketch_words, photo_words, maps, score_batch):
    for (top, bottom), (visual, lexical) in maps.items():
        q, p, shape = (F.normalize(vectors @ visual, dim=-1) for vectors in (queries, photos, contours))
        sketch, camera = (F.normalize(vectors @ lexical, dim=-1) for vectors in (sketch_words, photo_words))
        arguments = (q, p, shape, sketch, camera, score_batch)
        base, region, gallery, _, _ = full_gallery_inputs(*arguments)
        band = f"top{top}-bottom{bottom}"
        yield f"iso-paired-full-{band}", partial(linear_fusion_rows, base, region, score_batch, 0.25)
        transition = mutual_transition(gallery, 160)
        yield f"iso-paired-diffusion-{band}", partial(diffusion_rows, base, region, transition, score_batch, 0.9, 320, "regional")


def iso_context_fusion_rows(visual, context, score_batch, weight, mode):
    for iso, lexical in zip(visual.split(score_batch), context.split(score_batch), strict=True):
        if mode == "rrf":
            scores = reciprocal_ranks(iso) + weight * reciprocal_ranks(lexical)
        else:
            scores = (0.15 * (standardize(iso) + weight * standardize(lexical))).tanh()
        yield from scores.cpu()


def iso_context_settings(queries, photos, contours, sketch_words, photo_words, maps, score_batch):
    query, gallery = projected_feedback(queries, photos, contours, maps[(150, 50)])
    visual = query @ gallery.T
    base, regional, _, _, views = full_gallery_inputs(queries, photos, contours, sketch_words, photo_words, score_batch)
    evidence = standardize(base) + 0.25 * standardize(regional)
    walk = contextual_propagation(photos, views[:, 2].float(), base, regional, evidence, 192, score_batch)
    context = (0.15 * (evidence + 1.2 * walk)).tanh()
    del base, regional, evidence, walk
    for mode in ("rrf", "linear"):
        for weight in (0.25, 0.5, 1.0):
            yield f"iso-context-{mode}-w{weight:g}", partial(
                iso_context_fusion_rows, visual, context, score_batch, weight, mode,
            )
