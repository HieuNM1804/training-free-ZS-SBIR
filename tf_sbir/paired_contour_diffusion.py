"""Diffuse contour descriptors and paired RGB offsets independently on the gallery graph."""

import torch
from torch.nn import functional as F

from .contextual_rank import jaccard_graph
from .contour_graph_index import (
    gallery_walk, graph_inputs, prepare_index_ranker, restarted_features, row_transition, spectral_views,
)
from .global_methods import centered, prepare_methods


def contour_support(query, outlines):
    indices, weights = [], []
    for batch in query.split(256):
        values, ids = (batch @ outlines.T).topk(min(8, len(outlines)))
        indices.append(ids)
        weights.append((values / 0.07).softmax(dim=1))
    return torch.cat(indices), torch.cat(weights)


def apply_offset(query, offsets, support):
    indices, weights = support
    residual = (offsets[indices] * weights[..., None]).sum(dim=1)
    return F.normalize(query + 0.5 * residual, dim=-1)


def paired_contour_diffusion_settings(_queries, photos, contours, sketch_words, photo_words, maps, ink, score_batch):
    ink = F.normalize(ink.float(), dim=-1)
    sketch, target, outlines = spectral_views(ink, photos, contours, maps[(150, 50)])
    paired, gallery = prepare_methods(sketch, target, outlines, ("local_residual",))["local_residual"]
    lexical, language, banks, graphs = graph_inputs(photos, gallery, ink, sketch_words, photo_words)
    move, isolated = row_transition(len(gallery), graphs)
    smooth_gallery = gallery_walk(move, isolated, gallery, 0.8, 4)
    smooth_bank, *smooth_edges = jaccard_graph(smooth_gallery, 192)
    rank = prepare_index_ranker(smooth_gallery, lexical, language, (smooth_bank, banks[1]),
                                (smooth_edges, *graphs[1:]), score_batch)
    yield "paired-diffusion-reference", rank(paired)

    light_gallery = gallery_walk(move, isolated, gallery, 0.5, 4)
    rank_light = prepare_index_ranker(light_gallery, lexical, language, banks, graphs, score_batch)
    yield "paired-diffusion-reference-d0.5", rank_light(paired)
    del rank_light, light_gallery

    outlines = F.normalize(outlines.mean(dim=1), dim=-1)
    center = outlines.mean(dim=0)
    query, outlines = centered(sketch, center), centered(outlines, center)
    offsets = gallery - outlines
    raw_support = contour_support(query, outlines)
    gallery_fields = torch.cat((outlines, offsets), dim=1)

    _, *contour_edges = jaccard_graph(outlines, 192)
    contour_walk = row_transition(len(gallery), (contour_edges,))
    for graph_name, transition, dangling in (("fused", move, isolated), ("contour", *contour_walk)):
        for decay in (0.5, 0.8):
            mixed = restarted_features(transition, dangling, gallery_fields, decay, 4)
            smooth_outlines, smooth_offsets = mixed.chunk(2, dim=1)
            search = contour_support(query, F.normalize(smooth_outlines, dim=-1))
            prefix = f"paired-diffusion-{graph_name}"
            yield f"{prefix}-shape-d{decay:g}", rank(apply_offset(query, offsets, search))
            if graph_name == "fused":
                yield f"{prefix}-offset-d{decay:g}", rank(apply_offset(query, smooth_offsets, raw_support))
            yield f"{prefix}-pair-d{decay:g}", rank(apply_offset(query, smooth_offsets, search))
