"""Use sketch-constrained dominant sets to rerank a union of retrieval candidates."""

from functools import partial

import torch
from torch.nn import functional as F

from .gallery_diffusion import full_gallery_inputs, regional_cosines
from .isoclip import contextual_scores, projected_feedback
from .mta import sketch_view_modes
from .reranking import standardize
from .union_density import candidate_union, density_prefixes, density_rows, gallery_density_graph


def constrained_membership(affinity, candidates, valid, query_edges, penalty, steps):
    """Optimize the sketch-anchored quadratic by the pure-strategy InImDyn updates."""
    nodes = F.pad(valid, (1, 0), value=True)
    edges = F.pad(query_edges.masked_fill(~valid, 0.), (1, 0))
    weights = torch.zeros_like(edges)
    weights[:, 0] = 1.
    payoff = edges.clone()
    for _ in range(steps):
        value = (weights * payoff).sum(dim=1, keepdim=True)
        gains = payoff - value
        insertion, enter = gains.masked_fill(~nodes, -torch.inf).max(dim=1, keepdim=True)
        removal, leave = (-gains).masked_fill(weights <= 1e-8, -torch.inf).max(dim=1, keepdim=True)
        away = removal > insertion
        chosen = torch.where(away, leave, enter)
        mass = weights.gather(1, chosen)
        slope = torch.where(away, removal, insertion).clamp_min(0.)
        direction = torch.where(away, -1., 1.)
        limit = torch.where(away, mass / (1. - mass).clamp_min(1e-8), 1.)
        is_anchor = chosen == 0
        diagonal = torch.where(is_anchor, 0., -penalty)
        curvature = diagonal - 2 * payoff.gather(1, chosen) + value
        step = torch.where(curvature < 0., (slope / (-curvature).clamp_min(1e-12)).minimum(limit), limit)
        signed = (step * direction).masked_fill(slope <= 1e-9, 0.)

        photo = candidates.gather(1, (chosen - 1).clamp_min(0))
        column = affinity[photo, candidates].to(payoff.dtype).masked_fill(~valid, 0.)
        column = torch.cat((edges.gather(1, chosen), column), dim=1)
        column.scatter_(1, chosen, diagonal)
        column = torch.where(is_anchor, edges, column)
        payoff.mul_(1. - signed).addcmul_(signed, column)
        weights.mul_(1. - signed).scatter_add_(1, chosen, signed)

    # Remove the diagonal penalty and the query edges to retain gallery support.
    support = payoff[:, 1:] + penalty * weights[:, 1:] - weights[:, :1] * edges[:, 1:]
    return weights, support.clamp_min_(0.).masked_fill_(~valid, 0.)


def dominant_support(initial, streams, affinity, penalty, strength, score_batch, kernel_bank=None):
    rows = []
    representatives = []
    blocks = (matrix.split(min(score_batch, 512)) for matrix in (initial, *streams))
    for evidence, *cues in zip(*blocks, strict=True):
        indices, valid = candidate_union(cues, min(1280, evidence.size(1)))
        scores = evidence.gather(1, indices)
        peak = scores.masked_fill(~valid, -torch.inf).amax(dim=1, keepdim=True)
        scale = strength * penalty / valid.sum(dim=1, keepdim=True)
        query_edges = (0.5 * (scores - peak)).exp() * scale
        membership, support = constrained_membership(affinity, indices, valid, query_edges, penalty, steps=128)
        support = support / support.amax(dim=1, keepdim=True).clamp_min(1e-8)
        rows.append((indices.to(torch.int32), support.masked_fill_(~valid, -1.)))
        if kernel_bank is not None:
            # A pure-strategy trajectory can activate at most one new photo per step.
            mass, order = membership[:, 1:].topk(min(128, indices.size(1)), dim=1)
            photo = indices.gather(1, order)
            summary = mass.to(kernel_bank.dtype).unsqueeze(1) @ kernel_bank[photo]
            representatives.append(F.normalize(summary.squeeze(1).float(), dim=1))
    return rows, torch.cat(representatives) if representatives else None


def supportive_rows(initial, local_rows, score_batch, weight):
    for scores, (indices, support) in zip(initial.split(min(score_batch, 512)), local_rows, strict=True):
        fused = scores.scatter_add(1, indices.long(), weight * support.clamp_min(0.))
        yield from (0.15 * fused).tanh().cpu()


def supported_prefix(initial, local_rows, score_batch, kernel_score=None):
    prefix = []
    batch = min(score_batch, 512)
    for index, (scores, (ids, support)) in enumerate(zip(initial.split(batch), local_rows, strict=True)):
        ids = ids.long()
        if kernel_score is None:
            bonus = 2. * support
        else:
            block = kernel_score[index * batch:index * batch + len(scores)]
            bonus = 0.8 * block.gather(1, ids)
        priority = (scores.gather(1, ids) + bonus).masked_fill_(support < 0., -torch.inf)
        prefix.append(ids.gather(1, priority.topk(min(256, scores.size(1)), dim=1).indices))
    return torch.cat(prefix)


def dominant_set_settings(
    queries, photos, contours, sketch_words, photo_words, maps, sketch_views,
    logit_scale, sketch_regions, photo_regions, score_batch,
):
    _, robust = sketch_view_modes(queries, sketch_views, sketch_words, logit_scale)
    encoded, gallery = projected_feedback(torch.cat((queries, robust)), photos, contours, maps[(150, 50)])
    original, mode = encoded.split(len(queries))
    visual = mode @ gallery.T
    base, regional, _, _, views = full_gallery_inputs(queries, photos, contours, sketch_words, photo_words, score_batch)
    words = views[:, 2].float()
    context = contextual_scores(photos, words, base, regional, score_batch)
    del base, regional, views
    banks, graph = gallery_density_graph(photos, gallery, words)
    lexical = standardize(context)
    initial = standardize(visual) + lexical
    pair = (visual, context)
    prefix = density_prefixes(visual, context, pair, initial, banks, graph, 1280, score_batch, 256, 16.)
    yield "cds-mta-control", partial(density_rows, initial, prefix, score_batch, None)

    dtype = torch.float16 if queries.is_cuda else torch.float32
    q_regions = sketch_regions[:, 0].to(device=queries.device, dtype=dtype).contiguous()
    p_regions = photo_regions[:, 0].to(device=queries.device, dtype=dtype).contiguous()
    region = regional_cosines(q_regions, p_regions, min(score_batch, 192))
    expanded = (*pair, region)
    region_prefix = density_prefixes(visual, context, expanded, initial, banks, graph, 1280, score_batch, 256, 16.)
    yield "cds-mta-greedy-region-pool", partial(density_rows, initial, region_prefix, score_batch, None)

    original_visual = original @ gallery.T
    original_initial = standardize(original_visual) + lexical + 0.25 * standardize(region)
    prefix = density_prefixes(original_visual, context, (original_visual, context, region), original_initial,
                              banks, graph, 1280, score_batch)
    yield "cds-clearclip-control", partial(density_rows, original_initial, prefix, score_batch, 0.8)
    del region_prefix

    neighbor_ids, strengths = graph
    affinity = strengths.new_zeros((len(photos), len(photos)), dtype=dtype)
    affinity.scatter_add_(1, neighbor_ids, strengths.to(dtype))
    # The maximum weighted degree bounds the largest eigenvalue of every induced subgraph.
    penalty = 1.01 * strengths.sum(dim=1).amax().clamp_min(1e-6)
    kernel_bank = torch.cat([
        F.normalize(view - view.mean(dim=0), dim=1) for view in (gallery, words, photos)
    ], dim=1).to(dtype)

    local_rows, representative = dominant_support(
        original_initial, (original_visual, context, region), affinity, penalty, 2., score_batch, kernel_bank,
    )
    del local_rows
    cosine = standardize((representative.to(dtype) @ kernel_bank.T).float())
    for weight in (0.25, 0.8):
        yield f"cds-clearclip-residual-w{weight:g}", partial(
            density_rows, original_initial + weight * cosine, prefix, score_batch, 0.8,
        )
    del original_initial, original_visual, prefix, cosine

    for pool, streams in (("pair", pair), ("region", expanded)):
        for strength in (2., 8.):
            extend = kernel_bank if pool == "region" and strength == 2. else None
            support, representative = dominant_support(initial, streams, affinity, penalty, strength, score_batch, extend)
            for weight in (0.8, 2.):
                name = f"cds-{pool}-a{strength:g}-w{weight:g}"
                yield name, partial(supportive_rows, initial, support, score_batch, weight)
            if strength == 2.:
                hard = supported_prefix(initial, support, score_batch)
                yield f"cds-{pool}-a2-hard", partial(density_rows, initial, hard, score_batch, None)
            if representative is not None:
                cosine = standardize((representative.to(dtype) @ kernel_bank.T).float())
                hard = supported_prefix(initial, support, score_batch, cosine)
                yield "cds-region-a2-cosine-hard", partial(density_rows, initial, hard, score_batch, None)
                yield "cds-region-a2-cosine-soft", partial(density_rows, initial + 0.8 * cosine, None, score_batch, None)
