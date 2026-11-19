"""Adapt the mixture of gallery edge sources to each individual sketch."""

from functools import partial

import torch
from torch.nn import functional as F

from .conditional_view_affinity import shared_view_votes
from .contextual_rank import column_stochastic, jaccard_seed
from .contour_query_expansion import pagerank_tail
from .reranking import standardize
from .sketch_view_graph_fusion import sketch_view_graph_inputs, source_votes
from .union_density import candidate_union, density_prefixes, density_rows, grow_density, pack_affinity


def source_mixture(consistency, power):
    """Normalize agreement independently for every query, using equal weights if empty."""
    available = F.normalize(consistency, p=1, dim=-1)
    sharpened = F.normalize(available.pow(power), p=1, dim=-1)
    return torch.where(consistency.sum(dim=-1, keepdim=True) > 0, sharpened, 1. / consistency.size(1))


def graph_prefix_variants(visual, lexical, early, late_prior, early_prior, banks, reference_graph, graph, score_batch):
    fixed_weights = (0.15, 0.5, 0.7)
    names = ("late-reference", *(f"lexical{fraction:g}" for fraction in fixed_weights), "adaptive", "adaptive-p2",
             "degree-normalized", "degree-soft", "coupled", "early-adaptive", "early-degree")
    blocks = {name: [] for name in names}
    gallery_degree = graph[1].sum(dim=1).mean(dim=0).clamp_min(1e-8)
    inputs = (*visual, *lexical, *early, late_prior, early_prior)
    batches = (tensor.split(min(score_batch, 512)) for tensor in inputs)
    for crop, full, crop_words, full_words, mixed_visual, mixed_words, late_ties, early_ties in zip(*batches, strict=True):
        visual_bank, lexical_bank, rgb_bank = banks
        late_candidates = candidate_union((crop, full, crop_words, full_words), 1280)
        appearance = source_votes(crop, full, visual_bank, 0.35, True)
        words = source_votes(crop_words, full_words, lexical_bank, 0.65, True)
        rgb = source_votes(crop, full, rgb_bank, 0.35, True)
        shared_visual = shared_view_votes(crop, full, visual_bank, 0.35)
        shared_words = shared_view_votes(crop_words, full_words, lexical_bank, 0.65)
        seed = (appearance + words).lerp(shared_visual + shared_words, 0.5)
        evidence = torch.stack((appearance.sum(dim=1), words.sum(dim=1), rgb.sum(dim=1)), dim=1)
        blocks["late-reference"].append(grow_density(seed * 32, late_ties, *late_candidates, reference_graph, 256))

        def append_late(name, weights, votes=seed):
            result = grow_density(votes * 32, late_ties, *late_candidates, graph, 256, graph_weights=weights)
            blocks[name].append(result)

        for fraction in fixed_weights:
            fixed = ((1 - fraction) / 2, fraction, (1 - fraction) / 2)
            append_late(f"lexical{fraction:g}", seed.new_tensor(fixed).expand(len(seed), -1))
        append_late("adaptive", source_mixture(evidence, 1))
        append_late("adaptive-p2", source_mixture(evidence, 2))
        balanced = source_mixture(evidence / gallery_degree, 1)
        append_late("degree-normalized", balanced)
        append_late("degree-soft", balanced * 0.5 + 1. / 6)
        coupled = (shared_visual + appearance) * (1 - balanced[:, 1:2]) + (shared_words + words) * balanced[:, 1:2]
        append_late("coupled", balanced, coupled)

        early_candidates = candidate_union((mixed_visual, mixed_words), 1280)
        early_votes = tuple(jaccard_seed(scores, bank) for scores, bank in
                            zip((mixed_visual, mixed_words, mixed_visual), banks, strict=True))
        early_seed = (early_votes[0] + early_votes[1]) * 32
        early_strengths = torch.stack([votes.sum(dim=1) for votes in early_votes], dim=1)
        for name, strength in (("early-adaptive", early_strengths), ("early-degree", early_strengths / gallery_degree)):
            weights = source_mixture(strength, 1)
            result = grow_density(early_seed, early_ties, *early_candidates, graph, 256, graph_weights=weights)
            blocks[name].append(result)
    return {name: torch.cat(prefixes) for name, prefixes in blocks.items()}


def query_weighted_gallery_settings(queries, photos, contours, sketch_words, photo_words, maps, ink, score_batch):
    visual, lexical, banks, edges = sketch_view_graph_inputs(
        queries, photos, contours, sketch_words, photo_words, maps, ink, include_rgb_bank=True,
    )
    uniform = pack_affinity(len(photos), edges)
    graph = pack_affinity(len(photos), edges, separate=True)
    transition, dangling = column_stochastic(len(photos), edges)
    v0, v1 = (standardize(scores) for scores in visual)
    l0, l1 = (standardize(scores) for scores in lexical)
    early = (v0.lerp(v1, .5), l0.lerp(l1, .5))
    late = (v0.lerp(v1, .35), l0.lerp(l1, .65))
    del v0, v1, l0, l1
    early_prior = standardize(early[0]) + standardize(early[1])
    late_prior = standardize(late[0]) + standardize(late[1])

    early_prefix = density_prefixes(*early, early, early_prior, banks, uniform, 1280, score_batch, 256, 64.)
    early_tail = pagerank_tail(*early, early_prior, banks[:2], transition, dangling, score_batch)
    yield "weighted-gallery-early-reference", partial(density_rows, early_tail, early_prefix, score_batch, None)
    del early_prefix
    prefixes = graph_prefix_variants(visual, lexical, early, late_prior, early_prior, banks, uniform, graph, score_batch)
    del early_prior, early, visual, lexical, uniform, graph
    late_tail = pagerank_tail(*late, late_prior, banks[:2], transition, dangling, score_batch)
    for name, prefix in prefixes.items():
        tail = early_tail if name.startswith("early-") else late_tail
        yield f"weighted-gallery-{name}", partial(density_rows, tail, prefix, score_batch, None)
