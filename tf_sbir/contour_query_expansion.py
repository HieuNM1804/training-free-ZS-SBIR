"""Spectral sketch retrieval with contour-guided and graph-verified query expansion."""

from functools import partial

import torch
from torch.nn import functional as F

from .contextual_rank import column_stochastic, contextual_blocks, jaccard_graph, jaccard_seed
from .global_methods import prepare_methods
from .manifold import iterative_feedback
from .reranking import standardize
from .semantic import code_vectors, top_codes
from .union_density import candidate_union, density_prefixes, density_rows, grow_density, pack_affinity
from .view_hypotheses import view_concepts


def gallery_word_codes(photos, photo_words):
    codes = []
    for batch in photos.split(256):
        indices, weights = top_codes(batch @ photo_words.T, 16)
        codes.append(code_vectors(photo_words, indices, weights))
    return torch.cat(codes)


def verified_feedback(queries, gallery, selected, score_batch):
    """Expand each sketch using the corresponding graph-verified photo group."""
    expanded = []
    for query, members in zip(queries.split(score_batch), selected.split(score_batch), strict=True):
        support = gallery[members]
        similarity = (query[:, None] * support).sum(dim=-1)
        centroid = ((similarity / 0.05).softmax(dim=-1)[..., None] * support).sum(dim=1)
        expanded.append(F.normalize(query + 2 * centroid, dim=-1))
    return torch.cat(expanded)


def pagerank_tail(visual, lexical, prior, banks, transition, dangling, score_batch):
    walks = contextual_blocks(visual, lexical, prior, *banks, transition, dangling, min(256, score_batch))
    return prior + 1.2 * torch.cat(list(walks))


def weighted_query_prefix(visual, lexical, prior, banks, graph, score_batch):
    """Weight visual and lexical query edges by their gallery neighborhood agreement."""
    prefixes = []
    batches = (scores.split(min(score_batch, 512)) for scores in (visual, lexical, prior))
    for appearance, semantic, ties in zip(*batches, strict=True):
        candidates, valid = candidate_union((appearance, semantic), 1280)
        evidence = torch.stack((jaccard_seed(appearance, banks[0]), jaccard_seed(semantic, banks[1])), dim=1)
        coherence = evidence.sum(dim=-1, keepdim=True)
        mass = coherence / coherence.sum(dim=1, keepdim=True).clamp_min(1e-8)
        votes = (evidence * mass).sum(dim=1) * 64.
        prefixes.append(grow_density(votes, ties, candidates, valid, graph, 256))
    return torch.cat(prefixes)


def expansion_sweep(paired, sketch, target, outlines, gallery, lexical, language, banks, graphs, graph, score_batch):
    transition, dangling = column_stochastic(len(gallery), graphs)
    for count, rounds, strength, anchor in (
        (64, 2, 2., 64.), (16, 2, 2., 64.), (32, 2, 2., 64.), (128, 2, 2., 64.),
        (64, 1, 2., 64.), (64, 2, 1., 64.), (64, 2, 4., 64.), (64, 2, 2., 16.),
    ):
        expanded = iterative_feedback(paired, gallery, count, rounds, strength)
        visual = expanded @ gallery.T
        prior = standardize(visual) + language
        prefix = density_prefixes(visual, lexical, (visual, lexical), prior, banks, graph, 1280, score_batch, 256, anchor)
        tail = pagerank_tail(visual, lexical, prior, banks, transition, dangling, score_batch)
        name = f"contour-sweep-k{count}-r{rounds}-w{strength:g}-a{anchor:g}"
        yield name, partial(density_rows, tail, prefix, score_batch, None)
        if (count, rounds, strength, anchor) == (64, 2, 2., 64.):
            adaptive = weighted_query_prefix(visual, lexical, prior, banks, graph, score_batch)
            yield "contour-sweep-neighborhood-weights", partial(density_rows, tail, adaptive, score_batch, None)

    for strength in (0.25, 1.):
        adjusted, _ = prepare_methods(sketch, target, outlines, ("local_residual",), mix_weight=strength)["local_residual"]
        visual = iterative_feedback(adjusted, gallery, 64, 2, 2.) @ gallery.T
        prior = standardize(visual) + language
        prefix = density_prefixes(visual, lexical, (visual, lexical), prior, banks, graph, 1280, score_batch, 256, 64.)
        tail = pagerank_tail(visual, lexical, prior, banks, transition, dangling, score_batch)
        yield f"contour-sweep-residual-w{strength:g}", partial(density_rows, tail, prefix, score_batch, None)


def contour_expansion_settings(_queries, photos, contours, sketch_words, photo_words, maps, ink, score_batch, *, sweep=False):
    ink = F.normalize(ink.float(), dim=-1)
    transform = maps[(150, 50)]
    sketch, target, outlines = (F.normalize(vectors @ transform, dim=-1) for vectors in (ink, photos, contours))
    cues = prepare_methods(sketch, target, outlines, ("centered", "local_residual"))
    centered, gallery = cues["centered"]
    paired, _ = cues["local_residual"]

    words = gallery_word_codes(photos, photo_words)
    if photos.is_cuda:
        words = words.half().float()
    lexical = view_concepts(ink[:, None], sketch_words, photo_words).squeeze(1) @ words.T
    language = standardize(lexical)
    iso_bank, *iso_edges = jaccard_graph(gallery, 192)
    text_bank, *text_edges = jaccard_graph(words, 192)
    _, *rgb_edges = jaccard_graph(photos, 192)
    banks = (iso_bank, text_bank)
    graphs = (iso_edges, text_edges, rgb_edges)
    graph = pack_affinity(len(photos), graphs)

    if sweep:
        yield from expansion_sweep(paired, sketch, target, outlines, gallery, lexical, language,
                                   banks, graphs, graph, score_batch)
        return

    visual = centered @ gallery.T
    prior = standardize(visual) + language
    yield "contour-qe-no-graph", partial(density_rows, prior, None, score_batch, None)
    prefix = density_prefixes(visual, lexical, (visual, lexical), prior, banks, graph, 1280, score_batch, 256, 16.)
    yield "contour-qe-centered", partial(density_rows, prior, prefix, score_batch, None)

    visual = paired @ gallery.T
    prior = standardize(visual) + language
    paired_prefix = density_prefixes(visual, lexical, (visual, lexical), prior, banks, graph, 1280, score_batch, 256, 16.)
    yield "contour-qe-paired", partial(density_rows, prior, paired_prefix, score_batch, None)

    expanded = iterative_feedback(paired, gallery, 64, 2, 2.0)
    visual = expanded @ gallery.T
    prior = standardize(visual) + language
    transition, dangling = column_stochastic(len(photos), graphs)
    for anchor in (4., 16., 64.):
        prefix = density_prefixes(visual, lexical, (visual, lexical), prior, banks, graph, 1280, score_batch, 256, anchor)
        yield f"contour-qe-rgb-a{anchor:g}", partial(density_rows, prior, prefix, score_batch, None)
        if anchor == 16.:
            tail = pagerank_tail(visual, lexical, prior, banks, transition, dangling, score_batch)
            yield "contour-qe-rgb-a16-walk-tail", partial(density_rows, tail, prefix, score_batch, None)
            del tail

    for count in (32, 64):
        expanded = verified_feedback(paired, gallery, paired_prefix[:, :count], min(256, score_batch))
        visual = expanded @ gallery.T
        prior = standardize(visual) + language
        prefix = density_prefixes(visual, lexical, (visual, lexical), prior, banks, graph, 1280, score_batch, 256, 16.)
        yield f"contour-qe-verified{count}", partial(density_rows, prior, prefix, score_batch, None)
        if count == 64:
            tail = pagerank_tail(visual, lexical, prior, banks, transition, dangling, score_batch)
            yield "contour-qe-verified64-walk-tail", partial(density_rows, tail, prefix, score_batch, None)
