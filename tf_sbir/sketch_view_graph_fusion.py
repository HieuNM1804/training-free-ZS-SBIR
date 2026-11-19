"""Fuse the full sketch and its foreground crop as separate retrieval sources."""

from functools import partial

import torch
from torch.nn import functional as F

from .contextual_rank import column_stochastic, jaccard_graph, jaccard_seed
from .contour_graph_index import gallery_walk, row_transition
from .contour_query_expansion import gallery_word_codes, pagerank_tail
from .global_methods import prepare_methods
from .manifold import iterative_feedback
from .reranking import standardize
from .union_density import candidate_union, density_prefixes, density_rows, grow_density, pack_affinity
from .view_hypotheses import view_concepts


def source_votes(crop, full, bank, full_weight, adaptive):
    crop_votes = jaccard_seed(crop, bank)
    if full_weight == 0:
        return crop_votes
    full_votes = jaccard_seed(full, bank)
    if adaptive:
        crop_support = crop_votes.sum(dim=1, keepdim=True) * (1 - full_weight)
        full_support = full_votes.sum(dim=1, keepdim=True) * full_weight
        full_weight = full_support / (crop_support + full_support).clamp_min(1e-8)
    return crop_votes.lerp(full_votes, full_weight)


def separate_view_prefixes(visual, lexical, prior, banks, graph, visual_weight, lexical_weight, adaptive, score_batch):
    prefixes = []
    batches = (scores.split(min(score_batch, 512)) for scores in (*visual, *lexical, prior))
    for crop, full, cropped_lex, full_lex, ties in zip(*batches, strict=True):
        streams = [crop, cropped_lex]
        if visual_weight:
            streams.append(full)
        if lexical_weight:
            streams.append(full_lex)
        candidates, valid = candidate_union(streams, 1280)
        appearance = source_votes(crop, full, banks[0], visual_weight, adaptive)
        language = source_votes(cropped_lex, full_lex, banks[1], lexical_weight, adaptive)
        prefixes.append(grow_density((appearance + language) * 32, ties, candidates, valid, graph, 256))
    return torch.cat(prefixes)


def sketch_view_graph_fusion_settings(queries, photos, contours, sketch_words, photo_words, maps, ink, score_batch):
    ink = F.normalize(ink.float(), dim=-1)
    transform = maps[(150, 50)]
    views = tuple(F.normalize(vectors @ transform, dim=-1) for vectors in (ink, queries))
    target, outlines = (F.normalize(vectors @ transform, dim=-1) for vectors in (photos, contours))
    paired, gallery = prepare_methods(views[0], target, outlines, ("local_residual",))["local_residual"]
    full, _ = prepare_methods(views[1], target, outlines, ("local_residual",))["local_residual"]
    gallery_words = gallery_word_codes(photos, photo_words)
    if photos.is_cuda:
        gallery_words = gallery_words.half().float()
    lexical = tuple(view_concepts(view[:, None], sketch_words, photo_words).squeeze(1) @ gallery_words.T
                    for view in (ink, queries))

    _, *iso_edges = jaccard_graph(gallery, 192)
    lexical_bank, *text_edges = jaccard_graph(gallery_words, 192)
    _, *rgb_edges = jaccard_graph(photos, 192)
    move, isolated = row_transition(len(gallery), (iso_edges, text_edges, rgb_edges))
    smooth = gallery_walk(move, isolated, gallery, 0.8, 4)
    iso_bank, *smooth_edges = jaccard_graph(smooth, 192)
    edges = (smooth_edges, text_edges, rgb_edges)
    graph = pack_affinity(len(gallery), edges)
    transition, dangling = column_stochastic(len(gallery), edges)
    banks = iso_bank, lexical_bank
    visual = tuple(iterative_feedback(query, smooth, 16, 2, 2.) @ smooth.T for query in (paired, full))

    def rows(appearance, words, visual_weight=0., lexical_weight=0., separate=False, adaptive=False):
        prior = standardize(appearance) + standardize(words)
        if separate:
            prefix = separate_view_prefixes(visual, lexical, prior, banks, graph,
                                            visual_weight, lexical_weight, adaptive, score_batch)
        else:
            prefix = density_prefixes(appearance, words, (appearance, words), prior, banks, graph,
                                      1280, score_batch, 256, 64.)
        tail = pagerank_tail(appearance, words, prior, banks, transition, dangling, score_batch)
        return partial(density_rows, tail, prefix, score_batch, None)

    yield "sketch-views-crop-reference", rows(visual[0], lexical[0])
    yield "sketch-views-full-control", rows(visual[1], lexical[1])
    visual_standard = tuple(standardize(scores) for scores in visual)
    lexical_standard = tuple(standardize(scores) for scores in lexical)
    for weight in (0.25, 0.5):
        appearance = visual_standard[0].lerp(visual_standard[1], weight)
        words = lexical_standard[0].lerp(lexical_standard[1], weight)
        yield f"sketch-views-early-joint-w{weight:g}", rows(appearance, words)
        yield f"sketch-views-late-joint-w{weight:g}", rows(appearance, words, weight, weight, separate=True)

    del appearance, words
    appearance = 0.5 * (visual_standard[0] + visual_standard[1])
    words = 0.5 * (lexical_standard[0] + lexical_standard[1])
    yield "sketch-views-late-joint-adaptive", rows(appearance, words, 0.5, 0.5, separate=True, adaptive=True)
    yield "sketch-views-late-visual-w0.5", rows(appearance, lexical[0], 0.5, 0., separate=True)
    yield "sketch-views-late-lexical-w0.5", rows(visual[0], words, 0., 0.5, separate=True)
