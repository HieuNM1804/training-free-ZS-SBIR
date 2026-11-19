"""Keep direct contour evidence available after two-view RGB query expansion."""

from functools import partial

import torch
from torch.nn import functional as F

from .contextual_rank import column_stochastic, jaccard_graph, jaccard_seed
from .contour_query_expansion import pagerank_tail
from .global_methods import centered
from .reranking import standardize
from .sketch_view_graph_fusion import separate_view_prefixes, sketch_view_graph_inputs, source_votes
from .union_density import candidate_union, density_prefixes, density_rows, grow_density, pack_affinity


def mix_shape_votes(appearance, contour, fraction, adaptive):
    if adaptive:
        visual_support = (1 - fraction) * appearance.sum(dim=1, keepdim=True)
        shape_support = fraction * contour.sum(dim=1, keepdim=True)
        fraction = shape_support / (visual_support + shape_support).clamp_min(1e-8)
    return appearance.lerp(contour, fraction)


def direct_shape_prefixes(visual, lexical, combined, shape, prior, banks, graph, score_batch, *,
                          late, candidates_from_shape, shape_fraction, adaptive=False):
    prefixes = []
    inputs = (*visual, *lexical, *combined, shape, prior)
    batches = (scores.split(min(score_batch, 512)) for scores in inputs)
    for crop, full, crop_lex, full_lex, appearance, words, outlines, ties in zip(*batches, strict=True):
        if late:
            streams = [crop, full, crop_lex, full_lex]
            image_vote = source_votes(crop, full, banks[0], 0.5, True)
            word_vote = source_votes(crop_lex, full_lex, banks[1], 0.5, True)
        else:
            streams = [appearance, words]
            image_vote = jaccard_seed(appearance, banks[0])
            word_vote = jaccard_seed(words, banks[1])
        if candidates_from_shape:
            streams.append(outlines)
        if shape_fraction:
            shape_vote = jaccard_seed(outlines, banks[2])
            image_vote = mix_shape_votes(image_vote, shape_vote, shape_fraction, adaptive)
        candidates, valid = candidate_union(streams, 1280)
        prefixes.append(grow_density((image_vote + word_vote) * 32, ties, candidates, valid, graph, 256))
    return torch.cat(prefixes)


def shape_evidence_graph_fusion_settings(queries, photos, contours, sketch_words, photo_words, maps, ink, score_batch):
    visual, lexical, banks, edges, views, outlines = sketch_view_graph_inputs(
        queries, photos, contours, sketch_words, photo_words, maps, ink,
    )
    graph = pack_affinity(len(photos), edges)
    transition, dangling = column_stochastic(len(photos), edges)
    visual_standard = tuple(standardize(scores) for scores in visual)
    lexical_standard = tuple(standardize(scores) for scores in lexical)
    proxies = F.normalize(outlines.mean(dim=1), dim=-1)
    center = proxies.mean(dim=0)
    proxy_vectors = centered(proxies, center)
    contour_queries = (centered(views[0], center) + centered(views[1], center)) * 0.5
    shape = contour_queries @ proxy_vectors.T
    contour_bank = jaccard_graph(proxy_vectors, 192)[0]
    joint = (("joint-w0.25", True, 0.25, False), ("joint-w0.5", True, 0.5, False))
    late_settings = (
        ("candidate", True, 0., False), ("vote-w0.25", False, 0.25, False),
        ("joint-w0.125", True, 0.125, False), *joint, ("joint-adaptive", True, 0.5, True),
    )
    for late, label, settings in ((True, "late", late_settings), (False, "early", joint)):
        if late:
            appearance = 0.5 * (visual_standard[0] + visual_standard[1])
            words = 0.5 * (lexical_standard[0] + lexical_standard[1])
            prior = standardize(appearance) + standardize(words)
            prefix = separate_view_prefixes(visual, lexical, prior, banks, graph, 0.5, 0.5, True, score_batch)
        else:
            appearance = visual_standard[0].lerp(visual_standard[1], 0.5)
            words = lexical_standard[0].lerp(lexical_standard[1], 0.5)
            prior = standardize(appearance) + standardize(words)
            prefix = density_prefixes(appearance, words, (appearance, words), prior, banks, graph,
                                      1280, score_batch, 256, 64.)
        tail = pagerank_tail(appearance, words, prior, banks, transition, dangling, score_batch)
        yield f"shape-evidence-{label}-reference", partial(density_rows, tail, prefix, score_batch, None)
        for name, expand, fraction, adaptive in settings:
            prefix = direct_shape_prefixes(
                visual, lexical, (appearance, words), shape, prior, (*banks, contour_bank), graph, score_batch,
                late=late, candidates_from_shape=expand, shape_fraction=fraction, adaptive=adaptive,
            )
            yield f"shape-evidence-{label}-{name}", partial(density_rows, tail, prefix, score_batch, None)
