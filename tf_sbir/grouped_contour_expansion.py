"""Query expansion from separate RGB–contour neighborhoods of a sketch."""

from functools import partial

import torch
from torch.nn import functional as F

from .contextual_rank import column_stochastic, jaccard_graph, jaccard_seed
from .contour_query_expansion import gallery_word_codes, pagerank_tail
from .global_methods import centered, prepare_methods
from .manifold import iterative_feedback
from .reranking import standardize
from .union_density import candidate_union, density_prefixes, density_rows, grow_density, pack_affinity
from .view_hypotheses import view_concepts


def disjoint_support(candidates, affinity, scores, graph, group_size, group_count):
    """Grow one dense group at a time, excluding its members from subsequent groups."""
    available = torch.ones_like(candidates, dtype=torch.bool)
    groups = []
    for _ in range(group_count):
        selected = grow_density(affinity, scores, candidates, available, graph, group_size)
        groups.append(selected)
        available &= ~candidates[:, :, None].eq(selected[:, None]).any(dim=-1)
    return torch.stack(groups, dim=1)


def grouped_support(contour_scores, lexical, contour_bank, lexical_bank, graph, score_batch):
    proposals, joint_proposals, support_scores, reference_scores = [], [], [], []
    for shape, words in zip(contour_scores.split(score_batch), lexical.split(score_batch), strict=True):
        nearest = shape.topk(64, dim=1).indices
        contour_edge = jaccard_seed(shape, contour_bank)
        lexical_edge = jaccard_seed(words, lexical_bank)
        joint_edge = 0.5 * (contour_edge + lexical_edge)
        groups = disjoint_support(nearest, 4. * contour_edge, shape, graph, 8, 3)
        joint = grow_density(4. * joint_edge, standardize(shape) + standardize(words), nearest,
                             torch.ones_like(nearest, dtype=torch.bool), graph, 8)
        quality = joint_edge.gather(1, groups.flatten(1)).view_as(groups).mean(dim=-1)
        proposals.append(groups)
        joint_proposals.append(joint)
        support_scores.append(quality)
        reference_scores.append(joint_edge.gather(1, nearest[:, :8]).mean(dim=1, keepdim=True))
    return tuple(torch.cat(chunks) for chunks in (proposals, joint_proposals, support_scores, reference_scores))


def expand_groups(queries, gallery, contours, groups, contour_scores, score_batch):
    expanded = []
    for vector, ids, scores in zip(queries.split(score_batch), groups.split(score_batch), contour_scores.split(score_batch), strict=True):
        appearance = gallery[ids]
        outline = contours[ids]
        weights = (scores.gather(1, ids.flatten(1)).view_as(ids) / 0.07).softmax(dim=-1)
        shift = ((appearance - outline) * weights[..., None]).sum(dim=2)
        expanded.append(F.normalize(vector[:, None] + 0.5 * shift, dim=-1))
    candidates = torch.cat(expanded)
    refined = iterative_feedback(candidates.flatten(0, 1), gallery, 16, 2, 2.)
    return refined.unflatten(0, candidates.shape[:2])


def fuse_group_prefixes(embeddings, gallery, lexical, banks, graph, score_batch):
    prefixes = []
    for batch, words in zip(embeddings.split(score_batch), lexical.split(score_batch), strict=True):
        visual = (batch @ gallery.T).unbind(dim=1)
        ids, valid = candidate_union((*visual, words), 1280)
        visual_votes = torch.stack([jaccard_seed(view, banks[0]) for view in visual]).mean(dim=0)
        votes = 32. * (visual_votes + jaccard_seed(words, banks[1]))
        ties = torch.stack([standardize(view) for view in visual]).mean(dim=0) + standardize(words)
        prefixes.append(grow_density(votes, ties, ids, valid, graph, 256))
    return torch.cat(prefixes)


def grouped_contour_settings(_queries, photos, contours, sketch_words, photo_words, maps, ink, score_batch):
    ink = F.normalize(ink.float(), dim=-1)
    transform = maps[(150, 50)]
    sketch, target, outlines = (F.normalize(vectors @ transform, dim=-1) for vectors in (ink, photos, contours))
    cues = prepare_methods(sketch, target, outlines, ("centered", "local_residual"))
    query, gallery = cues["centered"]
    paired, _ = cues["local_residual"]
    contour = F.normalize(outlines.mean(dim=1), dim=-1)
    contour = centered(contour, contour.mean(dim=0))
    shape = query @ contour.T

    words = gallery_word_codes(photos, photo_words)
    if photos.is_cuda:
        words = words.half().float()
    lexical = view_concepts(ink[:, None], sketch_words, photo_words).squeeze(1) @ words.T
    language = standardize(lexical)
    iso_bank, *iso_edges = jaccard_graph(gallery, 192)
    lexical_bank, *lexical_edges = jaccard_graph(words, 192)
    _, *rgb_edges = jaccard_graph(photos, 192)
    banks = (iso_bank, lexical_bank)
    graphs = (iso_edges, lexical_edges, rgb_edges)
    graph = pack_affinity(len(photos), graphs)
    transition, dangling = column_stochastic(len(photos), graphs)

    reference = iterative_feedback(paired, gallery, 16, 2, 2.)
    visual = reference @ gallery.T
    prior = standardize(visual) + language
    ref_prefix = density_prefixes(visual, lexical, (visual, lexical), prior, banks, graph, 1280, score_batch, 256, 64.)
    tail = pagerank_tail(visual, lexical, prior, banks, transition, dangling, score_batch)
    yield "contour-modes-reference", partial(density_rows, tail, ref_prefix, score_batch, None)
    del visual, prior

    contour_bank = jaccard_graph(contour, 192)[0]
    groups, joint, quality, reference_quality = grouped_support(shape, lexical, contour_bank, lexical_bank, graph, score_batch)
    # A matched three-proposal control partitions the rank order into bands of eight.
    bands = shape.topk(24, dim=1).indices.unflatten(1, (3, 8))[:, 1:]
    referrals = torch.cat((groups, joint[:, None], bands), dim=1)
    embeddings = expand_groups(query, gallery, contour, referrals, shape, score_batch)
    del shape
    separate = embeddings[:, :3]
    choices = quality.argmax(dim=1)
    available = torch.cat((reference[:, None], separate), dim=1)
    choices_with_ref = torch.cat((reference_quality, quality), dim=1).argmax(dim=1)
    indices = torch.arange(len(query), device=query.device)

    shape_groups = tuple((f"group-{i+1}", mode) for i, mode in enumerate(separate.unbind(dim=1)))
    controls = tuple((f"rank-band-{i+2}", mode) for i, mode in enumerate(embeddings[:, 4:].unbind(dim=1)))
    variants = (*shape_groups, *controls,
        ("joint", embeddings[:, 3]),
        ("selected", separate[indices, choices]),
        ("selected-with-reference", available[indices, choices_with_ref]),
    )
    for name, embedding in variants:
        visual = embedding @ gallery.T
        prior = standardize(visual) + language
        prefix = density_prefixes(visual, lexical, (visual, lexical), prior, banks, graph, 1280, score_batch, 256, 64.)
        yield f"contour-modes-{name}", partial(density_rows, tail, prefix, score_batch, None)
    del visual, prior
    prefix = fuse_group_prefixes(separate, gallery, lexical, banks, graph, min(384, score_batch))
    yield "contour-modes-fused", partial(density_rows, tail, prefix, score_batch, None)
