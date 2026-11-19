"""Pool independent sketch evidence over a FINCH hierarchy of gallery photos."""

from functools import partial

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import torch
from torch.nn import functional as F

from .gallery_diffusion import full_gallery_inputs, nearest_gallery
from .isoclip import contextual_scores, projected_feedback
from .reranking import standardize
from .union_density import density_rows


def first_neighbor_groups(neighbors):
    """Compute FINCH components using only the first-neighbor edges."""
    # Nodes sharing a neighbor are already connected through that neighbor.
    count = len(neighbors)
    adjacency = csr_matrix((np.ones(count, dtype=np.int8), (np.arange(count), neighbors.cpu().numpy())), shape=(count, count))
    _, groups = connected_components(adjacency, directed=False)
    return torch.from_numpy(groups).to(device=neighbors.device, dtype=torch.long)


def finch_levels(features, levels=3):
    members = torch.arange(len(features), device=features.device)
    representatives = F.normalize(features, dim=1)
    for _ in range(levels):
        if len(representatives) > 1:
            neighbors, _ = nearest_gallery(representatives, 1)
            members = first_neighbor_groups(neighbors[:, 0])[members]
            sizes = torch.bincount(members)
            totals = features.new_zeros((len(sizes), features.size(1))).index_add_(0, members, features)
            representatives = F.normalize(totals, dim=1)
        else:
            sizes = torch.bincount(members)
        yield members, sizes


def cluster_mass(probability, assignment, counts):
    totals = probability.new_zeros((len(probability), len(counts)))
    totals.index_add_(1, assignment, probability)
    return (totals / counts)[:, assignment]


def cluster_prior_rows(initial, levels, batch, mix, temperature):
    for scores in initial.split(batch):
        prior = (scores / temperature).softmax(dim=1)
        pooled = sum(cluster_mass(prior, members, counts) for members, counts in levels) / len(levels)
        mixture = prior.lerp(pooled, mix)
        bounded = (0.15 * (mixture * scores.size(1)).log()).tanh()
        yield from bounded.cpu()


def cluster_first_rows(initial, assignment, counts, score_batch):
    """Rank clusters by query likelihood and rank members by the original evidence."""
    for scores in initial.split(score_batch):
        prior = scores.softmax(dim=1)
        pooled = cluster_mass(prior, assignment, counts)
        yield from order_cluster_members(scores, pooled).cpu()


def order_cluster_members(scores, pooled):
    # Stable sorting by group likelihood retains the evidence order within each group.
    ordered = scores.argsort(dim=1, descending=True, stable=True)
    clusters = pooled.gather(1, ordered).argsort(dim=1, descending=True, stable=True)
    ranked = ordered.gather(1, clusters)
    values = torch.linspace(0.9, -0.9, scores.size(1), device=scores.device, dtype=scores.dtype)
    return torch.empty_like(scores).scatter_(1, ranked, values.expand_as(scores))


def cluster_prior_settings(queries, photos, contours, sketch_words, photo_words, maps, score_batch):
    query, visual = projected_feedback(queries, photos, contours, maps[(150, 50)])
    base, regional, _, _, views = full_gallery_inputs(queries, photos, contours, sketch_words, photo_words, score_batch)
    concepts = views[:, 2].float()
    context = contextual_scores(photos, concepts, base, regional, score_batch)
    initial = standardize(query @ visual.T) + standardize(context)
    del base, regional, context, views
    yield "finch-prior-control", partial(density_rows, initial, None, score_batch, None)

    visual_levels = tuple(finch_levels(visual, 2))
    concept_levels = tuple(finch_levels(concepts, 2))
    joint_levels = tuple(finch_levels(torch.cat((visual, concepts), dim=1)))
    print("FINCH clusters by level", {name: [len(counts) for _, counts in layers] for name, layers in (
        ("iso", visual_levels), ("concept", concept_levels), ("joint", joint_levels),
    )}, flush=True)
    for depth, level in enumerate(joint_levels, start=1):
        for mix in (.5, .85):
            yield f"finch-prior-joint-l{depth}-w{mix:g}", partial(cluster_prior_rows, initial, (level,), score_batch, mix, 1.)
    for name, levels in (
        ("iso-l2", visual_levels[-1:]),
        ("concept-l2", concept_levels[-1:]),
        ("separate-l2", (visual_levels[-1], concept_levels[-1])),
        ("joint-l2l3", joint_levels[1:]),
    ):
        yield f"finch-prior-{name}-w0.75", partial(cluster_prior_rows, initial, levels, score_batch, .75, 1.)
    yield "finch-prior-joint-l2l3-w0.75-t0.5", partial(cluster_prior_rows, initial, joint_levels[1:], score_batch, .75, .5)
    for name, (assignments, counts) in (("joint-l2", joint_levels[1]), ("joint-l3", joint_levels[2]), ("iso-l2", visual_levels[1])):
        yield f"finch-prior-first-{name}", partial(cluster_first_rows, initial, assignments, counts, score_batch)


def prototype_evidence(query, photos, assignments, count):
    totals = photos.new_zeros((count, photos.size(1))).index_add_(0, assignments, photos)
    return standardize(query @ F.normalize(totals, dim=1).T)


def prototype_rows(initial, group_scores, assignments, score_batch, weight):
    batches = zip(initial.split(score_batch), group_scores.split(score_batch), strict=True)
    for evidence, groups in batches:
        group_evidence = groups[:, assignments]
        if weight is None:
            fused = order_cluster_members(evidence, group_evidence)
        else:
            fused = (0.15 * (evidence + weight * group_evidence)).tanh()
        yield from fused.cpu()


def cluster_prototype_settings(queries, photos, contours, sketch_words, photo_words, maps, score_batch):
    transform = maps[(150, 50)]
    query, gallery = projected_feedback(queries, photos, contours, transform)
    base, regional, _, concepts, views = full_gallery_inputs(queries, photos, contours, sketch_words, photo_words, score_batch)
    words = views[:, 2].float()
    context = contextual_scores(photos, words, base, regional, score_batch)
    visual = query @ gallery.T
    initial = standardize(visual) + standardize(context)
    del base, regional, views, context, visual
    yield "finch-prototype-control", partial(density_rows, initial, None, score_batch, None)

    shapes = F.normalize(F.normalize(contours @ transform, dim=-1).mean(dim=1), dim=-1)
    center = shapes.mean(dim=0)
    shape_query = F.normalize(F.normalize(queries @ transform, dim=-1) - center, dim=-1)
    shape_gallery = F.normalize(shapes - center, dim=-1)
    text_query = F.normalize(concepts.float().mean(dim=1), dim=-1)
    levels = tuple(finch_levels(torch.cat((gallery, words), dim=1)))
    for depth, (assignments, counts) in enumerate(levels[1:], start=2):
        shape = prototype_evidence(shape_query, shape_gallery, assignments, len(counts))
        rgb = prototype_evidence(query, gallery, assignments, len(counts))
        lexical = prototype_evidence(text_query, words, assignments, len(counts))
        combined = (rgb + shape + lexical) / 3
        for name, evidence, weight in (
            ("combined-w0.75", combined, .75),
            ("combined-w2", combined, 2.),
            ("combined-first", combined, None),
        ):
            yield f"finch-prototype-l{depth}-{name}", partial(prototype_rows, initial, evidence, assignments, score_batch, weight)
        if depth == 3:
            for name, evidence in (("contour", shape), ("iso", rgb), ("concept", lexical)):
                yield f"finch-prototype-l3-{name}-w0.75", partial(prototype_rows, initial, evidence, assignments, score_batch, .75)
