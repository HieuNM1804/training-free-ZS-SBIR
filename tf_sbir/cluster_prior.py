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
