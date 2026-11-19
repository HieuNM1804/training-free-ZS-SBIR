"""Query-specific fusion on gallery graphs weighted by neighborhood overlap."""

from functools import partial

import torch

from .gallery_diffusion import full_gallery_inputs, nearest_gallery
from .global_methods import centered
from .reranking import standardize


def neighborhood_overlap(left, right):
    """Jaccard similarity, with sorted neighbor indices on the right."""
    positions = torch.searchsorted(right, left).clamp_max(right.size(-1) - 1)
    shared = (right.gather(-1, positions) == left).sum(dim=-1).float()
    return shared / (left.size(-1) + right.size(-1) - shared)


def jaccard_graph(features, k):
    neighbors, _ = nearest_gallery(centered(features, features.mean(dim=0)), k)
    neighbors = neighbors.sort(dim=1).values
    source = torch.arange(len(neighbors), device=features.device).repeat_interleave(k)
    target = neighbors.flatten()
    sources, targets, weights = [], [], []
    for a, b in zip(source.split(16384), target.split(16384), strict=True):
        target_neighbors = neighbors[b]
        find = torch.searchsorted(target_neighbors, a[:, None].contiguous()).clamp_max(k - 1)
        mutual = target_neighbors.gather(1, find).squeeze(1).eq(a)
        a, b, target_neighbors = a[mutual], b[mutual], target_neighbors[mutual]
        sources.append(a)
        targets.append(b)
        weights.append(neighborhood_overlap(neighbors[a], target_neighbors))
    return neighbors, torch.cat(sources), torch.cat(targets), torch.cat(weights)


def column_stochastic(size, graphs):
    source, target, weights = (torch.cat(fields) for fields in zip(*graphs, strict=True))
    degree = weights.new_zeros(size).scatter_add_(0, source, weights)
    degree_inverse = degree.clamp_min(1e-12).reciprocal()
    edges = torch.stack((target, source))
    transition = torch.sparse_coo_tensor(edges, weights * degree_inverse[source], (size, size))
    return transition.coalesce().to_sparse_csr(), degree.eq(0)


def jaccard_seed(scores, neighbors):
    k = neighbors.size(1)
    top = scores.topk(k, dim=1)
    presence = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, top.indices, True)
    active = neighbors[top.indices].flatten(1)
    shared = presence.gather(1, active).view(len(scores), k, k).sum(dim=2).float()
    weights = shared / (2 * k - shared)
    return torch.zeros_like(scores).scatter_(1, top.indices, weights)


def personalized_walk(transition, dangling, seed, decay, iterations=16):
    """Compute Eq. 4 of Zhang et al. with restart at the sketch's candidate images."""
    initial = seed.T.contiguous()
    state = initial
    for _ in range(iterations):
        leaked = (state * dangling[:, None]).sum(dim=0)
        state = decay * torch.sparse.mm(transition, state) + ((1 - decay) + decay * leaked) * initial
    return state.T


def contextual_blocks(base, regional, evidence, rgb_bank, semantic_bank, transition, dangling, batch_size):
    for visual, local, fallback_score in zip(
        base.split(batch_size), regional.split(batch_size), evidence.split(batch_size), strict=True,
    ):
        mass = jaccard_seed(visual, rgb_bank) + jaccard_seed(local, semantic_bank)
        total = mass.sum(dim=1, keepdim=True)
        fallback = torch.zeros_like(mass).scatter_(1, fallback_score.argmax(dim=1, keepdim=True), 1)
        seed = torch.where(total > 0, mass / total.clamp_min(1e-12), fallback)
        stationary = personalized_walk(transition, dangling, seed, 0.85)
        yield standardize((stationary * base.size(1)).log1p())


def contextual_rows(evidence, walk, batch_size, weight):
    for initial, graph in zip(evidence.split(batch_size), walk.split(batch_size), strict=True):
        yield from (0.15 * (initial + weight * graph)).tanh().cpu()


def contextual_propagation(photos, words, base, regional, evidence, k, score_batch):
    rgb, *rgb_edges = jaccard_graph(photos, k)
    lexical, *word_edges = jaccard_graph(words, k)
    transition, dangling = column_stochastic(len(photos), (rgb_edges, word_edges))
    return torch.cat(list(contextual_blocks(
        base, regional, evidence, rgb, lexical, transition, dangling, min(256, score_batch),
    )))


def contextual_rank_settings(queries, photos, contours, sketch_words, photo_words, score_batch):
    base, regional, _, _, views = full_gallery_inputs(queries, photos, contours, sketch_words, photo_words, score_batch)
    evidence = standardize(base) + 0.25 * standardize(regional)
    yield "context-rank-control", lambda: (row for block in evidence.split(score_batch) for row in (0.15 * block).tanh().cpu())
    for k in (96, 192):
        walk = contextual_propagation(photos, views[:, 2].float(), base, regional, evidence, k, score_batch)
        for weight in (0.35, 0.7, 1.2):
            yield f"context-rank-k{k}-w{weight:g}", partial(contextual_rows, evidence, walk, score_batch, weight)
