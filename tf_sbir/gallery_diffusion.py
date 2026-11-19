"""Gallery-only score fusion and locally constrained Iscen diffusion."""

from functools import partial

import torch

from .global_methods import reciprocal_ranks
from .reranking import standardize
from .semantic_transport import conditional_similarity, prepare_transport


def mutual_transition(features, neighbors):
    """Return the symmetric, degree-normalized affinity of mutual neighbors."""
    size = len(features)
    node_blocks, similarity_blocks = [], []
    for start in range(0, size, 512):
        rows = torch.arange(start, min(start + 512, size), device=features.device)
        scores = features[rows] @ features.T
        scores[torch.arange(len(rows), device=features.device), rows] = -torch.inf
        top = scores.topk(neighbors, dim=1)
        node_blocks.append(top.indices)
        similarity_blocks.append(top.values.clamp_min(0).pow(3))
    targets = torch.cat(node_blocks).flatten()
    weights = torch.cat(similarity_blocks).flatten()
    sources = torch.arange(size, device=features.device).repeat_interleave(neighbors)
    edge_keys = (sources * size + targets).sort().values
    reversed_keys = targets * size + sources
    found = torch.searchsorted(edge_keys, reversed_keys).clamp_max(len(edge_keys) - 1)
    keep = edge_keys[found] == reversed_keys
    sources, targets, weights = sources[keep], targets[keep], weights[keep]
    degree = weights.new_zeros(size).scatter_add_(0, sources, weights).clamp_min(1e-12)
    inverse = degree.rsqrt()
    normalized = weights * inverse[sources] * inverse[targets]
    return torch.sparse_coo_tensor(torch.stack((sources, targets)), normalized, (size, size)).coalesce().to_sparse_csr()


def solve_diffusion(transition, seed, alpha, iterations=16):
    """Use the fixed-point iteration of Iscen et al. Eq. 3."""
    initial = seed.T.contiguous()
    restart = (1 - alpha) * initial
    solution = initial
    for _ in range(iterations):
        solution = torch.sparse.mm(transition, solution).mul_(alpha).add_(restart)
    return solution.T


def regional_cosines(concepts, target_views, batch_size):
    """Score the entire index with conditional transport, in gallery blocks."""
    rows = []
    for query in concepts.split(batch_size):
        count, words, width = query.shape
        blocks = []
        for gallery in target_views.split(2048):
            views = gallery.size(1)
            affinities = query.reshape(-1, width) @ gallery.flatten(0, 1).T
            affinities = affinities.reshape(count, words, len(gallery), views).transpose(1, 2)
            blocks.append(conditional_similarity(affinities.float()))
        rows.append(torch.cat(blocks, dim=1))
    return torch.cat(rows)


def fused_rank_rows(base, regional, batch_size, weight):
    for visual, local in zip(base.split(batch_size), regional.split(batch_size), strict=True):
        yield from (reciprocal_ranks(visual) + weight * reciprocal_ranks(local)).cpu()


def linear_fusion_rows(base, regional, batch_size, weight):
    for visual, local in zip(base.split(batch_size), regional.split(batch_size), strict=True):
        yield from (0.15 * (standardize(visual) + weight * standardize(local))).tanh().cpu()


def diffusion_rows(base, regional, transition, batch_size, alpha, seed_count, fusion):
    score_batch = min(batch_size, 256)
    for visual, local in zip(base.split(score_batch), regional.split(score_batch), strict=True):
        if fusion == "regional":
            evidence = standardize(visual) + 0.5 * standardize(local)
        else:
            evidence = standardize(visual)
        values, indices = evidence.topk(seed_count, dim=-1)
        seed = torch.zeros_like(evidence).scatter_(1, indices, (values * 0.6).softmax(dim=1))
        propagated = solve_diffusion(transition, seed, alpha)
        log_mass = (propagated.clamp_min(0) * base.size(1)).log1p()
        yield from (0.15 * (evidence + 0.5 * standardize(log_mass))).tanh().cpu()


def gallery_diffusion_settings(queries, photos, contours, sketch_words, photo_words, score_batch):
    query, gallery, offset, concepts, target_views = prepare_transport(queries, photos, contours, sketch_words, photo_words)
    dtype = torch.float16 if queries.is_cuda else torch.float32
    regional = regional_cosines(concepts.to(dtype), target_views.to(dtype), min(score_batch, 256))
    base = (query @ gallery.T - offset) / (1 + offset.abs().max())
    yield "full-gallery-control", lambda: (row for batch in base.split(score_batch) for row in batch.cpu())
    for weight in (0.25, 0.5):
        yield f"full-gallery-transport-w{weight:g}", partial(linear_fusion_rows, base, regional, score_batch, weight)
        yield f"full-gallery-rrf-w{weight:g}", partial(fused_rank_rows, base, regional, score_batch, weight)
    transition = mutual_transition(gallery, 160)
    for alpha, seeds in ((0.7, 128), (0.9, 320)):
        for fusion in ("visual", "regional"):
            name = f"mutual-diffusion-{fusion}-a{alpha:g}-seed{seeds}"
            yield name, partial(diffusion_rows, base, regional, transition, score_batch, alpha, seeds, fusion)
