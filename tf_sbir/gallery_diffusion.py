"""Gallery-only score fusion and locally constrained Iscen diffusion."""

from functools import partial

import torch
from torch.nn import functional as F

from .global_methods import reciprocal_ranks
from .reranking import standardize
from .semantic_transport import conditional_similarity, partial_similarity, prepare_transport, rerank_transport_batch


def nearest_gallery(features, neighbors):
    size = len(features)
    node_blocks, similarity_blocks = [], []
    for start in range(0, size, 512):
        rows = torch.arange(start, min(start + 512, size), device=features.device)
        scores = features[rows] @ features.T
        scores[torch.arange(len(rows), device=features.device), rows] = -torch.inf
        top = scores.topk(neighbors, dim=1)
        node_blocks.append(top.indices)
        similarity_blocks.append(top.values.clamp_min(0))
    return torch.cat(node_blocks), torch.cat(similarity_blocks)


def normalized_transition(size, sources, targets, weights):
    degree = weights.new_zeros(size).scatter_add_(0, sources, weights).clamp_min(1e-12)
    inverse = degree.rsqrt()
    normalized = weights * inverse[sources] * inverse[targets]
    return torch.sparse_coo_tensor(torch.stack((sources, targets)), normalized, (size, size)).coalesce().to_sparse_csr()


def mutual_transition(features, neighbors):
    """Return the symmetric, degree-normalized affinity of mutual neighbors."""
    size = len(features)
    neighbor_ids, similarities = nearest_gallery(features, neighbors)
    targets = neighbor_ids.flatten()
    weights = similarities.flatten().pow(3)
    sources = torch.arange(size, device=features.device).repeat_interleave(neighbors)
    edge_keys = (sources * size + targets).sort().values
    reversed_keys = targets * size + sources
    found = torch.searchsorted(edge_keys, reversed_keys).clamp_max(len(edge_keys) - 1)
    keep = edge_keys[found] == reversed_keys
    return normalized_transition(size, sources[keep], targets[keep], weights[keep])


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


def diffusion_blocks(base, regional, transition, batch_size, alpha, seed_count, fusion):
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
        yield (0.15 * (evidence + 0.5 * standardize(log_mass))).tanh()


def diffusion_rows(*args):
    for block in diffusion_blocks(*args):
        yield from block.cpu()


def full_gallery_inputs(queries, photos, contours, sketch_words, photo_words, score_batch):
    query, gallery, offset, concepts, target_views = prepare_transport(queries, photos, contours, sketch_words, photo_words)
    dtype = torch.float16 if queries.is_cuda else torch.float32
    concepts, target_views = concepts.to(dtype), target_views.to(dtype)
    regional = regional_cosines(concepts, target_views, min(score_batch, 256))
    base = (query @ gallery.T - offset) / (1 + offset.abs().max())
    return base, regional, gallery, concepts, target_views


def gallery_diffusion_settings(queries, photos, contours, sketch_words, photo_words, score_batch):
    base, regional, gallery, _, _ = full_gallery_inputs(queries, photos, contours, sketch_words, photo_words, score_batch)
    yield "full-gallery-control", lambda: (row for batch in base.split(score_batch) for row in batch.cpu())
    for weight in (0.25, 0.5):
        yield f"full-gallery-transport-w{weight:g}", partial(linear_fusion_rows, base, regional, score_batch, weight)
        yield f"full-gallery-rrf-w{weight:g}", partial(fused_rank_rows, base, regional, score_batch, weight)
    transition = mutual_transition(gallery, 160)
    for alpha, seeds in ((0.7, 128), (0.9, 320)):
        for fusion in ("visual", "regional"):
            name = f"mutual-diffusion-{fusion}-a{alpha:g}-seed{seeds}"
            yield name, partial(diffusion_rows, base, regional, transition, score_batch, alpha, seeds, fusion)


def two_stage_rows(base, regional, concepts, crops, shortlist, refine_weight, region_weight, batch_limit):
    score = partial(partial_similarity, dustbin=0.35)
    bytes_per_query = shortlist * crops.size(1) * crops.size(2) * crops.element_size()
    batch_size = min(batch_limit, max(1, 1024 * 2**20 // bytes_per_query))
    for visual, local, words in zip(base.split(batch_size), regional.split(batch_size), concepts.split(batch_size), strict=True):
        ranked = (0.15 * (standardize(visual) + region_weight * standardize(local))).tanh()
        yield from rerank_transport_batch(ranked, words, crops, shortlist, refine_weight, score).cpu()


def diffusion_then_transport_rows(base, regional, transition, concepts, crops, batch_limit, shortlist, refine_weight):
    score = partial(partial_similarity, dustbin=0.35)
    blocks = diffusion_blocks(base, regional, transition, batch_limit, 0.9, 320, "regional")
    for ranked, words in zip(blocks, concepts.split(min(batch_limit, 256)), strict=True):
        for scores, part in zip(ranked.split(64), words.split(64), strict=True):
            yield from rerank_transport_batch(scores, part, crops, shortlist, refine_weight, score).cpu()


def two_stage_settings(queries, photos, contours, sketch_words, photo_words, photo_augmentations, score_batch):
    base, regional, gallery, concepts, target_views = full_gallery_inputs(queries, photos, contours, sketch_words, photo_words, score_batch)
    crops = torch.cat((target_views, F.normalize(photo_augmentations.float(), dim=-1).to(target_views.dtype)), dim=1)
    for region_weight, shortlist, refine_weight in ((0.15, 640, 0.7), (0.25, 640, 1.2), (0.5, 1280, 0.7), (0.5, 640, 1.2)):
        name = f"full-gallery-then-partial-k{shortlist}-region{region_weight:g}-w{refine_weight:g}"
        yield name, partial(two_stage_rows, base, regional, concepts, crops, shortlist, refine_weight, region_weight, score_batch)
    transition = mutual_transition(gallery, 160)
    for shortlist, weight in ((640, 0.7), (1280, 0.7), (640, 1.2)):
        name = f"diffusion-then-partial-k{shortlist}-w{weight:g}"
        yield name, partial(diffusion_then_transport_rows, base, regional, transition, concepts, crops, score_batch, shortlist, weight)
