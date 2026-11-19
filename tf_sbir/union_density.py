"""Greedily grow query-anchored dense subgraphs on complementary candidate lists."""

from functools import partial

import torch
from torch.nn import functional as F

from .contextual_rank import jaccard_graph, jaccard_seed
from .gallery_diffusion import diffusion_blocks, full_gallery_inputs, mutual_transition, regional_cosines
from .isoclip import contextual_scores, local_feedback, projected_feedback
from .reranking import standardize


def pack_affinity(size, graphs):
    """Merge overlapping gallery edges and pack incoming updates by source."""
    source, target, weights = (torch.cat(parts) for parts in zip(*graphs, strict=True))
    affinity = torch.sparse_coo_tensor(torch.stack((source, target)), weights / len(graphs), (size, size)).coalesce()
    source, target = affinity.indices()
    lengths = torch.bincount(source, minlength=size)
    starts = lengths.cumsum(0) - lengths
    offsets = torch.arange(len(source), device=source.device) - starts[source]
    width = int(lengths.max())
    neighbors = torch.zeros((size, width), dtype=torch.long, device=source.device)
    strengths = weights.new_zeros((size, width))
    neighbors[source, offsets] = target
    strengths[source, offsets] = affinity.values()
    return neighbors, strengths


def candidate_union(score_lists, depth):
    ids = torch.cat([scores.topk(depth, dim=1).indices for scores in score_lists], dim=1).sort(dim=1).values
    unique = F.pad(ids[:, 1:].ne(ids[:, :-1]), (1, 0), value=True)
    return ids, unique


def grow_density(affinity, ties, candidates, valid, graph, steps):
    """Rank Eq. 5 of Zhang et al. by greedy marginal edge weight."""
    neighbor_ids, strengths = graph
    batch, width = candidates.shape
    positions = torch.arange(width, device=candidates.device).expand_as(candidates)
    # One extra slot absorbs scatter updates to images outside this sketch's union.
    inverse = torch.full_like(affinity, -1, dtype=torch.long)
    inverse.scatter_reduce_(1, candidates, positions.masked_fill(~valid, -1), reduce="amax", include_self=True)
    inverse.masked_fill_(inverse < 0, width)
    gains = affinity.gather(1, candidates) + 1e-5 * ties.gather(1, candidates)
    gains = F.pad(gains.masked_fill(~valid, -torch.inf), (0, 1), value=-torch.inf)
    selected = torch.empty((batch, steps), dtype=torch.long, device=candidates.device)
    for step in range(steps):
        best = gains.argmax(dim=1, keepdim=True)
        photo = candidates.gather(1, best)
        selected[:, step:step + 1] = photo
        gains.scatter_(1, best, -torch.inf)
        neighbors = inverse.gather(1, neighbor_ids[photo.squeeze(1)])
        gains.scatter_add_(1, neighbors, strengths[photo.squeeze(1)])
    return selected


def density_prefixes(iso, context, streams, initial, iso_bank, lexical_bank, graph, depth, score_batch):
    prefixes = []
    batch_size = min(score_batch, 128)
    batches = (scores.split(batch_size) for scores in (iso, context, initial, *streams))
    for visual, semantic, tied, *cues in zip(*batches, strict=True):
        candidates, valid = candidate_union(cues, depth)
        votes = (jaccard_seed(visual, iso_bank) + jaccard_seed(semantic, lexical_bank)) * 0.5
        prefixes.append(grow_density(votes, tied, candidates, valid, graph, min(256, depth)))
    return torch.cat(prefixes)


def density_rows(initial, selected, score_batch, weight):
    if selected is None:
        yield from (row for batch in initial.split(score_batch) for row in (0.15 * batch).tanh().cpu())
        return
    discount = torch.linspace(1, 1 / selected.size(1), selected.size(1), device=initial.device)
    for evidence, prefix in zip(initial.split(score_batch), selected.split(score_batch), strict=True):
        if weight is None:
            bounded = (0.15 * evidence).tanh()
            ceiling = bounded.amax(dim=1, keepdim=True)
            value = ceiling + (1 - ceiling) * (0.1 + 0.8 * discount)
            scores = bounded.scatter(1, prefix, value)
        else:
            promoted = evidence.scatter_add(1, prefix, (weight * discount).expand_as(prefix))
            scores = (0.15 * promoted).tanh()
        yield from scores.cpu()


def density_settings(queries, photos, contours, sketch_words, photo_words, maps, rendered, isolated, score_batch):
    query, iso_gallery = projected_feedback(queries, photos, contours, maps[(150, 50)])
    iso = query @ iso_gallery.T
    base, region, native, concepts, views = full_gallery_inputs(queries, photos, contours, sketch_words, photo_words, score_batch)
    words = views[:, 2].float()
    context = contextual_scores(photos, words, base, region, score_batch)
    initial = standardize(iso) + standardize(context)
    del region
    yield "union-density-control", partial(density_rows, initial, None, score_batch, None)

    geometry = F.normalize(isolated.to(device=photos.device, dtype=torch.float32), dim=-1).mean(dim=1)
    augmented = torch.cat((views, geometry[:, None].to(views.dtype)), dim=1)
    regional = regional_cosines(concepts, augmented, min(score_batch, 256))
    joint = torch.cat((native * (0.65 ** 0.5), F.normalize(geometry, dim=-1) * (0.35 ** 0.5)), dim=1)
    diffused = torch.cat(list(diffusion_blocks(base, regional, mutual_transition(joint, 160), score_batch, 0.9, 320, "regional")))
    del base, regional
    sq, sp = local_feedback(queries, photos, torch.cat((contours, rendered), dim=1))
    shape = sq @ sp.T

    iso_bank, *iso_edges = jaccard_graph(iso_gallery, 192)
    lexical_bank, *lexical_edges = jaccard_graph(words, 192)
    _, *rgb_edges = jaccard_graph(photos, 192)
    graph = pack_affinity(len(photos), (iso_edges, lexical_edges, rgb_edges))
    for group, streams in (("pair", (iso, context)), ("four", (iso, context, diffused, shape))):
        selected = density_prefixes(iso, context, streams, initial, iso_bank, lexical_bank, graph, 1280, score_batch)
        for weight in (0.8, 1.6, None):
            tag = "hard" if weight is None else f"w{weight:g}"
            yield f"union-density-{group}-{tag}", partial(density_rows, initial, selected, score_batch, weight)
