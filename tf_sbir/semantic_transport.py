"""Transfer STiTch and SuperGlue matching objectives to sketch retrieval."""

from functools import partial

import torch
from torch.nn import functional as F

from .global_methods import centered
from .manifold import build_graph
from .reranking import standardize
from .score_calibration import calibration_features, synthetic_domain_priors
from .semantic import gallery_memory, guided_term_scores


def conditional_similarity(similarity, temperature=0.07):
    """Average the two negative cosine transport costs from STiTch, Eq. 6–8."""
    logits = similarity / temperature
    forward = (logits.softmax(dim=-1) * similarity).sum(dim=-1).mean(dim=-1)
    backward = (logits.softmax(dim=-2) * similarity).sum(dim=-2).mean(dim=-1)
    return (forward + backward) * 0.5


def dustbin_plan(similarity, dustbin, temperature=0.07, iterations=16):
    """Scale the entropy-regularized assignment with SuperGlue's Eq. 8–9 marginals."""
    m, n = similarity.shape[-2:]
    # Subtract the common dustbin logit so all padded entries have unit affinity.
    kernel = F.pad((similarity - dustbin) / temperature, (0, 1, 0, 1)).exp()
    rows = similarity.new_ones(m + 1)
    columns = similarity.new_ones(n + 1)
    rows[-1] = n
    columns[-1] = m
    plan = kernel
    for _ in range(iterations):
        plan = plan * (rows[:, None] / plan.sum(dim=-1, keepdim=True))
        plan = plan * (columns / plan.sum(dim=-2, keepdim=True))
    return plan


def partial_similarity(similarity, dustbin):
    plan = dustbin_plan(similarity, dustbin)
    matched = plan[..., :-1, :-1]
    return (matched * (similarity - dustbin)).sum(dim=(-2, -1)) / min(similarity.shape[-2:])


def rerank_transport_batch(rows, words, target_views, count, weight, score):
    baseline, candidates = rows.topk(count, dim=1)
    available = target_views[candidates]
    pairwise = torch.matmul(words[:, None], available.transpose(-1, -2)).float()
    combined = standardize(baseline) + weight * standardize(score(pairwise))
    order = combined.argsort(dim=1, descending=True, stable=True)
    return rows.scatter_(1, candidates.gather(1, order), baseline)


def transport_rows(queries, gallery, offset, concepts, target_views, count, weight, batch_size, score=conditional_similarity):
    scale = 1 + offset.abs().max()
    for batch, words in zip(queries.split(batch_size), concepts.split(batch_size), strict=True):
        rows = (batch @ gallery.T - offset) / scale
        yield from rerank_transport_batch(rows, words, target_views, count, weight, score).cpu()


def transport_components(queries, photos, contours, sketch_words, photo_words):
    query, proxies, gallery = calibration_features(queries, photos, contours, sketch_words, photo_words)
    geometry = centered(photos, photos.mean(dim=0))
    prior = synthetic_domain_priors(proxies, gallery, geometry)["other-neighborhoods"]
    offset = 0.9 * (prior - prior.mean())
    memory = gallery_memory(photos, photo_words)
    graph = build_graph(geometry, geometry, 320)
    smoothed = F.normalize(graph.diffuse(photos, 0.1, 4), dim=-1)
    targets = torch.stack((photos, smoothed, memory.gallery_words), dim=1)
    return query, gallery, offset, memory, targets


def concept_candidates(queries, photos, sketch_words, photo_words, memory, count):
    scores, indices = [], []
    for batch in queries.split(256):
        guided = guided_term_scores(batch, photos, sketch_words, photo_words, memory, 0.3)
        nearest = guided.topk(count)
        scores.append(nearest.values)
        indices.append(nearest.indices)
    return torch.cat(scores), torch.cat(indices)


def prepare_transport(queries, photos, contours, sketch_words, photo_words):
    query, gallery, offset, memory, targets = transport_components(queries, photos, contours, sketch_words, photo_words)
    _, indices = concept_candidates(queries, photos, sketch_words, photo_words, memory, 5)
    return query, gallery, offset, photo_words[indices], targets


def transport_settings(queries, photos, contours, sketch_words, photo_words):
    query, gallery, offset, concepts, targets = prepare_transport(queries, photos, contours, sketch_words, photo_words)
    if queries.is_cuda:
        concepts, targets = concepts.half(), targets.half()
    yield "transport-control", query, gallery, (offset, None)
    for count in (640, 1280):
        for weight in (0.15, 0.35):
            yield f"conditional-transport-k{count}-w{weight:g}", query, gallery, (offset, concepts, targets, count, weight)


def transport_view_settings(queries, photos, contours, sketch_words, photo_words, photo_augmentations):
    query, gallery, offset, concepts, reference = prepare_transport(queries, photos, contours, sketch_words, photo_words)
    dtype = torch.float16 if queries.is_cuda else torch.float32
    photos_views = F.normalize(photo_augmentations.float(), dim=-1).to(dtype)
    reference = reference.to(dtype)
    concepts = concepts.to(dtype)
    yield "transport-control", query, gallery, (offset, None)
    for k, weight in ((640, 0.35), (640, 0.70), (1280, 0.70)):
        yield f"conditional-transport-reference-k{k}-w{weight:g}", query, gallery, (offset, concepts, reference, k, weight)
    view_groups = (("crop", reference[:, :1]), ("crop-gallery", reference))
    for name, base in view_groups:
        target_views = torch.cat((base, photos_views), dim=1)
        for weight in (0.35, 0.70):
            yield f"conditional-transport-{name}-k640-w{weight:g}", query, gallery, (offset, concepts, target_views, 640, weight)


def partial_transport_settings(queries, photos, contours, sketch_words, photo_words, photo_augmentations):
    settings = transport_view_settings(queries, photos, contours, sketch_words, photo_words, photo_augmentations)
    for name, query, gallery, correction in settings:
        if name == "transport-control":
            yield name, query, gallery, correction
        if name == "conditional-transport-crop-gallery-k640-w0.7":
            yield name, query, gallery, correction
            for dustbin, weight in ((0.20, 0.7), (0.35, 0.7), (0.35, 1.2)):
                scorer = partial(partial_similarity, dustbin=dustbin)
                label = f"partial-transport-crop-gallery-k640-z{dustbin:g}-w{weight:g}"
                yield label, query, gallery, (*correction[:-1], weight, scorer)


def maximal_marginal_relevance(relevance, descriptors, count, relevance_weight):
    """Select concepts with the SIGIR 1998 relevance-minus-redundancy objective."""
    pairwise = descriptors @ descriptors.mT
    rows = torch.arange(len(relevance), device=relevance.device)
    selected = torch.zeros_like(relevance, dtype=torch.bool)
    current = relevance.argmax(dim=1)
    redundancy = torch.full_like(relevance, -torch.inf)
    positions = []
    for _ in range(count):
        positions.append(current)
        selected[rows, current] = True
        redundancy = torch.maximum(redundancy, pairwise[rows, current])
        objective = relevance_weight * relevance - (1 - relevance_weight) * redundancy
        current = objective.masked_fill(selected, -torch.inf).argmax(dim=1)
    return torch.stack(positions, dim=1)


def diversified_transport_settings(queries, photos, contours, sketch_words, photo_words, photo_augmentations):
    query, gallery, offset, memory, reference = transport_components(queries, photos, contours, sketch_words, photo_words)
    relevance, ids = concept_candidates(queries, photos, sketch_words, photo_words, memory, 64)
    dtype = torch.float16 if queries.is_cuda else torch.float32
    targets = torch.cat((reference, F.normalize(photo_augmentations.float(), dim=-1)), dim=1).to(dtype)
    score = partial(partial_similarity, dustbin=0.35)
    yield "partial-transport-top5-control", query, gallery, (offset, photo_words[ids[:, :5]].to(dtype), targets, 640, 1.2, score)
    for weight in (0.9, 0.96, 0.98):
        selections = []
        for rel, candidates in zip(relevance.split(256), ids.split(256), strict=True):
            best = maximal_marginal_relevance(rel, photo_words[candidates], 5, weight)
            selections.append(candidates.gather(1, best))
        concepts = photo_words[torch.cat(selections)].to(dtype)
        name = f"partial-transport-mmr5-l{weight:g}"
        yield name, query, gallery, (offset, concepts, targets, 640, 1.2, score)
