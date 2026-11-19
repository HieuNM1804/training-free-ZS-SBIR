"""Transfer STiTch and SuperGlue matching objectives to sketch retrieval."""

from functools import partial

import torch
from torch.nn import functional as F

from .global_methods import centered
from .manifold import build_graph
from .reranking import standardize
from .score_calibration import calibration_features, synthetic_domain_priors
from .semantic import gallery_memory, guided_term_scores, top_codes


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


def transport_rows(queries, gallery, offset, concepts, target_views, count, weight, batch_size, score=conditional_similarity):
    scale = 1 + offset.abs().max()
    for batch, words in zip(queries.split(batch_size), concepts.split(batch_size), strict=True):
        rows = (batch @ gallery.T - offset) / scale
        baseline, candidates = rows.topk(count, dim=1)
        available = target_views[candidates]
        pairwise = torch.matmul(words[:, None], available.transpose(-1, -2)).float()
        similarity = score(pairwise)
        combined = standardize(baseline) + weight * standardize(similarity)
        order = combined.argsort(dim=1, descending=True, stable=True)
        rows.scatter_(1, candidates.gather(1, order), baseline)
        yield from rows.cpu()


def prepare_transport(queries, photos, contours, sketch_words, photo_words):
    query, proxies, gallery = calibration_features(queries, photos, contours, sketch_words, photo_words)
    geometry = centered(photos, photos.mean(dim=0))
    prior = synthetic_domain_priors(proxies, gallery, geometry)["other-neighborhoods"]
    offset = 0.9 * (prior - prior.mean())
    memory = gallery_memory(photos, photo_words)
    hypotheses = []
    for batch in queries.split(256):
        scores = guided_term_scores(batch, photos, sketch_words, photo_words, memory, 0.3)
        ids, _ = top_codes(scores, 5)
        hypotheses.append(photo_words[ids])
    graph = build_graph(geometry, geometry, 320)
    smoothed = F.normalize(graph.diffuse(photos, 0.1, 4), dim=-1)
    targets = torch.stack((photos, smoothed, memory.gallery_words), dim=1)
    concepts = torch.cat(hypotheses)
    return query, gallery, offset, concepts, targets


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
