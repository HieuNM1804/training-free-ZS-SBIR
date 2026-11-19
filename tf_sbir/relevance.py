"""Refine binary gallery relevance with LAME's per-node posterior update."""

from functools import partial

import torch

from .gallery_diffusion import full_gallery_inputs, mutual_transition
from .isoclip import contextual_scores, projected_feedback
from .reranking import standardize


def refine_relevance(prior, transition, strength, iterations=8, squared=False):
    """Apply the log-odds form of Eq. 5 to independent query columns."""
    unary = prior.T.contiguous()
    logits = unary
    for _ in range(iterations):
        support = torch.sparse.mm(transition, (0.5 * logits).tanh())
        if squared:
            support = torch.sparse.mm(transition, support)
        logits = unary + strength * support
    return logits.T


def relevance_rows(initial, transition, score_batch, cutoff, strength, squared):
    for evidence in initial.split(score_batch):
        if transition is not None:
            boundary = evidence.topk(min(cutoff, evidence.size(1)), dim=1).values[:, -1:]
            prior = evidence - boundary
            evidence = refine_relevance(prior, transition, strength, squared=squared) + boundary
        yield from (0.15 * evidence).tanh().cpu()


def relevance_settings(queries, photos, contours, sketch_words, photo_words, maps, score_batch):
    query, iso_gallery = projected_feedback(queries, photos, contours, maps[(150, 50)])
    base, regional, _, _, views = full_gallery_inputs(queries, photos, contours, sketch_words, photo_words, score_batch)
    words = views[:, 2].float()
    context = contextual_scores(photos, words, base, regional, score_batch)
    initial = standardize(query @ iso_gallery.T) + standardize(context)
    del base, regional, context, views
    yield "lame-relevance-control", partial(relevance_rows, initial, None, score_batch, 0, 0, False)

    neighbors = min(128, len(photos) - 1)
    transitions = [mutual_transition(view, neighbors).to_sparse_coo() for view in (iso_gallery, photos, words)]
    iso = transitions[0].to_sparse_csr()
    fused = ((transitions[0] + transitions[1] + transitions[2]).coalesce() / len(transitions)).to_sparse_csr()
    del transitions
    for name, transition, cutoff, strength, squared in (
        ("iso-k384-w1", iso, 384, 1., False),
        ("fused-k128-w1", fused, 128, 1., False),
        ("fused-k384-w0.5", fused, 384, .5, False),
        ("fused-k384-w1", fused, 384, 1., False),
        ("fused-k384-w2", fused, 384, 2., False),
        ("fused-k768-w1", fused, 768, 1., False),
        ("squared-k384-w1", fused, 384, 1., True),
        ("squared-k384-w2", fused, 384, 2., True),
    ):
        yield f"lame-relevance-{name}", partial(relevance_rows, initial, transition, score_batch, cutoff, strength, squared)
