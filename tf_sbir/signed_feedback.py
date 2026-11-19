"""Transfer softmax-weighted positive and negative Rocchio feedback to SBIR."""

from functools import partial

import torch

from .gallery_diffusion import full_gallery_inputs
from .isoclip import contextual_scores, projected_feedback
from .reranking import standardize
from .union_density import density_prefixes, density_rows, gallery_density_graph


def feedback_vectors(evidence, candidates, gallery, temperature):
    """Return the two Rocchio vectors from the same high-ranking candidates."""
    weights = (evidence.gather(1, candidates) / temperature).softmax(dim=1).unsqueeze(1)
    vectors = gallery[candidates]
    positive = (weights @ vectors).squeeze(1)
    negative = ((1 - weights) @ vectors).squeeze(1) / (candidates.size(1) - 1)
    return positive, negative


def feedback_blocks(initial, shortlist, gallery, score_batch, temperature, negative_weight):
    batches = zip(initial.split(score_batch), shortlist.split(score_batch), strict=True)
    for evidence, candidates in batches:
        positive, negative = feedback_vectors(evidence, candidates, gallery, temperature)
        yield standardize((positive - negative_weight * negative) @ gallery.T)


def signed_rows(initial, feedback, score_batch, amount):
    for evidence, correction in zip(initial.split(score_batch), feedback.split(score_batch), strict=True):
        yield from (0.15 * (evidence + amount * correction)).tanh().cpu()


def signed_feedback_settings(queries, photos, contours, sketch_words, photo_words, maps, score_batch):
    query, gallery = projected_feedback(queries, photos, contours, maps[(150, 50)])
    visual = query @ gallery.T
    base, region, _, _, views = full_gallery_inputs(queries, photos, contours, sketch_words, photo_words, score_batch)
    words = views[:, 2].float()
    context = contextual_scores(photos, words, base, region, score_batch)
    initial = standardize(visual) + standardize(context)
    del base, region, views
    yield "signed-feedback-control", partial(density_rows, initial, None, score_batch, None)

    banks, graph = gallery_density_graph(photos, gallery, words)
    graph_shortlist = density_prefixes(visual, context, (visual, context), initial, banks, graph, 1280, score_batch, 256, 16.)
    yield "signed-feedback-graph-control", partial(density_rows, initial, graph_shortlist, score_batch, None)
    del visual, context, banks, graph
    baseline_shortlist = initial.topk(64, dim=1).indices
    joint_gallery = torch.cat((gallery, words), dim=1) * (0.5 ** 0.5)
    for name, shortlist in (("top", baseline_shortlist), ("graph", graph_shortlist[:, :64])):
        for size, temperature, negative_weight in ((16, .5, 1.), (64, .5, 0.), (64, .5, 1.), (64, .25, 1.)):
            feedback = torch.cat(list(feedback_blocks(initial, shortlist[:, :size], joint_gallery, score_batch, temperature, negative_weight)))
            for amount in (.5, 1.25):
                label = f"{name}-k{size}-t{temperature:g}-neg{negative_weight:g}-w{amount:g}"
                yield f"signed-feedback-{label}", partial(signed_rows, initial, feedback, min(score_batch, 384), amount)
