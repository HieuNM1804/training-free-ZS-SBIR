"""Rank sketches against gallery photos by distributions on shared anchors."""

from functools import partial

import torch
from torch.nn import functional as F

from .reranking import standardize
from .semantic_transport import transport_components


def log_posterior(logits, temperature):
    return (logits / temperature).log_softmax(dim=-1)


def gallery_kl_parameters(log_gallery):
    probs = log_gallery.exp()
    entropy = -(probs * log_gallery).sum(dim=1)
    return probs, entropy


def negative_gallery_kl(log_query, gallery_probs, gallery_entropy):
    """Negative D_KL(p_gallery || p_query), Eq. 7–8 of Wu et al."""
    return log_query @ gallery_probs.T + gallery_entropy


def calibrated_anchor_logits(queries, photos, contours, sketch_words, photo_words):
    sketch_logits = queries @ sketch_words.T
    photo_logits = photos @ photo_words.T
    # Per-term priors come from RGB and contour views of the same gallery.
    contour = F.normalize(contours.mean(dim=1), dim=1)
    domain_shift = contour.mean(dim=0) @ sketch_words.T - photo_logits.mean(dim=0)
    return sketch_logits, photo_logits, domain_shift


def posterior_rows(log_query, log_gallery, query, gallery, offset, score_batch, weight, symmetric=False):
    parameters = gallery_kl_parameters(log_gallery)
    for q_log, visual in zip(log_query.split(score_batch), query.split(score_batch), strict=True):
        similarities = negative_gallery_kl(q_log, *parameters)
        if symmetric:
            # Entropy of the query is constant across gallery columns.
            similarities = 0.5 * (similarities + (q_log.exp() @ log_gallery.T))
        score = standardize(similarities)
        if weight is not None:
            score = standardize(visual @ gallery.T - offset) + weight * score
        yield from (0.15 * score).tanh().cpu()


def anchor_context_settings(queries, photos, contours, sketch_words, photo_words, score_batch):
    query, gallery, offset, memory, targets = transport_components(queries, photos, contours, sketch_words, photo_words)
    active = (memory.support > 0).sum().item()
    ids = memory.support.topk(min(active, 1024)).indices
    sketch_words, photo_words = sketch_words[ids], photo_words[ids]
    sketch, camera, shift = calibrated_anchor_logits(queries, photos, contours, sketch_words, photo_words)
    calibrated = sketch - shift
    photo_smoothed = targets[:, 1] @ photo_words.T
    for name, sketch_logits, photo_logits, temperature, weight, symmetric in (
        ("raw", sketch, camera, 0.035, None, False),
        ("calibrated", calibrated, camera, 0.035, None, False),
        ("calibrated-fuse-w0.35", calibrated, camera, 0.035, 0.35, False),
        ("calibrated-fuse-w0.75", calibrated, camera, 0.035, 0.75, False),
        ("calibrated-symmetric-w0.75", calibrated, camera, 0.035, 0.75, True),
        ("smoothed-t0.025-w0.75", calibrated, photo_smoothed, 0.025, 0.75, False),
        ("smoothed-t0.05-w0.75", calibrated, photo_smoothed, 0.05, 0.75, False),
        ("smoothed-symmetric-w0.75", calibrated, photo_smoothed, 0.035, 0.75, True),
    ):
        yield f"anchor-kl-{name}", partial(
            posterior_rows, log_posterior(sketch_logits, temperature), log_posterior(photo_logits, temperature),
            query, gallery, offset, score_batch, weight, symmetric,
        )
