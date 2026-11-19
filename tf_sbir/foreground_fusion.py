"""Isolate the visual and lexical effects of a whole-foreground sketch view."""

from functools import partial

import torch
from torch.nn import functional as F

from .isoclip import projected_feedback
from .reranking import standardize
from .union_density import density_prefixes, density_rows
from .view_hypotheses import view_concepts


def foreground_evidence(mode, ink, photos, contours, sketch_words, photo_words, transform, gallery, words):
    ink = F.normalize(ink.float(), dim=-1)
    # Sum normalized descriptors before applying the same cross-domain feedback.
    pooled = F.normalize(mode + ink, dim=-1)
    projected, _ = projected_feedback(torch.cat((ink, pooled)), photos, contours, transform)
    ink_embedding, pooled_embedding = projected.split(len(ink))
    ink_visual = ink_embedding @ gallery.T
    ink_words = view_concepts(ink[:, None], sketch_words, photo_words).squeeze(1) @ words.T
    return ink_visual, ink_words, pooled_embedding


def joint_foreground_rank(visual, context, ink_visual, ink_words, pooled_visual, banks, graph, score_batch):
    semantic = 0.5 * (standardize(context) + standardize(ink_words))
    prior = standardize(pooled_visual) + standardize(semantic)
    streams = (pooled_visual, semantic, visual, ink_visual, context, ink_words)
    prefix = density_prefixes(pooled_visual, semantic, streams, prior, banks, graph, 1280, score_batch, 256, 16.)
    return prior, prefix, None


def foreground_fusion_settings(
    mode, ink, photos, contours, sketch_words, photo_words, transform, gallery, words,
    visual, context, mta_prior, banks, graph, score_batch,
):
    ink_visual, ink_words, pooled = foreground_evidence(
        mode, ink, photos, contours, sketch_words, photo_words, transform, gallery, words,
    )
    pooled_visual = pooled @ gallery.T

    ink_scores = standardize(ink_visual) + standardize(ink_words)
    ink_prefix = density_prefixes(ink_visual, ink_words, (ink_visual, ink_words), ink_scores,
                                  banks, graph, 1280, score_batch, 256, 16.)
    yield "foreground-ink-control", partial(density_rows, mta_prior, ink_prefix, score_batch, None)
    yield "foreground-ink-tail", partial(density_rows, ink_scores, ink_prefix, score_batch, None)

    mixed_context = 0.5 * (standardize(context) + standardize(ink_words))
    pairs = (
        ("ink-visual", ink_visual, context),
        ("ink-lexical", visual, ink_words),
        ("pooled-visual", pooled_visual, context),
        ("pooled-lexical", visual, mixed_context),
    )
    for name, view, semantic in pairs:
        prior = standardize(view) + standardize(semantic)
        prefix = density_prefixes(view, semantic, (view, semantic), prior, banks, graph, 1280, score_batch, 256, 16.)
        yield f"foreground-{name}-hard", partial(density_rows, prior, prefix, score_batch, None)
        yield f"foreground-{name}-soft", partial(density_rows, prior, prefix, score_batch, 0.8)
    prior, prefix, _ = joint_foreground_rank(visual, context, ink_visual, ink_words, pooled_visual, banks, graph, score_batch)
    yield "foreground-pooled-joint-hard", partial(density_rows, prior, prefix, score_batch, None)
    yield "foreground-pooled-joint-soft", partial(density_rows, prior, prefix, score_batch, 0.8)
