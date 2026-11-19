"""Fuse complementary full-gallery scores using query-adaptive curve weighting."""

from functools import partial

import torch
from torch.nn import functional as F

from .gallery_diffusion import diffusion_blocks, full_gallery_inputs, mutual_transition
from .reranking import standardize
from .semantic_transport import partial_similarity, rerank_transport_batch


def curve_samples(sorted_scores, fractions):
    locations = (fractions * (sorted_scores.size(1) - 1)).round().long()
    return sorted_scores[:, locations]


def curve_reliability(scores, reference, fractions, tail_start):
    """Reference subtraction and inverse area, Zheng et al. Eq. 4–7."""
    ordered = scores.sort(dim=1, descending=True).values
    curve = curve_samples(ordered, fractions)
    if reference is not None:
        distances = torch.cdist(curve[:, tail_start:], reference[:, tail_start:])
        nearest = distances.topk(10, largest=False).indices
        curve = curve - reference[nearest].mean(dim=1)
    shifted = curve - curve.amin(dim=1, keepdim=True)
    maximum = shifted.amax(dim=1, keepdim=True)
    normalized = shifted / maximum.clamp_min(1e-8)
    area = torch.trapezoid(normalized, fractions, dim=1)
    return torch.where(maximum.squeeze(1) > 1e-8, area.clamp_min(1e-4).reciprocal(), 0)


def fusion_rows(crop_scores, diffusion_scores, score_batch, weight):
    for crop, diffusion, weights in zip(
        crop_scores.split(score_batch), diffusion_scores.split(score_batch), weight.split(score_batch), strict=True,
    ):
        combined = (1 - weights) * standardize(crop) + weights * standardize(diffusion)
        yield from (0.15 * combined).tanh().cpu()


def adaptive_fusion_settings(queries, photos, contours, sketch_words, photo_words, photo_augmentations, score_batch):
    count = len(queries)
    generator = torch.Generator(device=photos.device).manual_seed(0)
    probe_ids = torch.randperm(len(photos), generator=generator, device=photos.device)[:1024]
    probes = F.normalize(contours[probe_ids].mean(dim=1), dim=-1)
    base, regional, gallery, concepts, targets = full_gallery_inputs(
        torch.cat((queries, probes)), photos, contours, sketch_words, photo_words, score_batch,
    )
    crops = torch.cat((targets, F.normalize(photo_augmentations.float(), dim=-1).to(targets.dtype)), dim=1)
    transition = mutual_transition(gallery, 160)
    diffusion_scores = torch.cat(list(diffusion_blocks(base, regional, transition, score_batch, 0.9, 320, "regional")))
    del regional
    crop_scores = torch.empty_like(base)
    scorer = partial(partial_similarity, dustbin=0.35)
    for start in range(0, len(base), 64):
        end = min(start + 64, len(base))
        crop_scores[start:end] = rerank_transport_batch(
            base[start:end], concepts[start:end], crops, 640, 1.2, scorer,
        )
    del base

    # Approximate unrelated galleries with contour probes whose RGB neighbors are removed.
    fractions = torch.cat((
        torch.linspace(0, 0.08, 128, device=photos.device),
        torch.linspace(0.08, 1, 129, device=photos.device)[1:],
    ))
    tail_start = 127
    blocked = (photos[probe_ids] @ photos.T).topk(512, dim=1).indices
    pair = (crop_scores, diffusion_scores)
    references = [
        curve_samples(scores[count:].scatter(1, blocked, -torch.inf).sort(descending=True).values[:, :-512], fractions)
        for scores in pair
    ]
    pair = (crop_scores[:count], diffusion_scores[:count])
    for weight in (0.25, 0.5, 0.75):
        weights = crop_scores.new_full((count, 1), weight)
        yield f"late-fusion-fixed-diffusion{weight:g}", partial(fusion_rows, *pair, score_batch, weights)

    for name, curves in (("no-reference", (None, None)), ("gallery-reference", references)):
        reliability = torch.stack([
            torch.cat([
                curve_reliability(block, reference, fractions, tail_start)
                for block in scores.split(score_batch)
            ])
            for scores, reference in zip(pair, curves, strict=True)
        ], dim=1)
        total = reliability.sum(dim=1, keepdim=True)
        weights = torch.where(total > 0, reliability[:, 1:2] / total.clamp_min(1e-8), 0.5)
        yield f"late-fusion-{name}", partial(fusion_rows, *pair, score_batch, weights)
