"""Adapt ZLaP's inductive photo-text graph to categorical SBIR."""

from functools import partial

import torch

from .gallery_diffusion import nearest_gallery, normalized_transition, solve_diffusion
from .reranking import standardize
from .semantic_transport import concept_candidates, transport_components


def bimodal_transition(photos, anchors, photo_neighbors, text_neighbors, exponent):
    """ZLaP Eq. 10 with separate neighbor lists and powered cross-modal edges."""
    indices, similarities = nearest_gallery(photos, photo_neighbors)
    cross_indices, cross_scores = [], []
    for block in photos.split(512):
        top = (block @ anchors.T).topk(text_neighbors, dim=1)
        cross_indices.append(top.indices)
        cross_scores.append(top.values.clamp_min(0).pow(exponent))
    concept_nodes, inverse = torch.cat(cross_indices).flatten().unique(return_inverse=True)
    count = len(photos)
    image_ids = torch.arange(count, device=photos.device)
    source = torch.cat((image_ids.repeat_interleave(photo_neighbors), image_ids.repeat_interleave(text_neighbors)))
    target = torch.cat((indices.flatten(), inverse + count))
    weight = torch.cat((similarities.flatten(), torch.cat(cross_scores).flatten()))
    transition = normalized_transition(
        count + len(concept_nodes), torch.cat((source, target)), torch.cat((target, source)), weight.repeat(2),
    )
    return transition, concept_nodes


def bimodal_rows(query, gallery, offset, transition, active, concept_ids, concept_scores, batch_size, alpha, text_mass, weight):
    size = len(gallery)
    lookup = torch.full((active.max().item() + 1,), -1, device=active.device, dtype=torch.long)
    lookup[active] = torch.arange(len(active), device=active.device)
    gallery_scale = 1 + offset.abs().max()
    seed_batches = zip(query.split(batch_size), concept_ids.split(batch_size), concept_scores.split(batch_size), strict=True)
    for block, terms, logits in seed_batches:
        base = (block @ gallery.T - offset) / gallery_scale
        candidates = base.topk(256, dim=1)
        photo_seed = torch.zeros_like(base).scatter_(1, candidates.indices, (candidates.values / 0.07).softmax(dim=1))
        matched = lookup[terms.clamp_max(len(lookup) - 1)]
        valid = (terms < len(lookup)) & (matched >= 0)
        text_weights = (logits / 0.04).softmax(dim=1) * valid
        text_weights = text_weights / text_weights.sum(dim=1, keepdim=True).clamp_min(1e-10)
        words_seed = base.new_zeros((len(block), len(active))).scatter_add_(1, matched.clamp_min(0), text_weights)
        values = torch.cat((photo_seed, text_mass * words_seed), dim=1)
        walk = solve_diffusion(transition, values, alpha)[:, :size]
        walk_score = (walk.clamp_min(0) * size).log1p()
        scores = (0.15 * (standardize(base) + weight * standardize(walk_score))).tanh()
        yield from scores.cpu()


def bimodal_settings(queries, photos, contours, sketch_words, photo_words, score_batch):
    query, gallery, offset, memory, _ = transport_components(queries, photos, contours, sketch_words, photo_words)
    scores, ids = concept_candidates(queries, photos, sketch_words, photo_words, memory, 32)
    for exponent in (1, 3):
        transition, active = bimodal_transition(photos, photo_words, 64, 8, exponent)
        for alpha, text_mass, weight in ((0.8, 1, 0.5), (0.94, 1, 0.5), (0.94, 3, 0.8)):
            label = f"zlap-gallery-g{exponent}-a{alpha:g}-text{text_mass}-w{weight:g}"
            yield label, partial(
                bimodal_rows, query, gallery, offset, transition, active, ids, scores,
                min(384, score_batch), alpha, text_mass, weight,
            )
