"""Use per-sketch neighborhood agreement to fuse early and late query edges."""

from functools import partial

import torch

from .contextual_rank import column_stochastic, jaccard_seed
from .contour_query_expansion import pagerank_tail
from .reranking import standardize
from .sketch_view_graph_fusion import separate_view_prefixes, sketch_view_graph_inputs, source_votes
from .union_density import candidate_union, density_prefixes, density_rows, grow_density, pack_affinity


def consistency_fraction(early, late, power=1, prior=0.5):
    """Convert total edge strength into a query-specific weight for the late graph."""
    total = early + late
    early_strength = (1 - prior) * (early / total.clamp_min(1e-8)).pow(power)
    late_strength = prior * (late / total.clamp_min(1e-8)).pow(power)
    weight = late_strength / (early_strength + late_strength).clamp_min(1e-24)
    return torch.where(total > 0, weight, prior)


def shared_view_votes(crop, full, bank, full_weight):
    """Evaluate both view neighborhoods at every node retrieved by either view."""
    depth = bank.size(1)
    crop_ids = crop.topk(depth, dim=1).indices
    full_ids = full.topk(depth, dim=1).indices
    candidates = torch.cat((crop_ids, full_ids), dim=1)
    neighbors = bank[candidates].flatten(1)

    def overlaps(ids):
        membership = torch.zeros_like(crop, dtype=torch.bool).scatter_(1, ids, True)
        shared = membership.gather(1, neighbors).view(len(crop), 2 * depth, depth).sum(dim=2).float()
        return shared / (2 * depth - shared)

    crop_votes = overlaps(crop_ids)
    full_votes = overlaps(full_ids)
    weight = consistency_fraction(crop_votes[:, :depth].sum(dim=1, keepdim=True),
                                  full_votes[:, depth:].sum(dim=1, keepdim=True), prior=full_weight)
    # Common nodes carry exactly the same value in both copies of the candidate list.
    return torch.zeros_like(crop).scatter_(1, candidates, crop_votes.lerp(full_votes, weight))


def conditional_prefixes(visual, lexical, combined, ties, banks, graph, score_batch):
    policy_parameters = ((1, 0.5), (4, 0.5), (16, 0.5), (4, 0.75))
    policy_names = tuple(f"mass-p{power}-prior{prior:g}" for power, prior in policy_parameters)
    variants = policy_names + ("per-modality", "appearance-gate", "shared-visual", "shared-joint", "shared-half")
    prefixes = {name: [] for name in variants}
    tensors = (*visual, *lexical, *combined, ties)
    batches = (scores.split(min(512, score_batch)) for scores in tensors)
    for crop, full, crop_words, full_words, mixed_visual, mixed_words, prior_scores in zip(*batches, strict=True):
        candidates, valid = candidate_union((crop, full, crop_words, full_words), 1280)
        early_visual = jaccard_seed(mixed_visual, banks[0])
        early_words = jaccard_seed(mixed_words, banks[1])
        late_visual = source_votes(crop, full, banks[0], 0.35, True)
        late_words = source_votes(crop_words, full_words, banks[1], 0.65, True)
        early = early_visual + early_words
        late = late_visual + late_words
        strengths = (early.sum(dim=1, keepdim=True), late.sum(dim=1, keepdim=True))
        visual_strength = (early_visual.sum(dim=1, keepdim=True), late_visual.sum(dim=1, keepdim=True))
        word_strength = (early_words.sum(dim=1, keepdim=True), late_words.sum(dim=1, keepdim=True))

        def append(name, votes):
            prefixes[name].append(grow_density(votes * 32., prior_scores, candidates, valid, graph, 256))

        for name, (power, prior) in zip(policy_names, policy_parameters, strict=True):
            append(name, early.lerp(late, consistency_fraction(*strengths, power=power, prior=prior)))
        visual_gate = consistency_fraction(*visual_strength, power=4)
        word_gate = consistency_fraction(*word_strength, power=4)
        append("per-modality", early_visual.lerp(late_visual, visual_gate) + early_words.lerp(late_words, word_gate))
        append("appearance-gate", early.lerp(late, visual_gate))

        shared_visual = shared_view_votes(crop, full, banks[0], 0.35)
        shared_words = shared_view_votes(crop_words, full_words, banks[1], 0.65)
        shared = shared_visual + shared_words
        append("shared-visual", shared_visual + late_words)
        append("shared-joint", shared)
        append("shared-half", late.lerp(shared, 0.5))
    return {name: torch.cat(result) for name, result in prefixes.items()}


def conditional_view_affinity_settings(queries, photos, contours, sketch_words, photo_words, maps, ink, score_batch):
    visual, lexical, banks, edges = sketch_view_graph_inputs(queries, photos, contours, sketch_words, photo_words, maps, ink)
    graph = pack_affinity(len(photos), edges)
    transition, dangling = column_stochastic(len(photos), edges)
    visual_standard = tuple(standardize(scores) for scores in visual)
    lexical_standard = tuple(standardize(scores) for scores in lexical)
    early = (visual_standard[0].lerp(visual_standard[1], 0.5), lexical_standard[0].lerp(lexical_standard[1], 0.5))
    late = (visual_standard[0].lerp(visual_standard[1], 0.35), lexical_standard[0].lerp(lexical_standard[1], 0.65))
    del visual_standard, lexical_standard

    early_prior = standardize(early[0]) + standardize(early[1])
    prefix = density_prefixes(*early, early, early_prior, banks, graph, 1280, score_batch, 256, 64.)
    tail = pagerank_tail(*early, early_prior, banks, transition, dangling, score_batch)
    yield "conditional-affinity-early-reference", partial(density_rows, tail, prefix, score_batch, None)
    del early_prior, tail, prefix

    late_prior = standardize(late[0]) + standardize(late[1])
    prefix = separate_view_prefixes(visual, lexical, late_prior, banks, graph, 0.35, 0.65, True, score_batch)
    tail = pagerank_tail(*late, late_prior, banks, transition, dangling, score_batch)
    yield "conditional-affinity-late-reference", partial(density_rows, tail, prefix, score_batch, None)
    for name, prefix in conditional_prefixes(visual, lexical, early, late_prior, banks, graph, score_batch).items():
        yield f"conditional-affinity-{name}", partial(density_rows, tail, prefix, score_batch, None)
