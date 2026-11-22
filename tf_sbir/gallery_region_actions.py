"""Value of a single unseen query region for pairwise decisions in a photo gallery."""

import torch
import torch.nn.functional as F

from .cluster_mine import cluster_assignments


def photo_channels(photos, crops, target_text, scale):
    labels = cluster_assignments(F.normalize(photos - photos.mean(0)), 64)
    _, labels = labels.unique(sorted=True, return_inverse=True)
    count = int(labels.max()) + 1
    generator = torch.Generator().manual_seed(13643)
    panels = [[], []]
    for group in range(count):
        ids = (labels == group).nonzero().flatten().cpu()
        if len(ids) < 2:
            raise ValueError("Each visual group needs two disjoint photo panels")
        ids = ids[torch.randperm(len(ids), generator=generator)]
        for half in (0, 1):
            panels[half].append(ids[half::2])
    banks = []
    for panel in panels:
        ids = torch.cat(panel)
        channel = photos.new_zeros((count, 13, len(target_text)))
        for chunk in ids.split(128):
            response = (crops[chunk].cuda().float() @ target_text.T * scale).softmax(-1)
            channel.index_add_(0, labels[chunk.to(labels.device)], response)
        channel /= labels[ids.to(labels.device)].bincount()[:, None, None]
        banks.append(channel)
    return labels, torch.stack(banks)


def utilities(prior, groups, channels):
    """All action values depend only on the global query and indexed photo data."""
    return conditional_utilities(prior, channels[0][groups], channels[1][groups])


def conditional_utilities(prior, first_channel, second_channel):
    """The two observation models may be different for each independent query."""
    probability = prior / prior.sum(-1, keepdim=True)
    first, second = (bank * probability[:, :, None, None] for bank in (first_channel, second_channel))
    i, j = torch.triu_indices(prior.size(1), prior.size(1), 1, device=prior.device)
    before = torch.minimum(probability[:, i], probability[:, j]).sum(1, keepdim=True)
    within = before.new_zeros((len(prior), first_channel.size(2)))
    across = torch.zeros_like(within)
    for left, right in zip(i.split(8), j.split(8)):
        a, b = first[:, left], first[:, right]
        c, d = second[:, left], second[:, right]
        within += .5 * (torch.minimum(a, b) + torch.minimum(c, d)).sum((1, 3))
        across += .5 * (torch.where(a >= b, d, c) + torch.where(c >= d, b, a)).sum((1, 3))
    pooled = .5 * (first + second)
    marginal = pooled.sum(1, keepdim=True)
    information = (pooled * (pooled.clamp_min(1e-30).log() - probability[:, :, None, None].clamp_min(1e-30).log()
                              - marginal.clamp_min(1e-30).log())).sum((1, 3))
    return {"heldout": before - across, "within": before - within, "information": information}


def group_prior(global_scores, assignment, temperature, rivals):
    evidence = ((global_scores - global_scores.amax(1, keepdim=True)) * temperature).exp()
    means = evidence.new_zeros((len(evidence), int(assignment.max()) + 1))
    means.index_add_(1, assignment, evidence)
    means /= assignment.bincount()
    prior, groups = means.topk(rivals, dim=1)
    return prior, groups


def within_scale_mismatch(actions):
    wrong = actions.new_tensor([3, 4, 5, 6, 7, 8, 0, 1, 2, 11, 12, 9, 10])
    return wrong[actions]


def concordance(prediction, realized):
    i, j = torch.triu_indices(prediction.size(1), prediction.size(1), 1, device=prediction.device)
    x = prediction[:, i] - prediction[:, j]
    y = realized[:, i] - realized[:, j]
    valid = (x.abs() > 1e-7) & (y.abs() > 1e-7)
    comparable = valid.sum(1)
    rate = ((x * y > 0) & valid).sum(1) / comparable.clamp_min(1)
    return rate, comparable
