"""One-sided transport of per-concept discovery ranks under uncertain query intent."""

import torch
from torch.nn import functional as F

from .cluster_mine import cluster_assignments


def populations(photo, count=64):
    g = cluster_assignments(F.normalize(photo - photo.mean(0)), count)
    return g.unique(sorted=True, return_inverse=True)[1]


def group_mass(logits, groups, sizes):
    """Assume equal prior over groups and estimate each conditional image likelihood by its mean."""
    weights = (logits - logits.amax(-1, keepdim=True)).exp()
    mass = weights.new_zeros((*weights.shape[:-1], len(sizes))).index_add_(-1, groups, weights)
    mass /= sizes
    return mass / mass.sum(-1, keepdim=True).clamp_min(1e-30)


def intent_weights(word_logits, word_photo, visual_photo, groups, scale):
    sizes = groups.bincount()
    visual = group_mass(.35 * scale * visual_photo, groups, sizes)
    lexical = group_mass(.35 * scale * word_photo, groups, sizes)
    lexical = ((.3*word_logits).softmax(1)[:, :, None]*lexical).sum(1)
    return torch.stack((lexical, visual), 1)


def inspection_cost(length, device, powers):
    positions = torch.arange(1, length+1, device=device, dtype=torch.float64)
    cols = [positions.log() if gamma == 0 else (1-positions.pow(-gamma))/gamma for gamma in powers]
    return torch.stack(cols, 1).float()


class Coverage:
    def __init__(self, base_order, groups, posterior, cost):
        self.cost = cost
        self.groups = groups
        sizes = groups.bincount()
        self.sorted_groups = groups.sort().values
        self.base_position = groups[base_order].argsort(dim=1, stable=True)
        self.old_cost = cost[self.base_position]
        self.base_ids = base_order.gather(1, self.base_position)
        # Each group assigns equal mass to all its quantiles. Per-image weights
        # sum to that group's posterior, irrespective of population size.
        self.weights = (posterior / sizes[None, None, :])[:, :, self.sorted_groups].contiguous()
        self.positions = torch.arange(len(groups), device=groups.device).expand_as(base_order)

    def read(self, order):
        current = self.groups[order].argsort(dim=1, stable=True)
        travel = self.old_cost - self.cost[current]
        value = self.weights @ travel
        delay = self.weights @ (-travel).clamp_min(0)
        inverse = torch.empty_like(order).scatter_(1, order, self.positions)
        new_location = inverse.gather(1, self.base_ids)
        identity_travel = self.old_cost - self.cost[new_location]
        identity_delay = self.weights @ (-identity_travel).clamp_min(0)
        # Check the equality of the signed gains and the optimal coupling inequality.
        signed_gap = (value - self.weights @ identity_travel).abs().amax()
        lower_violation = (delay - identity_delay).clamp_min(0).amax()
        return value, delay, identity_delay, signed_gap, lower_violation


def utility(values, loss, penalty, prior):
    by_view = values - penalty*loss
    if prior == "both":
        return by_view.amin(-1)
    if prior == "average":
        return by_view.mean(-1)
    if prior == "lexical":
        return by_view[..., 0]
    if prior == "visual":
        return by_view[..., 1]
    raise ValueError(prior)
