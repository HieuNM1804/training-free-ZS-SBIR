"""Estimate which parts of each candidate appearance can be predicted from its paired source domain."""

import torch
from torch.nn import functional as F

from .contour_graph_index import gallery_walk
from .transfer_reconstruction import transfer_coordinates


def disjoint_anchor_support(source, count, seed):
    """Two disjoint reference banks; a candidate's own id is excluded from both predictions."""
    generator = torch.Generator(device=source.device).manual_seed(seed)
    order = torch.randperm(len(source), device=source.device, generator=generator)
    if len(source) // 2 <= count:
        raise ValueError("Each half of the reference bank must have more entries than the support")
    supports = []
    for half in order.tensor_split(2):
        bank = source[half]
        position = half.new_full((len(source),), -1)
        position[half] = torch.arange(len(half), device=half.device)
        nearest = []
        for start in range(0, len(source), 256):
            q = source[start:start + 256]
            scores = q @ bank.T
            own = position[start:start + len(q)]
            valid = own.ge(0)
            rows = torch.arange(len(q), device=q.device)
            scores[rows[valid], own[valid]] = -torch.inf
            nearest.append(half[scores.topk(count, dim=1).indices])
        supports.append(torch.cat(nearest))
    return supports


def predictable_residuals(target, first, second):
    """Separate a repeatable prediction discrepancy from instability of the reference support."""
    x, y = target - first, target - second
    residual = (x + y) * .5
    repeatable = (x * y).sum(dim=-1, keepdim=True)
    fraction = (repeatable / residual.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(0., 1.)
    return residual, fraction


def gallery_variants(source, target, smoothing, metadata):
    supports = disjoint_anchor_support(source, 64, seed=881)
    for count in (16, 64):
        predictions = [transfer_coordinates(source, source, target, ids[:, :count], .1) for ids in supports]
        residual, fraction = predictable_residuals(target, *predictions)
        metadata[f"k{count}"] = {"mean_reliability": fraction.mean().item(),
                                 "mean_residual_length": residual.norm(dim=-1).mean().item()}
        generator = torch.Generator(device=source.device).manual_seed(887)
        permutation = torch.randperm(len(source), device=source.device, generator=generator)
        strengths = (.25, .5, .75, 1.) if count == 16 else (.5, 1.)
        controls = [("adaptive", strength, fraction, residual) for strength in strengths]
        if count == 16:
            controls += [("uniform", .75, torch.ones_like(fraction), residual),
                         ("mean", .75, fraction.mean(), residual),
                         ("shuffled-reliability", .75, fraction[permutation], residual),
                         ("shuffled-residual", .75, fraction, residual[permutation]),
                         ("sharpen", -.5, fraction, residual)]
        for mode, strength, weight, error in controls:
            corrected = F.normalize(target - strength * weight * error, dim=-1)
            smooth = gallery_walk(*smoothing, corrected, .8, 4)
            yield f"predictable-{mode}-k{count}-s{strength:g}", smooth, {"anchor_support": count, "strength": strength, "mode": mode}
