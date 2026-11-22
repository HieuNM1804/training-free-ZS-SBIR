"""Test a common latent visual deviation against two independent word-matched photographs."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .cluster_mine import cluster_assignments


def lexical_groups(weights, count=64):
    probability = weights.float().clamp_min(0)
    probability /= probability.sum(1, keepdim=True).clamp_min(1e-30)
    points = probability.sqrt()
    labels = cluster_assignments(points, count)
    sizes = labels.bincount(minlength=count)
    eligible = (sizes >= 8).nonzero().flatten()
    if not len(eligible):
        raise ValueError("Insufficient gallery support for cross-fitting")
    if (sizes[labels] < 8).any():
        centroid = points.new_zeros((count, points.size(1))).index_add_(0, labels, points)
        bad = sizes[labels] < 8
        labels[bad] = eligible[(points[bad] @ F.normalize(centroid[eligible]).T).argmax(1)]
    return labels.unique(sorted=True, return_inverse=True)[1]


def partition(groups, seed=71413):
    rng = torch.Generator(device=groups.device).manual_seed(seed)
    folds = torch.empty_like(groups)
    for g in groups.unique():
        ids = (groups == g).nonzero().flatten()
        ids = ids[torch.randperm(len(ids), device=groups.device, generator=rng)]
        folds[ids] = torch.arange(len(ids), device=groups.device) % 4
    return folds


def within_permutation(labels, seed):
    rng = torch.Generator(device=labels.device).manual_seed(seed)
    permutation = torch.arange(len(labels), device=labels.device)
    for g in labels.unique():
        ix = (labels == g).nonzero().flatten()
        ordered = ix[torch.randperm(len(ix), generator=rng, device=ix.device)]
        permutation[ordered] = ordered.roll(1)
    return permutation


def group_means(x, g, count):
    size = g.bincount(minlength=count)
    if int(size.min()) == 0:
        raise ValueError("Missing group during independent gallery calibration")
    return x.new_zeros((count, x.size(1))).index_add_(0, g, x) / size[:, None]


def whitener(data, ridge):
    cov = (data.T @ data) / len(data)
    values, basis = torch.linalg.eigh(cov.double())
    regularizer = ridge * cov.diagonal().mean()
    inv = (values.clamp_min(0) + regularizer).rsqrt()
    return (basis * inv @ basis.T).float()


@dataclass
class CanonicalView:
    transform: torch.Tensor
    mean: torch.Tensor
    overall: torch.Tensor
    candidate: torch.Tensor


def make_view(train, heldout, train_labels, eval_view, calibration_labels, transform, groups):
    means = group_means(train, train_labels, groups)
    calibrated = eval_view @ transform
    calibrated -= group_means(calibrated, calibration_labels, groups)[calibration_labels]
    sd = calibrated.square().mean(0).clamp_min(1e-12).sqrt()
    transform = transform / sd
    return CanonicalView(transform, means @ transform, train.mean(0) @ transform,
                         heldout @ transform), calibrated / sd


def oriented_score(query, observed, candidate, group, rho, k, mode):
    r = rho[:k]
    x = query @ observed.transform[:, :k]
    y = candidate.candidate[:, :k]
    if mode == "pooled_center":
        sa, sb = observed.overall[None, :k], candidate.overall[None, :k]
        g = torch.zeros_like(group)
    else:
        sa, sb = observed.mean[:, :k], candidate.mean[:, :k]
        g = group
    y = y - sb[g]
    product = r / (1 - r.square())
    cross = x @ (y * product).T - (sa[g] * y * product).sum(1)
    if mode == "cross_only":
        return cross / k
    penalty = .5 * r * product
    target_penalty = (y.square() * penalty).sum(1)
    query_penalty = (x.square() * penalty).sum(1, keepdim=True) + (sa.square() * penalty).sum(1) - 2 * (x * penalty) @ sa.T
    constant = -.5 * torch.log1p(-r.square()).sum()
    return (cross - target_penalty - query_penalty[:, g] + constant) / k


@dataclass
class Heldout:
    ids: torch.Tensor
    group: torch.Tensor
    a: CanonicalView
    b: CanonicalView
    rho: torch.Tensor
    diagnostics: dict

    def score(self, a_query, b_query, k, mode):
        ab = oriented_score(a_query, self.a, self.b, self.group, self.rho, k, mode)
        ba = oriented_score(b_query, self.b, self.a, self.group, self.rho, k, mode)
        return .5 * (ab + ba)


def fit_gallery(a, b, lex, fold, setting, ridge):
    if setting == "global":
        groups = torch.zeros_like(lex)
    elif setting == "permuted":
        groups = lex[within_permutation(fold, 67481)]
    else:
        groups = lex
    count = int(groups.max()) + 1
    models = []
    for i in range(4):
        test = (fold == i).nonzero().flatten()
        calibration = (fold == (i + 1) % 4).nonzero().flatten()
        train = ((fold != i) & (fold != (i + 1) % 4)).nonzero().flatten()
        gt, gc = groups[train], groups[calibration]
        xa, xb = a[train], b[train]
        da = xa - group_means(xa, gt, count)[gt]
        db = xb - group_means(xb, gt, count)[gt]
        if setting == "sham":
            db = db[within_permutation(gt, 19145 + i)]
        wa, wb = whitener(da, ridge), whitener(db, ridge)
        u, train_rho, vh = torch.linalg.svd((wa.T @ ((da.T @ db) / len(da)) @ wb).double(), full_matrices=False)
        pa, pb = wa @ u.float(), wb @ vh.T.float()
        view_a, z_a = make_view(xa, a[test], gt, a[calibration], gc, pa, count)
        view_b, z_b = make_view(xb, b[test], gt, b[calibration], gc, pb, count)
        z_other = z_b[within_permutation(gc, 44711 + i)]
        observed_corr = (z_a * z_b).mean(0)
        null_corr = (z_a * z_other).mean(0)
        calibration_corr = null_corr if setting == "sham" else observed_corr
        rho = calibration_corr.clamp(min=0., max=1. - 1e-5)
        # A third split measures whether the estimated evidence prefers the paired detail.
        ga = groups[test]
        ax = view_a.candidate - view_a.mean[ga]
        by = view_b.candidate - view_b.mean[ga]
        impostor = within_permutation(lex[test], 93532 + i)
        shuffled = by[impostor]
        impostor_x = view_a.candidate - view_a.mean[ga[impostor]]
        diagnostics = {}
        for k in (16, 64, 192):
            r = rho[:k]
            weights = r / (1 - r.square())
            paired_advantage = (weights * (ax[:, :k] * by[:, :k] - impostor_x[:, :k] * shuffled[:, :k])
                                - .5 * r * weights * (ax[:, :k].square() + by[:, :k].square()
                                                      - impostor_x[:, :k].square() - shuffled[:, :k].square())).sum(1)
            diagnostics[str(k)] = {"train_corr": float(train_rho[:k].mean()),
                                   "calib_pair_corr": float(observed_corr[:k].mean()),
                                   "calib_sham_corr": float(null_corr[:k].mean()),
                                   "paired_over_same_group_impostor": float((paired_advantage > 0).float().mean())}
        models.append(Heldout(test, ga, view_a, view_b, rho, diagnostics))
    return models


def visual_score(models, a, b, size, k, mode):
    result = a.new_empty((len(a), size))
    for model in models:
        result[:, model.ids] = model.score(a, b, k, mode)
    return result


def zscore(score):
    centered = score - score.mean(1, keepdim=True)
    return centered / centered.square().mean(1, keepdim=True).sqrt().clamp_min(1e-8)


def within_group_update(base, aux, group, weight):
    n = int(group.max()) + 1
    def partition(scores):
        top = scores.amax(1, keepdim=True)
        z = scores.new_zeros((len(scores), n)).index_add_(1, group, (scores - top).exp())
        return z.clamp_min(1e-30).log() + top
    updated = base + weight * aux
    adjustment = partition(updated) - partition(base)
    return updated - adjustment[:, group]
