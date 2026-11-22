"""Gallery observation channels conditioned on a native-resolution CLIP preview."""

import torch
import torch.nn.functional as F

from .cluster_mine import cluster_assignments


def signature(regional, full, terms):
    x = (regional.float() - full.float()[:, None]) @ terms.T
    return F.normalize(x - x.mean(-1, keepdim=True), dim=-1)


def build_bank(photos, preview, zoom, target, scale, maximum=96):
    labels = cluster_assignments(F.normalize(photos - photos.mean(0)), 64)
    _, labels = labels.unique(sorted=True, return_inverse=True)
    groups = int(labels.max()) + 1
    ids = torch.zeros((2, groups, maximum), dtype=torch.long)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    shuffle = torch.zeros_like(ids)
    rng = torch.Generator().manual_seed(13643)
    for group in range(groups):
        ix = (labels == group).nonzero().flatten().cpu()
        ix = ix[torch.randperm(len(ix), generator=rng)]
        for fold in (0, 1):
            sample = ix[fold::2][:maximum]
            n = len(sample)
            if n < 2:
                raise ValueError("A within-panel pairing control needs at least two gallery photos")
            ids[fold, group, :n] = sample
            valid[fold, group, :n] = True
            shift = int(torch.randint(1, n, (), generator=rng))
            shuffle[fold, group, :n] = torch.arange(n).roll(shift)
    shape = (2, groups, 13, maximum, len(target))
    stored_preview = photos.new_empty(shape, dtype=torch.float16)
    stored_zoom = torch.empty_like(stored_preview)
    shuffled_zoom = torch.empty_like(stored_preview)
    for fold in range(2):
        for group in range(groups):
            rows = ids[fold, group]
            p = signature(preview[rows].cuda(), photos[rows.cuda()], target)
            z = (zoom[rows].cuda().float() @ target.T * scale).softmax(-1)
            stored_preview[fold, group] = p.transpose(0, 1).half()
            stored_zoom[fold, group] = z.transpose(0, 1).half()
            shuffled_zoom[fold, group] = z[shuffle[fold, group].cuda()].transpose(0, 1).half()
    return labels, stored_preview, stored_zoom, shuffled_zoom, valid.cuda()


def predict_observation(q, groups, bank, temperatures):
    """Estimate all possible zoom word distributions without reading any actual sketch zoom."""
    fine, normal, shuffled, valid = bank
    valid = valid[:, groups]
    counts = valid.sum(-1)
    output = {}
    swapped = (3, 4, 5, 6, 7, 8, 0, 1, 2, 11, 12, 9, 10)
    actions = range(q.size(1))
    for r in actions:
        coarse = fine[:, :, r][:, groups].float()
        zoom = normal[:, :, r][:, groups].float()
        placebo = shuffled[:, :, r][:, groups].float()
        aligned = (coarse * q[None, :, None, r:r + 1]).sum(-1)
        miscued = (coarse * q[None, :, None, swapped[r]:swapped[r] + 1]).sum(-1)
        base = (zoom * valid[..., None]).sum(-2) / counts[..., None]
        output.setdefault(("unconditional", 0.), []).append(base)
        for temperature in temperatures:
            for kind, similarity in (("aligned", aligned), ("wrong_preview", miscued)):
                logit = (temperature * similarity).masked_fill(~valid, -torch.inf)
                weights = logit.softmax(-1)
                observed = (weights[..., None] * zoom).sum(-2)
                output.setdefault((kind, temperature), []).append(observed)
                if kind == "aligned":
                    output.setdefault(("broken_pair", temperature), []).append((weights[..., None] * placebo).sum(-2))
                    log_response = logit.logsumexp(-1) - counts.log()
                    output.setdefault(("preview_response", temperature), []).append(log_response)
                    output.setdefault(("ess", temperature), []).append(weights.square().sum(-1).reciprocal() / counts)
    # The channels have shape fold, query, rival, action, word.
    return {key: torch.stack(values, dim=3) for key, values in output.items()}
