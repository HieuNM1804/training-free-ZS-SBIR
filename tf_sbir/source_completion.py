"""Condition the target feature on measurements of one source image. Photo index stays fixed."""

import torch
import torch.nn.functional as F


def paired_axes(source_text, target_text, kind="paired", seed=42):
    s, t = source_text.double(), target_text.double()
    if kind == "shuffled":
        rng = torch.Generator("cpu").manual_seed(seed)
        t = t[torch.randperm(len(t), generator=rng).to(t.device)]
    midpoint, shift = (s + t) / 2, (s - t) / 2
    midpoint = midpoint - midpoint.mean(0)
    covariance = midpoint.T @ midpoint
    if kind in ("paired", "shuffled"):
        covariance -= shift.T @ shift
    elif kind != "midpoint":
        raise ValueError(kind)
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    return vectors.flip((1,)).float().contiguous(), eigenvalues.flip((0,)).cpu()


def transfer(sources, photos, target_text, scale, axes, rank, counts, arms):
    """Complete each source from a target conditional. Gallery self-exclusion is unnecessary here."""
    basis = axes[:, :rank]
    observed_photos = photos @ basis
    gt = (photos @ target_text.T) * scale
    gs = (observed_photos @ (target_text @ basis).T) * scale
    gallery_radius = (1 - observed_photos.square().sum(1)).clamp_min(0).sqrt()
    if gallery_radius.min() <= 1e-6:
        raise ValueError("Degenerate target hidden subspace")
    remaining = (gt - gs) / gallery_radius[:, None]
    gallery_words = gt.softmax(1)
    projected_search = F.normalize(observed_photos, dim=1).T.contiguous()
    output = {arm: {k: gallery_words.new_zeros(len(sources[0]), len(target_text)) for k in counts} for arm in arms}
    kmax = max(counts)
    for view in sources:
        for start in range(0, len(view), 128):
            sl = slice(start, min(len(view), start+128))
            x = F.normalize(view[sl].float(), dim=1)
            observed = x @ basis
            source_radius = (1 - observed.square().sum(1)).clamp_min(0).sqrt()
            seen_logits = (observed @ (target_text @ basis).T) * scale
            ids = (F.normalize(observed, dim=1) @ projected_search).topk(kmax).indices
            if "full_donors" in arms:
                full_ids = (x @ photos.T).topk(kmax).indices
            if {"completion", "mean_logits"} & set(arms):
                hypothetical = torch.addcmul(seen_logits[:, None], source_radius[:, None, None], remaining[ids])
            for arm in arms:
                if arm == "completion":
                    values = hypothetical.softmax(-1)
                elif arm == "mean_logits":
                    values = hypothetical
                elif arm == "matched_donors":
                    values = gallery_words[ids]
                elif arm == "full_donors":
                    values = gallery_words[full_ids]
                else:
                    raise ValueError(arm)
                total = torch.zeros_like(seen_logits)
                first = 0
                for k in sorted(counts):
                    total += values[:, first:k].sum(1)
                    estimate = total / k
                    if arm == "mean_logits":
                        estimate = estimate.softmax(1)
                    output[arm][k][sl].add_(estimate, alpha=1/len(sources))
                    first = k
    return output
