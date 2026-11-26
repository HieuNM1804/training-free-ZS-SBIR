"""Conditional resampling on a sphere using only a visible subspace to select donors."""

import torch
import torch.nn.functional as F

from .gallery_diffusion import nearest_gallery


def random_frames(dimension, count, seed, device):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    frames = []
    for _ in range(count):
        gaussian = torch.randn(dimension, dimension, generator=rng, dtype=torch.float64)
        frame, triangular = torch.linalg.qr(gaussian)
        frame = frame * triangular.diag().sign()
        frames.append(frame.to(device=device, dtype=torch.float32))
    return frames


def counterexample():
    # Both endpoints are neutral about the word; the visibly identical part of the
    # candidate is kept exactly while the missing direction is supplied by the donor.
    b = .25
    a = (1 - 2 * b**2) ** .5
    x = torch.tensor([a, b, -b], dtype=torch.float64)
    y = torch.tensor([a, -b, b], dtype=torch.float64)
    word = F.normalize(torch.tensor([0., 1., 1.], dtype=torch.float64), dim=0)
    z = torch.cat((x[:2], x[2:].norm() * F.normalize(y[2:], dim=0)))
    odds = lambda v: torch.sigmoid(2 * v.dot(word)).item()
    result = dict(candidate=odds(x), donor=odds(y), completion=odds(z),
                  visible_cosine=F.cosine_similarity(x[:2], y[:2], dim=0).item())
    assert result["candidate"] == result["donor"] == .5 and result["completion"] > .65
    assert torch.equal(z[:2], x[:2]) and abs(z.norm().item() - 1.) < 1e-12
    return result


def unconditioned_indices(n, k, seed, device):
    rng = torch.Generator("cpu").manual_seed(seed)
    order = torch.randperm(n, generator=rng, device="cpu")
    offsets = 1 + torch.randperm(n - 1, generator=rng)[:k]
    shuffled = order[(torch.arange(n)[:, None] + offsets) % n]
    return shuffled[order.argsort()].to(device)


def complete(photos, text, scale, frames, rank, counts, *, full_indices=None,
             readout="posterior", selection="visible", seed=42, block=96):
    """Return one normalized index per k. Scores or labels of queries are never accessed."""
    count = max(counts)
    full_logits = (photos @ text.T) * scale
    native = full_logits.softmax(1)
    accum = {k: torch.zeros_like(full_logits) for k in counts}
    agreement = []
    check_error = 0.
    for trial, frame in enumerate(frames):
        basis = frame[:, :rank]
        visible = photos @ basis
        radius = (1 - visible.square().sum(1)).clamp_min(0).sqrt()
        if radius.min().item() <= 1e-6:
            raise ValueError("This gallery has a degenerate hidden direction")
        if selection == "visible":
            indices, _ = nearest_gallery(F.normalize(visible, dim=1), count)
        elif selection == "full":
            indices = full_indices[:, :count]
        elif selection == "random":
            indices = unconditioned_indices(len(photos), count, seed + trial, photos.device)
        else:
            raise ValueError(selection)
        seen_logits = (visible @ (text @ basis).T) * scale
        hidden_unit_logits = (full_logits - seen_logits) / radius[:, None]
        for start in range(0, len(photos), block):
            sl = slice(start, min(len(photos), start + block))
            neighbors = indices[sl]
            if readout == "donor":
                values = native[neighbors]
            else:
                logits = torch.addcmul(seen_logits[sl, None], radius[sl, None, None],
                                       hidden_unit_logits[neighbors])
                if trial == 0 and start == 0:
                    target = neighbors[:min(16, len(neighbors)), 0]
                    donor_hidden = photos[target] - visible[target] @ basis.T
                    owner = slice(0, len(target))
                    z = visible[owner] @ basis.T + radius[owner, None] * F.normalize(donor_hidden, dim=1)
                    check_error = max((z.norm(dim=1) - 1).abs().max().item(),
                                      ((z @ basis) - visible[owner]).abs().max().item(),
                                      (scale * (z @ text.T) - logits[owner, 0]).abs().max().item())
                    if check_error > 1.5e-4:
                        raise ValueError(f"Closed form failed direct completion: {check_error}")
                values = logits.softmax(-1) if readout == "posterior" else logits
                if readout not in ("posterior", "mean_logits"):
                    raise ValueError(readout)
            first = 0
            running = torch.zeros_like(accum[count][sl])
            for k in sorted(counts):
                running += values[:, first:k].sum(1)
                accum[k][sl].add_(running, alpha=1 / (k * len(frames)))
                first = k
        # Independent descriptive check of whether visible-conditioned donors predict
        # the hidden direction. It does not enter any ranking or hyperparameter selection.
        ids = torch.arange(0, len(photos), 8, device=photos.device)
        owner_hidden = photos[ids] - visible[ids] @ basis.T
        donors = indices[ids, :min(16, count)]
        other_hidden = photos[donors] - visible[donors] @ basis.T
        agreement.append(F.cosine_similarity(owner_hidden[:, None], other_hidden, dim=-1).mean().item())
    if readout == "mean_logits":
        accum = {k: matrix.softmax(1) for k, matrix in accum.items()}
    for matrix in accum.values():
        if not torch.isfinite(matrix).all() or (matrix.sum(1) - 1).abs().amax() > 2e-5:
            raise ValueError("Invalid concept distribution")
    return accum, {"average_hidden_cosine_top16": sum(agreement) / len(agreement),
                   "direct_completion_max_error": check_error}


def full_donor_controls(photos, text, scale, indices, counts, *, centroid=False):
    native = (scale * (photos @ text.T)).softmax(1)
    result = {k: torch.empty_like(native) for k in counts}
    for start in range(0, len(photos), 96):
        sl = slice(start, min(start + 96, len(photos)))
        bank = photos if centroid else native
        values = bank[indices[sl, :max(counts)]]
        for k in counts:
            mean = values[:, :k].mean(1)
            if centroid:
                mean = (scale * (F.normalize(mean, dim=1) @ text.T)).softmax(1)
            result[k][sl] = mean
    return result
