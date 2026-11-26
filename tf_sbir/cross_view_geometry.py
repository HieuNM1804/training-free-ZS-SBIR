"""Closed-form similarity verification on two observations of patch matches."""

from dataclasses import dataclass

import torch

from .depiction_pivot_geometry import distance_agreement


@dataclass(frozen=True)
class Moments:
    source: torch.Tensor
    target: torch.Tensor
    source_sq: torch.Tensor
    target_sq: torch.Tensor
    cross: torch.Tensor


def moments(coupling, source, target):
    weighted_target = coupling @ target
    source_mass = coupling.sum(-1)
    target_mass = coupling.sum(-2)
    return Moments(
        source_mass @ source,
        target_mass @ target,
        source_mass @ source.square().sum(-1),
        target_mass @ target.square().sum(-1),
        source.T @ weighted_target,
    )


def outer(left, right):
    return left.unsqueeze(-1) * right.unsqueeze(-2)


def fit_similarity(train):
    """Fit one 2-D similarity with either orientation, and a free translation."""
    x, y = train.source, train.target
    h = train.cross - outer(x, y)
    energy = (train.source_sq - x.square().sum(-1)).clamp_min(1e-6)
    preserve_a = h[..., 0, 0] + h[..., 1, 1]
    preserve_b = h[..., 0, 1] - h[..., 1, 0]
    reverse_a = h[..., 0, 0] - h[..., 1, 1]
    reverse_b = h[..., 0, 1] + h[..., 1, 0]
    reverse = reverse_a.square() + reverse_b.square() > preserve_a.square() + preserve_b.square()
    a = torch.where(reverse, reverse_a, preserve_a) / energy
    b = torch.where(reverse, reverse_b, preserve_b) / energy
    sign = torch.where(reverse, -1., 1.)
    matrix = torch.stack((a, b, -sign * b, sign * a), dim=-1).unflatten(-1, (2, 2))
    return matrix, a.square() + b.square()


def explained_error(train, test):
    """Predict coordinates in the second coupling, relative to a constant prediction."""
    matrix, scale_sq = fit_similarity(train)
    x, y = train.source, train.target
    x_energy = test.source_sq - 2 * (test.source * x).sum(-1) + x.square().sum(-1)
    y_energy = test.target_sq - 2 * (test.target * y).sum(-1) + y.square().sum(-1)
    cross = test.cross - outer(x, test.target) - outer(test.source, y) + outer(x, y)
    return (2 * (cross * matrix).sum((-1, -2)) - scale_sq * x_energy) / y_energy.clamp_min(1e-6)


def cross_prediction(first, second):
    return .5 * (explained_error(first, second) + explained_error(second, first))


def grid_controls(device, seed):
    grid = torch.cartesian_prod(torch.arange(-3, 4), torch.arange(-3, 4))
    radius = grid.square().sum(1)
    permutation = torch.arange(len(grid))
    generator = torch.Generator().manual_seed(seed)
    for level in radius.unique():
        idx = (radius == level).nonzero().flatten()
        permutation[idx] = idx[torch.randperm(len(idx), generator=generator)]
    if not torch.equal(radius, radius[permutation]):
        raise ValueError("Coordinate shuffle changed distances to the image center")
    coords = grid.to(device=device, dtype=torch.float32)
    flip = torch.arange(len(grid), device=device).reshape(7, 7).flip(1).flatten()
    return coords, flip, coords[permutation.to(device)]


def coupling(cosines, scale):
    logits = scale * cosines
    return (logits.softmax(-1) + logits.softmax(-2)) / (2 * logits.size(-1))


def evidence(sketch, reflected, gallery, scale, coords, flip, shuffled, distance):
    gallery_t = gallery.transpose(-1, -2)
    original_cos = (sketch[:, None] @ gallery_t).float()
    reflected_cos = (reflected[:, None] @ gallery_t).float()
    first = coupling(original_cos, scale)
    flipped = coupling(reflected_cos, scale)
    canonical = flipped.index_select(-2, flip)
    a, b = moments(first, coords, coords), moments(canonical, coords, coords)
    wrong = moments(flipped, coords, coords)
    real = cross_prediction(a, b)
    wrong_frame = cross_prediction(a, wrong)
    radial = cross_prediction(moments(first, coords, shuffled), moments(canonical, coords, shuffled))
    return {
        "appearance": .5 * ((first * original_cos).sum((-1, -2)) + (flipped * reflected_cos).sum((-1, -2))),
        "distance": .5 * (distance_agreement(first, distance, distance)
                         + distance_agreement(canonical, distance, distance)),
        "within_view": .5 * (explained_error(a, a) + explained_error(b, b)),
        "cross_view": real,
        "coordinate_contrast": real - wrong_frame,
        "wrong_frame": wrong_frame,
        "radial_shuffle": radial,
    }


def audit_geometry(coords, flip):
    """Check true transfers, coordinate-only matching, and shuffled correspondences."""
    true = torch.eye(len(coords), device=coords.device) / len(coords)
    reference = moments(true, coords, coords)
    photo_shift_scale = coords * 1.7 + coords.new_tensor([2., -1.])
    moved = moments(true, coords, photo_shift_scale)
    position_only = moments(true[flip], coords, coords)
    independent = moments(torch.full_like(true, 1 / len(coords)**2), coords, coords)
    measured = torch.stack((cross_prediction(reference, reference), cross_prediction(moved, moved),
                            cross_prediction(reference, position_only), cross_prediction(independent, independent)))
    expected = coords.new_tensor([1., 1., -1., 0.])
    if not torch.allclose(measured, expected, atol=2e-6, rtol=0):
        raise ValueError(f"Unexpected geometry sanity scores: {measured.tolist()}")
    return measured.tolist()
