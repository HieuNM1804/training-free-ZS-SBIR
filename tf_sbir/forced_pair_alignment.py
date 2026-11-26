"""Price a proposed correspondence by the anchor fit lost when the match is compulsory."""

import math

import numpy as np
import torch
import torch.nn.functional as F


QUADRATURE_POINTS = 32


def quadrature(spectrum, count):
    # A rational map of (0, 1) to (0, infinity), scaled in the anchor's units.
    nodes, weights = np.polynomial.legendre.leggauss(count)
    z = (nodes + 1) / 2
    scale = spectrum.double().square().mean().sqrt().item()
    nodes = scale * z / (1 - z)
    weights = scale * weights / (2 * math.pi * (1 - z) ** 2)
    return list(zip(nodes.tolist(), weights.tolist(), strict=True))


def profile_terms(x, y, spectrum, count):
    # Coordinates already use the left and right singular bases of the cross-moment.
    x2, y2 = x.square(), y.square()
    bilinear = (x * spectrum) @ y.T
    separable = bilinear.clone()
    interaction = torch.zeros_like(bilinear)
    spectral_max = spectrum.max().item()
    for t, weight in quadrature(spectrum, count):
        denom = t*t + spectrum.square()
        retained = t*t / denom
        removed = spectrum.square() / denom
        # Keeping both forms avoids cancellation in log(A) near A=1.
        ax, ay = x2 @ retained, y2 @ retained
        if t < spectral_max:
            log_ax, log_ay = ax.log(), ay.log()
        else:
            log_ax = torch.log1p(-(x2 @ removed))
            log_ay = torch.log1p(-(y2 @ removed))
        separable.add_((log_ax[:, None] + log_ay[None, :]), alpha=weight)
        cross = (x * (t * spectrum / denom)) @ y.T
        interaction.add_(torch.log1p(cross.square() / (ax[:, None] * ay[None, :])), alpha=weight)
    return bilinear, separable, separable + interaction


class AnchorGeometry:
    def __init__(self, source_text, target_text):
        a, b = source_text.double(), target_text.double()
        a, b = a - a.mean(0), b - b.mean(0)
        u, s, vh = torch.linalg.svd(a.T @ b / len(a), full_matrices=False)
        self.source = (a @ u).float().contiguous()
        self.target = (b @ vh.T).float().contiguous()
        # Units affect cost magnitudes but not rankings. Use the largest singular value as unit.
        self.spectrum = (s / s.max()).float().contiguous()
        permutation = torch.randperm(len(s), generator=torch.Generator().manual_seed(42))
        self.shuffled = self.spectrum[permutation].contiguous()

    def project(self, q, p):
        return F.normalize(q.float() @ self.source), F.normalize(p.float() @ self.target)

    def scores(self, x, y):
        bilinear, separable, profile = profile_terms(x, y, self.spectrum, QUADRATURE_POINTS)
        _, _, shuffled = profile_terms(x, y, self.shuffled, QUADRATURE_POINTS)
        return {
            "profile": profile, "fixed_map": x @ y.T, "bilinear": bilinear,
            "separable": separable, "shuffled_spectrum": shuffled,
        }

    def check(self, x, y):
        # Validate the pricing formula in the real anchor geometry. Computing this SVD
        # for all gallery pairs is precisely the expense avoided by the Jacobi identity.
        x, y = F.normalize(x.double()), F.normalize(y.double())
        s = self.spectrum.double()
        expected = torch.empty((len(x), len(y)), device=x.device, dtype=x.dtype)
        for row, xv in enumerate(x):
            left = torch.diag(s) - xv[:, None] * (xv * s)[None, :]
            for col, yv in enumerate(y):
                projected = left - (left @ yv)[:, None] * yv[None, :]
                expected[row, col] = (xv * s) @ yv + torch.linalg.svdvals(projected).sum() - s.sum()
        tested = profile_terms(x.float(), y.float(), self.spectrum, QUADRATURE_POINTS)[2].double()
        refined = profile_terms(x, y, s, 2 * QUADRATURE_POINTS)[2]
        error = float((tested - expected).abs().max())
        refined_error = float((refined - expected).abs().max())
        if error > 2e-4 or refined_error > 2e-5 or torch.any(expected > 2e-8):
            raise ArithmeticError(f"Constrained Procrustes verification failed {error=} {refined_error=}")
        toy_s = s.new_tensor([1, 2, 3, 4]) / 4
        toy_x = F.normalize(s.new_tensor([[1, math.sqrt(2), 1, 1]]))
        toy_y = F.normalize(s.new_tensor([
            [1, -math.sqrt(2), 1, 1], [-1, math.sqrt(2), -1, 1],
        ]))
        toy_bilinear, toy_separable, toy_profile = profile_terms(toy_x, toy_y, toy_s, 64)
        for control in (toy_x @ toy_y.T, toy_bilinear, toy_separable):
            torch.testing.assert_close(control[0, 0], control[0, 1])
        if abs(float(toy_profile[0, 0] - toy_profile[0, 1])) <= 1e-6:
            raise ArithmeticError("Two distinct conditional constraints collapsed onto the surrogate")
        return {
            "pairs": expected.numel(), "max_absolute_error_vs_exact_svd": error,
            "double_order_error_vs_exact_svd": refined_error, "quadrature_nodes": QUADRATURE_POINTS,
            "spectrum_units": "largest singular value = 1", "anchor_dimension": len(s),
            "surrogate_tie_example": {
                "fixed_map": (toy_x @ toy_y.T).tolist()[0], "bilinear": toy_bilinear.tolist()[0],
                "separable": toy_separable.tolist()[0], "profile": toy_profile.tolist()[0],
            },
        }
