"""Compare a proposed gallery correspondence with the candidate's observed correspondence."""

import torch
import torch.nn.functional as F

from .forced_pair_alignment import QUADRATURE_POINTS, profile_terms, quadrature
from .forced_pair_feasible_audit import feasible_costs


def sphere(probabilities):
    return F.normalize(probabilities.float().clamp_min(0).sqrt())


def paired_profiles(x, y, spectrum):
    """Evaluate just the corresponding pairs of rows, with no N by N matrix."""
    score = (x * y) @ spectrum
    for t, weight in quadrature(spectrum, QUADRATURE_POINTS):
        denom = t*t + spectrum.square()
        retained = t*t / denom
        ax, ay = x.square() @ retained, y.square() @ retained
        if t >= 1:
            removed = spectrum.square() / denom
            ax_log = torch.log1p(-(x.square() @ removed))
            ay_log = torch.log1p(-(y.square() @ removed))
        else:
            ax_log, ay_log = ax.log(), ay.log()
        cross = (x * y) @ (t * spectrum / denom)
        score.add_(ax_log + ay_log + torch.log1p(cross.square() / (ax*ay)), alpha=weight)
    return score


class GallerySubstitution:
    def __init__(self, source_posterior_rows, target_posterior_rows, seed=42):
        self.source = sphere(source_posterior_rows)
        self.target = sphere(target_posterior_rows)
        self.n = len(self.source)
        matrix = self.source.double().T @ self.target.double() / self.n
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        self.unit = s[0].item()
        self.s = (s / self.unit).float()
        self.u = u.float().contiguous()
        self.a = (self.source @ self.u).contiguous()
        self.b = (self.target @ vh.float().T).contiguous()
        self.observed = paired_profiles(self.a, self.b, self.s)
        feasible = []
        for a, b in zip(self.a.split(128), self.b.split(128), strict=True):
            pair_costs, _ = feasible_costs(a, b, self.s)
            feasible.append(-pair_costs.diagonal().clone())
        self.observed_feasible = torch.cat(feasible)
        # Controls keep the distribution of reference costs, with local permutations
        # among candidates of similar fixed-map agreement and posterior concentration.
        cosine = (self.a * self.b).sum(1)
        concentration = target_posterior_rows.square().sum(1)
        gen = torch.Generator().manual_seed(seed)
        permutation = torch.arange(self.n, device=self.s.device)
        for group in cosine.argsort(stable=True).tensor_split(8):
            order = group[concentration[group].argsort(stable=True)]
            for block in order.split(32):
                permutation[block] = block[torch.randperm(len(block), generator=gen).to(block.device)]
        self.permuted = self.observed[permutation]

    def project_queries(self, q):
        return sphere(q) @ self.u

    def scores(self, x):
        raw = profile_terms(x, self.b, self.s, QUADRATURE_POINTS)[2]
        # In the omitted-anchor objective the removed outer product has known
        # value under each compulsory map, hence the exact correction below.
        correction = (1 - x @ self.a.T) / (self.n * self.unit)
        best, _ = feasible_costs(x, self.b, self.s)
        return {
            "substitution": raw - self.observed + correction,
            "uncalibrated": raw, "fixed_map": x @ self.b.T,
            "fixed_substitution": -best - self.observed_feasible + correction,
            "shuffled_reference": raw - self.permuted + correction,
            "hellinger": (x @ self.u.T) @ self.target.T,
        }

    def check(self, x):
        # Direct double SVD explicitly deletes each candidate from the anchor set.
        matrix = self.source.double().T @ self.target.double() / self.n
        proposed = F.normalize((x.double() @ self.u.double().T))
        measured = self.scores(x)["substitution"][:, :2].double()
        expected = torch.empty_like(measured)
        for col in range(2):
            a, b = F.normalize(self.source[col].double(), dim=0), F.normalize(self.target[col].double(), dim=0)
            minus = matrix - torch.outer(a, b) / self.n
            right = minus - torch.outer(minus @ b, b)
            def optimum(left):
                compressed = right - torch.outer(left, left @ right)
                return left @ minus @ b + torch.linalg.svdvals(compressed).sum()
            natural = optimum(a)
            for row, xv in enumerate(proposed):
                expected[row, col] = (optimum(xv) - natural) / self.unit
        error = (measured - expected).abs().max().item()
        identity = self.scores(self.a[:2])["substitution"][:, :2].diagonal().abs().max().item()
        if error > 2e-4 or identity > 2e-4:
            raise ValueError(f"Explicit leave-one-out substitution mismatch {error=} {identity=}")
        return {"exact_deleted_pair_checks": expected.numel(), "max_error": error,
                "observed_self_substitution_max_error": identity, "spectral_unit": self.unit,
                "words": self.s.numel(), "gallery_readings": self.n}
