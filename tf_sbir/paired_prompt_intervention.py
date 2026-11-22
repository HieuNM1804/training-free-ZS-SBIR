"""Per-image closed-form head reweighting under paired domain prompts."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def center(items):
    return items - items.mean(1, keepdim=True)


def project(rows, vector):
    return (rows @ vector.unsqueeze(-1)).squeeze(-1)


def exchange_difference(source, target):
    difference = source - target
    exchanged = source - target.roll(1, dims=1)
    return F.normalize(exchanged, dim=-1) * difference.norm(dim=-1, keepdim=True)


@dataclass
class Panel:
    midpoint: torch.Tensor
    design: torch.Tensor
    response: torch.Tensor
    penalty: torch.Tensor

    @classmethod
    def measure(cls, q, basis, midpoint, difference):
        content = center(midpoint @ basis.transpose(1, 2))
        domain = center(difference @ basis.transpose(1, 2))
        response = center(project(difference, q))
        # The trace is computed on each individual sketch and panel.
        scale = content.square().mean((1, 2)).clamp_min(1e-14)
        return cls(content, domain, response, scale)

    def fit(self, ridge, strength):
        n = self.midpoint.size(1)
        directions = self.midpoint.size(2)
        mm = self.midpoint.transpose(1, 2) @ self.midpoint / n
        dd = self.design.transpose(1, 2) @ self.design / n
        d0 = project(self.design.transpose(1, 2), self.response) / n
        identity = torch.eye(directions, device=mm.device, dtype=mm.dtype)
        matrix = mm + strength * dd + (ridge * self.penalty)[:, None, None] * identity
        return torch.linalg.solve(matrix, (-strength * d0).unsqueeze(-1)).squeeze(-1)

    def shrink(self, coefficient, ridge, strength):
        cm = project(self.midpoint, coefficient)
        cd = project(self.design, coefficient)
        numerator = -strength * (self.response * cd).mean(1)
        denominator = (cm.square() + strength * cd.square()).mean(1)
        denominator += ridge * self.penalty * coefficient.square().sum(1)
        return (numerator / denominator.clamp_min(1e-20)).clamp(0, 1)


def fit_panels(q, basis, source, target, exchange=False):
    panels = []
    for parity in (0, 1):
        src, tgt = source[:, parity::2], target[:, parity::2]
        difference = exchange_difference(src, tgt) if exchange else src - tgt
        panels.append(Panel.measure(q, basis, (src + tgt) * .5, difference))
    return panels


def intervention(basis, panels, ridge, strength):
    a, b = panels
    ca, cb = a.fit(ridge, strength), b.fit(ridge, strength)
    ab, ba = b.shrink(ca, ridge, strength), a.shrink(cb, ridge, strength)
    coefficients = (ab[:, None] * ca + ba[:, None] * cb) * .5
    raw = (ca + cb) * .5
    return (project(basis.transpose(1, 2), coefficients),
            project(basis.transpose(1, 2), raw), (ab + ba) * .5)


def preserve_spread(q, step, midpoint):
    before = center(project(midpoint, q)).square().mean(1).sqrt()
    after = center(project(midpoint, q + step)).square().mean(1).sqrt()
    ratio = torch.where((before > 1e-8) & (after > 1e-8), before / after.clamp_min(1e-8), 1.)
    return (q + step) * ratio[:, None]


def text_reference(source, target, components):
    diff = source - target
    diff = diff - diff.mean(0)
    _, vec = torch.linalg.eigh(diff.T @ diff / len(diff))
    return vec[:, -components:].T.contiguous()


def reference_bases(heads, dictionary_axes, perm, sign):
    # Recover the same Gram matrix in a common text-only subspace.
    # TF32 dot products can break definiteness for redundant heads.
    wide = heads.double()
    gram = wide @ wide.transpose(1, 2)
    ridge = gram.diagonal(dim1=1, dim2=2).mean(1).clamp_min(1e-12) * 1e-7
    identity = torch.eye(gram.size(1), device=heads.device, dtype=gram.dtype)
    factor = torch.linalg.cholesky(gram + ridge[:, None, None] * identity)
    text = (factor @ dictionary_axes.double()).float()
    rotated = heads.index_select(2, perm) * sign
    return {"heads": heads, "rotate": rotated, "text": text, "exchange": heads}


def holdout_measure(q, edited, midpoint, difference):
    domain0 = center(project(difference, q))
    domain1 = center(project(difference, edited))
    mid0 = center(project(midpoint, q))
    mid1 = center(project(midpoint, edited))
    ratio = domain1.square().mean(1) / domain0.square().mean(1).clamp_min(1e-12)
    content = F.cosine_similarity(mid0, mid1)
    shape_change = (mid1 - mid0).square().mean(1).sqrt() / mid0.square().mean(1).sqrt().clamp_min(1e-8)
    return torch.stack((1 - ratio, content, shape_change, (edited - q).norm(dim=-1)), 1)
