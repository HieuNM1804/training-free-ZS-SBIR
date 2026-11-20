"""Smooth independent query evidence over a photo gallery with two CLIP similarity graphs.

The full Gaussian status matrices and local row-stochastic transitions follow Wang et al.,
CVPR 2012, Eqs. 1–6. Applying them to a signal avoids forming their dense products.
"""

from dataclasses import dataclass, replace

import torch
from torch.nn import functional as F

from .gallery_diffusion import nearest_gallery


@dataclass(frozen=True)
class Config:
    neighbors: int = 64
    bandwidth: float = .36
    iterations: int = 1
    mode: str = "cross"
    normalize_mass: bool = True
    shuffle_semantic: bool = False


def canonical():
    return Config()


def candidates():
    return [Config(k, mu, t, view) for k in (16, 64, 256) for mu in (.06, .36)
            for t in (1, 2) for view in ("rgb", "semantic", "cross")]


def controls(c):
    return {"primary": c, "direct_concepts": replace(c, mode="direct"),
            "same_depth_rgb": replace(c, mode="rgb"), "same_depth_semantic": replace(c, mode="semantic"),
            "crossed": replace(c, mode="cross"), "separate_paths": replace(c, mode="independent"),
            "initial_status_only": replace(c, mode="cross", iterations=0),
            "unnormalized_cross": replace(c, mode="cross", normalize_mass=False),
            "shuffled_semantic_edges": replace(c, mode="cross", shuffle_semantic=True)}


def cache_key(c):
    return c.mode == "direct", c.shuffle_semantic, c.neighbors, c.bandwidth, c.iterations, c.mode


class Kernel:
    def __init__(self, features, top, k, mu, permutation=None):
        ids, similarity = top
        ids, similarity = ids[:, :k], similarity[:, :k]
        distance_sq = (2 * (1 - similarity)).clamp_min(0)
        bandwidth = mu * distance_sq.sqrt().mean().square().clamp_min(1e-12)
        weights = (-distance_sq / bandwidth).softmax(1)
        n = len(features)
        device = features.device
        if permutation is not None:
            # Reassign the entire graph to unrelated photos, preserving its geometry.
            ids = permutation.argsort()[ids[permutation]]
            weights = weights[permutation]
            features = features[permutation]
        rows = torch.arange(n, device=device).repeat_interleave(k)
        sparse = torch.sparse_coo_tensor(torch.stack((rows, ids.flatten())), weights.flatten(),
                                         (n, n)).coalesce()
        self.local = sparse.to_sparse_csr()
        self.reverse = sparse.transpose(0, 1).coalesce().to_sparse_csr()
        self.status = features.new_empty((n, n))
        for first in range(0, n, 384):
            x = features[first:first + 384]
            squared = (2 * (1 - x @ features.T)).clamp_min_(0)
            self.status[first:first + len(x)] = (-squared / bandwidth).softmax(1)
        self.bandwidth = float(bandwidth)


class Index:
    def __init__(self, data, *, rgb_only=False):
        self.d = data
        self.features = (data.p,)
        if not rgb_only:
            self.features += (F.normalize(data.a @ data.target, dim=1),)
        self.neighbors = [nearest_gallery(x, 256) for x in self.features]
        self.permutation = torch.randperm(len(data.p), generator=torch.Generator().manual_seed(17491)).to(data.p.device)
        self.cache = None
        self.kernels = []
        self.info = {}

    def prepare(self, c):
        key = (c.neighbors, c.bandwidth, c.shuffle_semantic)
        if key == self.cache:
            return
        self.kernels.clear()
        for i, (features, top) in enumerate(zip(self.features, self.neighbors, strict=True)):
            permutation = self.permutation if i == 1 and c.shuffle_semantic else None
            self.kernels.append(Kernel(features, top, c.neighbors, c.bandwidth, permutation))
        self.info[str(key)] = {"gaussian_denominators": [v.bandwidth for v in self.kernels]}
        self.cache = key

    def path(self, signals, start, c):
        views = [start]
        for _ in range(c.iterations):
            views.append(1 - views[-1] if c.mode == "cross" else start)
        for i in views[:-1]:
            signals = torch.sparse.mm(self.kernels[i].reverse, signals)
        signals = self.kernels[views[-1]].status @ signals
        for i in reversed(views[:-1]):
            signals = torch.sparse.mm(self.kernels[i].local, signals)
        return signals

    def score(self, c, evidence=None):
        if evidence is None:
            evidence = self.d.base
        if c.mode == "direct":
            return evidence
        if len(self.features) == 1 and c.mode != "rgb":
            raise ValueError("An RGB-only index requires the RGB propagation mode")
        self.prepare(c)
        previous_precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("high")
            return self.propagate(c, evidence)
        finally:
            torch.set_float32_matmul_precision(previous_precision)

    def propagate(self, c, evidence=None):
        if evidence is None:
            evidence = self.d.base
        starts = (0,) if c.mode == "rgb" else (1,) if c.mode == "semantic" else (0, 1)
        def apply(signals):
            combined = self.path(signals, starts[0], c)
            if len(starts) == 2:
                combined.add_(self.path(signals, starts[1], c)).mul_(.5)
            return combined
        # Kernel row normalization makes output the expected evidence of a neighboring
        # photo. The raw status matrix is also evaluated as a prospective ablation.
        mass = apply(evidence.new_ones((len(self.d.p), 1))).clamp_min_(1e-30) if c.normalize_mass else 1.
        scores = torch.empty_like(evidence)
        for first in range(0, len(scores), 384):
            x = evidence[first:first + 384].T.contiguous()
            scores[first:first + x.size(1)] = (apply(x) / mass).T
        return scores
