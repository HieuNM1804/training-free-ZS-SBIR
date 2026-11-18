"""Training-free retrieval on a graph of unlabeled gallery photographs."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class GalleryGraph:
    indices: torch.Tensor
    weights: torch.Tensor

    def step(self, values):
        """Apply the row-normalized gallery transition to node features."""
        outputs = []
        for indices, weights in zip(self.indices.split(128), self.weights.split(128), strict=True):
            outputs.append((values[indices] * weights.unsqueeze(-1)).sum(dim=1))
        return torch.cat(outputs)

    def diffuse(self, values, restart, steps):
        initial = values
        for _ in range(steps):
            values = restart * initial + (1.0 - restart) * self.step(values)
        return values


def build_graph(photos, proxies, k, shape_weight=0.0, reciprocal=False):
    """Build a photo semantic graph, optionally regularized by paired visual proxies."""
    n = len(photos)
    if not 0 < k < n:
        raise ValueError("k must be positive and smaller than the gallery")
    # Half precision is used only to construct the approximate gallery graph.
    # Query scoring and the reference evaluator retain float32.
    if photos.is_cuda:
        photos = photos.half()
        proxies = proxies.half()
    neighbor_ids, values = [], []
    for start in range(0, n, 256):
        end = min(start + 256, n)
        sim = (photos[start:end] @ photos.T).float()
        if shape_weight:
            sim.lerp_((proxies[start:end] @ proxies.T).float(), shape_weight)
        block = torch.arange(end - start, device=sim.device)
        sim[block, block + start] = -torch.inf
        score, indices = sim.topk(k, dim=1)
        neighbor_ids.append(indices)
        values.append(score)
    indices = torch.cat(neighbor_ids)
    weights = (torch.cat(values) / 0.07).softmax(dim=1)
    if reciprocal:
        # Search sorted encoded edge IDs for the reversed edge instead of
        # materializing a dense gallery-by-gallery indicator.
        rows = torch.arange(n, device=indices.device).unsqueeze(1)
        edge_ids = (rows * n + indices).flatten().sort().values
        reverse = indices * n + rows
        positions = torch.searchsorted(edge_ids, reverse).clamp_max(len(edge_ids) - 1)
        both = edge_ids[positions] == reverse
        weights = weights * torch.where(both, 1.0, 0.25)
        weights = weights / weights.sum(dim=1, keepdim=True)
    return GalleryGraph(indices, weights)


def add_gallery_feedback(queries, photos, top_k, weight, temperature=0.07, seed_gallery=None):
    """Expand one incoming query using the gallery's semantic neighborhood."""
    bank = photos if seed_gallery is None else seed_gallery
    result = []
    for batch in queries.split(256):
        similarities, indices = (batch @ bank.T).topk(top_k, dim=1)
        votes = (similarities / temperature).softmax(dim=1)
        supported = (photos[indices] * votes.unsqueeze(-1)).sum(dim=1)
        result.append(F.normalize(batch + weight * supported, dim=-1))
    return torch.cat(result)


def fused_features(first_q, first_p, second_q, second_p, weight):
    """Represent a fixed weighted sum of two cosine similarities as one cosine."""
    if weight == 0:
        return first_q, first_p
    first_multiplier = (1.0 - weight) ** 0.5
    second_multiplier = weight ** 0.5
    return (
        torch.cat((first_multiplier * first_q, second_multiplier * second_q), dim=1),
        torch.cat((first_multiplier * first_p, second_multiplier * second_p), dim=1),
    )
