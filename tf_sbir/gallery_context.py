"""Training-free gallery context through shared neighbors and spectral coordinates."""

import torch
from torch.nn import functional as F

from .manifold import GalleryGraph


def contextual_graph(photos, initial, neighborhood, k, overlap_weight):
    """Combine visual similarity with the fraction of neighbors two images share."""
    n = len(photos)
    compute_type = torch.float16 if photos.is_cuda else torch.float32
    features = photos.to(compute_type)
    neighbors = torch.zeros((n, n), dtype=compute_type, device=photos.device)
    neighbors.scatter_(1, initial.indices[:, :neighborhood], 1)
    neighbors.fill_diagonal_(1)
    distances, indices = [], []
    for start in range(0, n, 512):
        end = min(start + 512, n)
        shared = (neighbors[start:end] @ neighbors.T).float() / (neighborhood + 1)
        similarity = (features[start:end] @ features.T).float().lerp_(shared, overlap_weight)
        row = torch.arange(end - start, device=photos.device)
        similarity[row, row + start] = -torch.inf
        values, ids = similarity.topk(k, dim=1)
        distances.append(values)
        indices.append(ids)
    return GalleryGraph(torch.cat(indices), (torch.cat(distances) / 0.07).softmax(dim=1))


def spectral_coordinates(graph, rank, iterations=12):
    """Approximate the leading nonconstant eigenvectors of the normalized gallery graph."""
    n, k = graph.indices.shape
    device = graph.weights.device
    source = torch.arange(n, device=device).repeat_interleave(k)
    target = graph.indices.flatten()
    weights = graph.weights.flatten()
    degree = graph.weights.sum(dim=1).scatter_add_(0, target, weights)
    weights = weights * degree[source].rsqrt() * degree[target].rsqrt()
    edges = torch.stack((torch.cat((source, target)), torch.cat((target, source))))
    normalized = torch.sparse_coo_tensor(edges, weights.repeat(2), (n, n)).coalesce().to_sparse_csr()
    random = torch.Generator(device=device).manual_seed(0)
    basis = torch.randn((n, min(n, rank + 9)), generator=random, device=device)
    for _ in range(iterations):
        basis = torch.linalg.qr(basis + torch.sparse.mm(normalized, basis)).Q
    projected = basis.T @ torch.sparse.mm(normalized, basis)
    eigenvalues, coefficients = torch.linalg.eigh((projected + projected.T).cpu() * 0.5)
    scale = eigenvalues[-rank - 1:-1].to(device).clamp_min(0).square()
    vectors = basis @ coefficients[:, -rank - 1:-1].to(device)
    return F.normalize(vectors * scale, dim=-1)


def spectral_query(queries, seed_gallery, coordinates, neighbors=96):
    """Project individual sketches with a soft vote over their candidate photos."""
    result = []
    for batch in queries.split(256):
        values, ids = (batch @ seed_gallery.T).topk(neighbors, dim=1)
        weights = (values / 0.045).softmax(dim=1)
        result.append(F.normalize((coordinates[ids] * weights.unsqueeze(-1)).sum(dim=1), dim=-1))
    return torch.cat(result)
