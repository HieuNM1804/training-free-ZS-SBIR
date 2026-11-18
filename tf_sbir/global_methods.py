"""Training-free global scores and contour-based gallery calibration."""

import torch
from torch.nn import functional as F

GLOBAL_METHODS = (
    "rgb", "contour", "rrf", "centered", "query_expansion", "local_residual",
    "procrustes",
)


def centered(features, mean):
    return F.normalize(features - mean, dim=-1)


def reciprocal_ranks(scores, constant=60):
    indices = scores.argsort(dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(indices)
    ranks.scatter_(1, indices, torch.arange(1, scores.size(1) + 1, device=scores.device).expand_as(indices))
    return (ranks.float() + constant).reciprocal()


@torch.inference_mode()
def prepare_methods(queries, gallery, contours, methods):
    """Map a method to its query/gallery matrices, or an explicit scoring function."""
    result = {}
    if "rgb" in methods:
        result["rgb"] = (queries, gallery)
    if contours is None:
        return result
    proxies = F.normalize(contours.mean(dim=1), dim=-1)
    if "contour" in methods:
        result["contour"] = (queries, proxies)
    if "rrf" in methods:
        def fuse(batch):
            return reciprocal_ranks(batch @ gallery.T) + reciprocal_ranks(batch @ proxies.T)

        result["rrf"] = fuse

    q = centered(queries, proxies.mean(dim=0))
    p = centered(gallery, gallery.mean(dim=0))
    z = centered(proxies, proxies.mean(dim=0))
    if "centered" in methods:
        result["centered"] = (q, p)
    if {"query_expansion", "local_residual"} & set(methods):
        photo_neighbors, contour_neighbors = [], []
        for batch in q.split(256):
            values, indices = (batch @ z.T).topk(min(8, len(z)))
            weights = (values / 0.07).softmax(dim=1).unsqueeze(-1)
            photo_neighbors.append((p[indices] * weights).sum(dim=1))
            contour_neighbors.append((z[indices] * weights).sum(dim=1))
        photo_average = torch.cat(photo_neighbors)
        if "query_expansion" in methods:
            result["query_expansion"] = (F.normalize(q + 0.5 * photo_average, dim=-1), p)
        if "local_residual" in methods:
            offset = photo_average - torch.cat(contour_neighbors)
            result["local_residual"] = (F.normalize(q + 0.5 * offset, dim=-1), p)
    if "procrustes" in methods:
        # Solve the orthogonal alignment in closed form on paired gallery views.
        left, _, right = torch.linalg.svd((z.T @ p).cpu(), full_matrices=False)
        rotation = (left @ right).to(q.device)
        result["procrustes"] = (F.normalize(q @ rotation, dim=-1), p)
    return result
