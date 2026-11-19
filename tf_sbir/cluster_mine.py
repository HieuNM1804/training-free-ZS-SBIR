"""Transfer WACV 2026 ClusterMine concept voting to an unlabeled photo gallery."""

import torch
from torch.nn import functional as F

from .semantic import agreement_settings


def cluster_assignments(photos, clusters):
    """Spherical k-means on frozen gallery vectors instead of a trained clustering head."""
    rng = torch.Generator(device=photos.device).manual_seed(0)
    centers = photos[torch.randperm(len(photos), generator=rng, device=photos.device)[:clusters]]
    for _ in range(30):
        assignments = (photos @ centers.T).argmax(dim=1)
        totals = torch.zeros_like(centers).index_add_(0, assignments, photos)
        centers = torch.where(totals.norm(dim=1, keepdim=True) > 0, F.normalize(totals, dim=-1), centers)
    return (photos @ centers.T).argmax(dim=1)


def majority_concepts(assignments, words, clusters, vocabulary_size):
    votes = torch.bincount(assignments * vocabulary_size + words, minlength=clusters * vocabulary_size)
    votes = votes.reshape(clusters, vocabulary_size)
    nonempty = votes.sum(dim=1).bool()
    return votes[nonempty].argmax(dim=1).unique()


def cluster_mine_settings(queries, photos, contours, sketch_words, photo_words):
    gallery_geometry = F.normalize(photos - photos.mean(dim=0), dim=-1)
    labels = torch.cat([(batch @ photo_words.T).argmax(dim=1) for batch in photos.split(256)])
    for clusters in (128, 512):
        assignments = cluster_assignments(gallery_geometry, clusters)
        terms = majority_concepts(assignments, labels, clusters, len(photo_words))
        print(f"ClusterMine {clusters} clusters selected {len(terms)} concepts", flush=True)
        variants = agreement_settings(queries, photos, contours, sketch_words[terms], photo_words[terms], topology=(320, "rgb"))
        for name, query, gallery in variants:
            if name.startswith("fuse-k320-rgb"):
                fusion = name.rsplit("-w", 1)[-1]
                yield f"cluster-mine-c{clusters}-w{fusion}", query, gallery
