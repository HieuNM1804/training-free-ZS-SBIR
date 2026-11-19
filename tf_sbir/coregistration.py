"""Gallery spectral projections with co-regularization across representations."""

from functools import partial

import torch
from torch.nn import functional as F

from .gallery_diffusion import mutual_transition
from .global_methods import centered
from .reranking import standardize
from .semantic_transport import transport_components


def projected_laplacian(transition, vectors, partners, strength):
    """Apply Eq. 6 of Kumar et al. without forming a dense projector."""
    values = torch.sparse.mm(transition, vectors)
    for other in partners:
        values.addmm_(other, other.T @ vectors, beta=1, alpha=strength)
    return values


def leading_basis(transition, initial, partners=(), strength=0., iterations=18):
    basis = initial
    for _ in range(iterations):
        laplacian = projected_laplacian(transition, basis, partners, strength)
        basis = torch.linalg.qr(basis + laplacian).Q
    small = basis.T @ projected_laplacian(transition, basis, partners, strength)
    _, rotation = torch.linalg.eigh((small + small.T) * 0.5)
    return basis @ rotation.flip(dims=(1,))


def spectral_gallery(geometries, rank, neighbors=160):
    transitions = [
        mutual_transition(centered(view, view.mean(dim=0)), neighbors)
        for view in geometries
    ]
    sample = transitions[0]
    rng = torch.Generator(device=sample.device).manual_seed(0)
    initial = torch.randn((sample.size(0), rank + 8), generator=rng, device=sample.device)
    initial = torch.linalg.qr(initial).Q
    independent = [leading_basis(graph, initial, iterations=30) for graph in transitions]
    plain = independent[0][:, :rank]
    yield "rgb-only", centered(plain, plain.mean(dim=0))
    for graph_name, num_views in (("rgb-text", 2), ("rgb-text-contour", 3)):
        bases = list(independent[:num_views])
        for _ in range(4):
            for index, transition in enumerate(transitions[:num_views]):
                partners = [other[:, :rank] for step, other in enumerate(bases) if index != step]
                bases[index] = leading_basis(transition, bases[index], partners, 0.25, iterations=8)
        joint = torch.cat([basis[:, :rank] for basis in bases], dim=1)
        yield graph_name, centered(joint, joint.mean(dim=0))


def spectral_rows(query, gallery, offset, coordinates, batch_size, neighbors, weight):
    scale = 1 + offset.abs().max()
    for batch in query.split(batch_size):
        baseline = (batch @ gallery.T - offset) / scale
        top = baseline.topk(neighbors, dim=1)
        mass = (top.values / 0.04).softmax(dim=1)
        projected = F.normalize((coordinates[top.indices] * mass.unsqueeze(-1)).sum(1), dim=-1)
        score = standardize(baseline) + weight * standardize(projected @ coordinates.T)
        yield from (0.15 * score).tanh().cpu()


def coregistration_settings(queries, photos, contours, sketch_words, photo_words, score_batch):
    query, gallery, offset, memory, _ = transport_components(queries, photos, contours, sketch_words, photo_words)
    proxies = F.normalize(contours.mean(dim=1), dim=-1)
    for rank in (32, 64):
        views = (photos, memory.gallery_words, proxies)
        for name, coordinates in spectral_gallery(views, rank):
            for neighbors, weight in ((64, 0.35), (256, 0.7)):
                label = f"coreg-spectral-{name}-r{rank}-k{neighbors}-w{weight:g}"
                yield label, partial(spectral_rows, query, gallery, offset, coordinates, score_batch, neighbors, weight)
