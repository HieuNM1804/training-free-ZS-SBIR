"""Induce visual reference points for a public vocabulary from replicated naming channels."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Observations:
    # Each of the two readers has (independent gallery halves, words, features).
    source: torch.Tensor
    target: torch.Tensor
    mass: torch.Tensor
    source_mean: torch.Tensor
    target_mean: torch.Tensor
    target_prior: torch.Tensor


def measurements(photos, source_text, target_text, scale, seed=42):
    halves = torch.randperm(len(photos), generator=torch.Generator().manual_seed(seed)).chunk(2)
    means, priors, pooled = [], [], []
    for text in (source_text, target_text):
        prob = (scale * (photos @ text.T)).softmax(1)
        subset_means = []
        for half in halves:
            idx = half.to(photos.device)
            probs = prob[idx]
            subset_means.append((probs.T @ photos[idx]) / probs.sum(0)[:, None].clamp_min(1e-24))
        means.append(torch.stack(subset_means).double())
        pooled.append((prob.T @ photos) / prob.sum(0)[:, None].clamp_min(1e-24))
        priors.append(prob.mean(0))
    mass = (priors[0] * priors[1]).sqrt().double()
    return Observations(means[0], means[1], mass, pooled[0], pooled[1], priors[1])


def cross_gram(source, target):
    products = (source[0] @ target[1].T + source[1] @ target[0].T) / 2
    return (products + products.T) / 2


def select_anchors(obs, limit, kind="replicated", seed=42):
    mass = obs.mass
    if kind == "frequency":
        return mass.argsort(descending=True, stable=True)[:limit]
    if kind == "random":
        generator = torch.Generator().manual_seed(seed)
        return torch.multinomial(mass.cpu(), limit, replacement=False, generator=generator).to(mass.device)
    source, target = obs.source, obs.target
    if kind == "within_fold":
        gram = cross_gram(source, target.flip(0))
    elif kind == "photo_only":
        gram = cross_gram(target, target)
    elif kind == "unreplicated":
        # Conventional single-reader volume, including the variance of its empirical means.
        gram = obs.target_mean.double() @ obs.target_mean.double().T
    elif kind in ("replicated", "unweighted"):
        gram = cross_gram(source, target)
    else:
        raise ValueError(kind)
    # Putting the origin at the first anchor makes every later pivot an affine volume.
    first = mass.argmax().item()
    anchor_ids = [first]
    residual = gram - gram[:, first, None] - gram[first, None, :] + gram[first, first]
    weights = torch.ones_like(mass) if kind == "unweighted" else mass
    chosen = torch.zeros_like(mass, dtype=torch.bool)
    chosen[first] = True
    for _ in range(1, limit):
        diagonal = residual.diagonal()
        scores = (weights * diagonal).masked_fill(chosen | (diagonal <= 1e-11), -torch.inf)
        pivot = int(scores.argmax())
        if not torch.isfinite(scores[pivot]):
            break
        col = residual[:, pivot].clone()
        residual = residual - torch.outer(col, col) / diagonal[pivot]
        chosen[pivot] = True
        anchor_ids.append(pivot)
    return torch.tensor(anchor_ids, device=mass.device)


def project_simplex(matrix):
    ordered = matrix.sort(dim=1, descending=True).values
    count = torch.arange(1, ordered.size(1) + 1, device=ordered.device)
    threshold = (ordered.cumsum(1) - 1) / count
    last = (ordered > threshold).sum(1, keepdim=True).clamp_min(1) - 1
    return (matrix - threshold.gather(1, last)).clamp_min(0)


def barycentric(means, ids, max_steps=400):
    """Simplex-constrained least squares with alternating linear and simplex projections."""
    x = means.double()
    vertices = x[ids]
    offset = vertices.mean(0)
    vertices = vertices - offset
    gram = vertices @ vertices.T
    cov = (x - offset) @ vertices.T
    rho = gram.diagonal().mean().clamp_min(1e-8)
    eye = torch.eye(len(ids), dtype=x.dtype, device=x.device)
    factor = torch.linalg.cholesky(gram + rho * eye)
    inv = torch.cholesky_inverse(factor)
    intercept = cov @ inv
    step = rho * inv
    z = torch.full_like(cov, 1/len(ids))
    dual = torch.zeros_like(cov)
    for iteration in range(max_steps):
        current = intercept + (z - dual) @ step
        updated = project_simplex(current + dual)
        dual = dual + current - updated
        if iteration % 16 == 15 and max((current-updated).abs().max(), (updated-z).abs().max()) < 2e-6:
            z = updated
            break
        z = updated
    # A nominated anchor encodes itself by construction, including near-degenerate vertices.
    z[ids] = eye
    if (z.sum(1)-1).abs().max() > 1e-5 or z.min() < 0:
        raise ValueError("Invalid simplex coordinates")
    return z.float()


def index(obs, photos, queries, ids, *, query_mode="barycentric"):
    source_mix = barycentric(obs.source_mean, ids)
    target_mix = barycentric(obs.target_mean, ids)
    if query_mode == "barycentric":
        source = queries @ source_mix
    elif query_mode == "restricted":
        source = F.normalize(queries[:, ids], p=1, dim=1)
    else:
        raise ValueError(query_mode)
    centers = obs.target_mean[ids]
    # Log Gaussian density up to terms shared by all components for each photo.
    evidence = photos @ centers.T - .5 * centers.square().sum(1)
    prior = obs.target_prior @ target_mix
    return source, evidence, prior.clamp_min(1e-20).log(), source_mix, target_mix


def fixture():
    points = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[.5,.5,0.]], dtype=torch.float64)
    weights = torch.tensor([2.,1.,.9,.8,.1], dtype=torch.float64)
    def observed(target):
        return Observations(points.repeat(2,1,1), target.repeat(2,1,1), weights,
                            points.float(), target.float(), weights.float()/weights.sum())
    aligned = observed(points)
    shifted = observed(points[:, [2,1,0]])
    a = select_anchors(aligned, 2).tolist()
    b = select_anchors(shifted, 2).tolist()
    unpaired = select_anchors(shifted, 2, "unreplicated").tolist()
    coeff = barycentric(points.float(),torch.tensor([0,1,2,3]))[-1]
    assert a == [0,1] and b == [0,2] and unpaired == a
    assert torch.allclose(coeff, torch.tensor([0.,.5,.5,0.]), atol=1e-5)
    return {"aligned":a, "exchanged_axes":b, "within_reader_only":unpaired, "mixture":coeff.tolist()}
