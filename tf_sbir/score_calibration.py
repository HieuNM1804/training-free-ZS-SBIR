"""Estimate cross-modal gallery hubness using only gallery-derived sketch proxies."""

import torch
from torch.nn import functional as F

from .global_methods import centered
from .semantic import agreement_settings


def synthetic_domain_priors(proxies, targets, photo_geometry):
    """Score out-of-pair contour queries, optionally ignoring close RGB neighbors."""
    temperature = 0.055
    neighbors = min(128, len(targets) - 2)
    priors = {"inverted-softmax": [], "inbound-tail": [], "other-neighborhoods": []}
    for start in range(0, len(targets), 512):
        end = min(start + 512, len(targets))
        similarities = proxies @ targets[start:end].T
        local = torch.arange(end - start, device=targets.device)
        similarities[local + start, local] = -torch.inf
        priors["inverted-softmax"].append(temperature * (similarities / temperature).logsumexp(dim=0))
        priors["inbound-tail"].append(similarities.topk(neighbors, dim=0).values.mean(dim=0))
        near = (photo_geometry @ photo_geometry[start:end].T).topk(neighbors, dim=0).indices
        similarities.scatter_(0, near, -torch.inf)
        priors["other-neighborhoods"].append(temperature * (similarities / temperature).logsumexp(dim=0))
    return {name: torch.cat(blocks) for name, blocks in priors.items()}


def normalized_score_rows(queries, gallery, offset, batch_size, activation=None):
    """Preserve ordering while keeping adjusted scores within the evaluator's cosine range."""
    scale = 1 + offset.abs().max()
    for batch in queries.split(batch_size):
        scores = batch @ gallery.T
        adjusted = (scores - offset) / scale
        if activation is not None:
            adjusted = torch.where(activation[scores.argmax(dim=1), None], adjusted, scores)
        yield from adjusted.cpu()


def calibration_features(queries, photos, contours, sketch_words, photo_words):
    proxies = F.normalize(contours.mean(dim=1), dim=-1)
    settings = agreement_settings(
        torch.cat((queries, proxies)), photos, contours, sketch_words, photo_words, topology=(320, "rgb"),
    )
    combined, gallery = next((q, p) for name, q, p in settings if name == "fuse-k320-rgb-w0.6")
    query, proxy = combined.split((len(queries), len(proxies)))
    return query, proxy, gallery


def calibrated_settings(queries, photos, contours, sketch_words, photo_words):
    query, proxy, gallery = calibration_features(queries, photos, contours, sketch_words, photo_words)
    yield "uncalibrated-k320", query, gallery, None
    bank = synthetic_domain_priors(proxy, gallery, centered(photos, photos.mean(dim=0)))
    for method, prior in bank.items():
        offset = prior - prior.mean()
        for weight in (0.25, 0.50, 0.90):
            yield f"{method}-w{weight:g}", query, gallery, (offset * weight, None)


def dynamic_activation(probe_scores, neighbors):
    """CVPR 2022 QB-Norm, Equation (2)."""
    indices = probe_scores.topk(neighbors, dim=1).indices
    activated = torch.zeros(probe_scores.size(1), dtype=torch.bool, device=probe_scores.device)
    return activated.scatter_(0, indices.flatten(), True)


def sinkhorn_target_hubness(probe_scores, temperature, iterations=10):
    """ACM MM 2025, Algorithm A1 with the gallery-derived query bank."""
    kernel = ((probe_scores - probe_scores.amax(dim=1, keepdim=True)) / temperature).exp()
    rows, columns = kernel.shape
    beta = kernel.new_ones(columns)
    for _ in range(iterations):
        alpha = (kernel @ beta).clamp_min(1e-30).reciprocal() / rows
        beta = (kernel.T @ alpha).clamp_min(1e-30).reciprocal() / columns
    return -temperature * beta.log()


def querybank_settings(queries, photos, contours, sketch_words, photo_words):
    query, proxies, gallery = calibration_features(queries, photos, contours, sketch_words, photo_words)
    rng = torch.Generator(device=proxies.device).manual_seed(0)
    sample = torch.randperm(len(proxies), generator=rng, device=proxies.device)[:4096]
    probe_scores = proxies[sample] @ gallery.T
    yield "uncalibrated-k320", query, gallery, None
    for temperature in (0.025, 0.055):
        bias = temperature * (probe_scores / temperature).logsumexp(dim=0)
        offset = bias - bias.mean()
        yield f"qb-inverted-softmax-t{temperature:g}", query, gallery, (offset, None)
        for k in (1, 5):
            activation = dynamic_activation(probe_scores, k)
            yield f"qb-dynamic-is-t{temperature:g}-k{k}", query, gallery, (offset, activation)
        balanced = sinkhorn_target_hubness(probe_scores, temperature)
        bias = balanced - balanced.mean()
        yield f"sinkhorn-querybank-t{temperature:g}", query, gallery, (bias, None)
