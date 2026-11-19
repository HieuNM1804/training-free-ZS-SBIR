"""Verify crop proposals with whole-sketch evidence and gallery Gaussian moments."""

from functools import partial
import math

import torch
from torch.nn import functional as F

from .reranking import standardize
from .union_density import density_blocks, density_rows, select_density_rows
from .view_hypotheses import view_concepts


def residual_covariance(features, neighbors):
    """Estimate within-neighborhood variation from the unlabeled gallery."""
    pieces = []
    for gallery, ids in zip(features.split(256), neighbors.split(256), strict=True):
        candidates = features[ids]
        similarity = (gallery[:, None] * candidates).sum(dim=-1)
        nearest = similarity.topk(min(32, ids.size(1)), dim=-1).indices
        cluster = candidates.gather(1, nearest[..., None].expand(-1, -1, features.size(-1)))
        pieces.append(gallery - cluster.mean(dim=1))
    residuals = torch.cat(pieces)
    residuals = residuals - residuals.mean(dim=0)
    covariance = residuals.T @ residuals / len(residuals)
    ridge = 0.25 * covariance.diagonal().mean().clamp_min(1e-7)
    covariance.diagonal().add_(ridge)
    return covariance


def diagonal_log_likelihood(observations, means, variance):
    """Gaussian log density per feature, up to a common additive constant."""
    offsets = observations[:, None] - means[:, :, None]
    square = offsets.square() / variance[:, :, None]
    return -0.5 * (square + variance[:, :, None].log()).mean(dim=-1)


def proposal_evidence(probes, features, bank, heads, score_batch):
    covariance = residual_covariance(features, bank)
    root = torch.linalg.cholesky(covariance)
    reference_variance = covariance.diagonal()
    inverse_root = torch.linalg.solve_triangular(root, torch.eye(len(root), device=root.device, dtype=root.dtype), upper=False)
    white_probes = probes @ inverse_root.T
    depth = min(heads.size(-1), 128)
    storage = features.half() if features.is_cuda else features
    batch_size = min(64, score_batch)
    results = []
    for query, whitened, ids in zip(probes.split(batch_size), white_probes.split(batch_size), heads.split(batch_size), strict=True):
        samples = storage[ids[..., :depth]].float()
        variance, means = torch.var_mean(samples, dim=-2, correction=0)
        means_white = means @ inverse_root.T
        isotropic = -((query[:, None] - means[:, :, None]).square().mean(dim=-1))
        shared = -((whitened[:, None] - means_white[:, :, None]).square().mean(dim=-1))
        diagonal = diagonal_log_likelihood(query, means, 0.5 * (variance + reference_variance))
        raw = torch.stack((isotropic, shared, diagonal)).transpose(-1, -2)
        results.append(standardize(raw).mean(dim=-2))
    return torch.cat(results, dim=1)


def ranked_prefixes(rankings, score_batch):
    width = min(rankings[0][0].size(1), 256)
    heads = []
    for prior, prefix, weight in rankings:
        if weight is None:
            heads.append(prefix[..., :width])
        else:
            heads.append(torch.cat([scores.topk(width, dim=-1).indices
                                    for scores in density_blocks(prior, prefix, score_batch, weight)]))
    return torch.stack(heads, dim=1)


def evidence_rank_fusion(heads, posterior, gallery_size, score_batch):
    rank = torch.arange(1, heads.size(-1) + 1, device=heads.device, dtype=posterior.dtype)
    prefixes = []
    for ids, mass in zip(heads.split(score_batch), posterior.split(score_batch), strict=True):
        votes = mass[..., None] / (60 + rank)
        scores = mass.new_zeros((len(ids), gallery_size))
        scores.scatter_add_(1, ids.flatten(1), votes.flatten(1))
        prefixes.append(scores.topk(heads.size(-1), dim=-1).indices)
    return torch.cat(prefixes)


def gaussian_verification_settings(
    rankings, focused_count, queries, ink, original_embedding, pooled_embedding, contours, gallery, words,
    sketch_words, photo_words, banks, score_batch,
):
    raw = torch.stack((queries, F.normalize(ink.float(), dim=-1)), dim=1)
    lexical = view_concepts(raw, sketch_words, photo_words)
    visual = torch.stack((original_embedding, pooled_embedding), dim=1)
    shape = F.normalize(contours.float().mean(dim=1), dim=-1)
    heads = ranked_prefixes(rankings, score_batch)
    spaces = ((raw, shape, banks[0]), (visual, gallery, banks[0]), (lexical, words, banks[1]))
    evidence = torch.stack([proposal_evidence(query, photos, bank, heads, score_batch) for query, photos, bank in spaces])
    isotropic, shared, diagonal = standardize(evidence.mean(dim=0))

    for name, scores in (("isotropic", isotropic), ("shared", shared), ("diagonal", diagonal)):
        choice = scores.argmax(dim=-1)
        print("verification", name, "switches", (choice != 0).sum().item(), flush=True)
        yield f"verify-{name}", partial(select_density_rows, rankings, choice, score_batch)

    agreement = standardize(0.5 * (shared + diagonal))
    for odds in (1, 4, len(rankings) - 1):
        posterior = agreement.clone()
        posterior[:, 0] += math.log(odds)
        choice = posterior.argmax(dim=-1)
        print("verification", f"joint-prior{odds}", "switches", (choice != 0).sum().item(), flush=True)
        yield f"verify-joint-prior{odds}", partial(select_density_rows, rankings, choice, score_batch)

    shortlist = standardize(agreement[:, :focused_count])
    shortlist[:, 0] += math.log(focused_count - 1)
    choice = shortlist.argmax(dim=-1)
    print("verification", "joint-small", "switches", (choice != 0).sum().item(), flush=True)
    yield "verify-joint-small", partial(select_density_rows, rankings[:focused_count], choice, score_batch)

    prior = rankings[0][0]
    prefix = evidence_rank_fusion(heads, posterior.softmax(dim=-1), prior.size(-1), score_batch)
    yield "verify-joint-rank-fusion", partial(density_rows, prior, prefix, score_batch, None)
