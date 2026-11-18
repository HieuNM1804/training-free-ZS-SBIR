"""Rerank a common shortlist with paired patch matches and stroke stability."""

from dataclasses import dataclass

import torch
from tqdm.auto import tqdm

PATCH_METHODS = (
    "patch_uniform", "patch_ink", "patch_contour", "patch_independent",
    "patch_consensus", "patch_stable",
)


@dataclass(frozen=True)
class RankingConfig:
    method: str
    top_k: int
    weight: float


def standardize(scores):
    mean = scores.mean(dim=-1, keepdim=True)
    scale = scores.std(dim=-1, correction=0, keepdim=True).clamp_min(1e-6)
    return (scores - mean) / scale


def similarities(query, gallery):
    return torch.matmul(query.unsqueeze(1), gallery.transpose(-1, -2)).float()


def match_support(matrix, sketch_valid, photo_valid):
    best = matrix.masked_fill(~photo_valid.unsqueeze(-2), -1.0).amax(dim=-1)
    valid = sketch_valid.unsqueeze(1)
    return (best * valid).sum(dim=-1) / valid.sum(dim=-1)


def score_shortlist(sketch, gallery, shortlist, methods):
    valid = sketch["valid"]
    photo_valid = gallery["valid"][shortlist]
    rgb = gallery["primary"][shortlist]
    scores = {}
    if "patch_uniform" in methods:
        all_patches = torch.ones_like(valid)
        scores["patch_uniform"] = match_support(
            similarities(sketch["uniform"], rgb), all_patches, photo_valid,
        )
    if methods == {"patch_uniform"}:
        return scores

    rgb_match = similarities(sketch["primary"], rgb)
    if "patch_ink" in methods:
        scores["patch_ink"] = match_support(rgb_match, valid, photo_valid)
    if not (methods - {"patch_uniform", "patch_ink"}):
        return scores

    contour = gallery["secondary"][shortlist]
    contour_match = similarities(sketch["primary"], contour)
    if "patch_contour" in methods:
        scores["patch_contour"] = match_support(contour_match, valid, photo_valid)
    if "patch_independent" in methods:
        scores["patch_independent"] = (
            match_support(rgb_match, valid, photo_valid) + match_support(contour_match, valid, photo_valid)
        ) * 0.5
    agreement = torch.minimum(rgb_match, contour_match)
    if "patch_consensus" in methods:
        scores["patch_consensus"] = match_support(agreement, valid, photo_valid)
    if "patch_stable" in methods:
        thicker = sketch["secondary"]
        perturbed = torch.minimum(similarities(thicker, rgb), similarities(thicker, contour))
        scores["patch_stable"] = match_support(torch.minimum(agreement, perturbed), valid, photo_valid)
    return scores


@torch.inference_mode()
def cache_shortlist_scores(
    queries, photos, sketch_patches, photo_patches, query_ids, photo_ids, maximum_k, methods, batch_size,
):
    ids, local = [], {method: [] for method in methods}
    device = queries.device
    # Keep paired gallery patches on the scoring device and stream query patches.
    gallery = {
        name: value[photo_ids].to(
            device=device, dtype=torch.float32 if device.type == "cpu" and value.is_floating_point() else value.dtype,
        )
        for name, value in photo_patches.items()
    }
    for start in tqdm(range(0, len(queries), batch_size), desc="Score DINO shortlist"):
        end = start + batch_size
        selected = (queries[start:end] @ photos.T).topk(maximum_k, dim=-1).indices
        sketch = {
            name: value[query_ids[start:end]].to(
                device=device, dtype=gallery["primary"].dtype if value.is_floating_point() else value.dtype,
            )
            for name, value in sketch_patches.items()
        }
        ids.append(selected.cpu())
        for name, score in score_shortlist(sketch, gallery, selected, set(methods)).items():
            local[name].append(score.cpu())
    return torch.cat(ids), {name: torch.cat(scores) for name, scores in local.items()}


def reranked_rows(queries, photos, ids, local, config):
    for start in range(0, len(queries), 128):
        end = start + 128
        rows = (queries[start:end] @ photos.T).cpu()
        candidates = ids[start:end, :config.top_k]
        baseline = rows.gather(1, candidates)
        if config.weight:
            local_scores = local[start:end, :config.top_k]
            fused = standardize(baseline) + config.weight * standardize(local_scores)
            order = fused.argsort(dim=1, descending=True, stable=True)
            # Reassign the original shortlist scores, preserving the full-gallery
            # score range and ordering of every photo outside the shortlist.
            rows.scatter_(1, candidates.gather(1, order), baseline.sort(descending=True).values)
        yield from rows
