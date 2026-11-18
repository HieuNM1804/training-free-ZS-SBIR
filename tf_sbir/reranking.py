"""Rerank a common shortlist with paired patch matches and stroke stability."""

from dataclasses import dataclass
import os

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from .metrics import cosine_batches

PATCH_METHODS = (
    "patch_uniform", "patch_ink", "patch_contour", "patch_independent",
    "patch_consensus", "patch_distinctive", "patch_stable",
    "patch_centered_ink", "patch_centered_consensus",
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


def match_support(matrix, sketch_valid, photo_valid, distinctiveness=False):
    best = matrix.masked_fill(~photo_valid.unsqueeze(-2), -1.0).amax(dim=-1)
    weights = sketch_valid.unsqueeze(1)
    if distinctiveness:
        # Strokes that match nearly every candidate carry little ranking evidence.
        top = best.topk(max(1, best.size(1) // 10), dim=1).values.mean(dim=1, keepdim=True)
        contrast = (top - best.mean(dim=1, keepdim=True)).clamp_min(0)
        weights = weights * (contrast + 1e-6)
    return (best * weights).sum(dim=-1) / weights.sum(dim=-1)


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
    if "patch_distinctive" in methods:
        scores["patch_distinctive"] = match_support(agreement, valid, photo_valid, distinctiveness=True)
    if "patch_stable" in methods:
        thicker = sketch["secondary"]
        perturbed = torch.minimum(similarities(thicker, rgb), similarities(thicker, contour))
        scores["patch_stable"] = match_support(torch.minimum(agreement, perturbed), valid, photo_valid)
    return scores


def gallery_patch_mean(patches, valid):
    totals = [
        (chunk * mask.unsqueeze(-1)).sum(dim=(0, 1), dtype=torch.float32)
        for chunk, mask in zip(patches.split(256), valid.split(256), strict=True)
    ]
    return torch.stack(totals).sum(dim=0) / valid.sum()


def center_patch_blocks(patches, mean):
    result = torch.empty_like(patches)
    for start in range(0, len(patches), 256):
        block = patches[start:start + 256]
        result[start:start + len(block)] = F.normalize(block.float() - mean, dim=-1).to(patches.dtype)
    return result


def centered_patch_gallery(gallery):
    valid = gallery["valid"]
    rgb_mean = gallery_patch_mean(gallery["primary"], valid)
    contour_mean = gallery_patch_mean(gallery["secondary"], valid)
    centered = {
        "primary": center_patch_blocks(gallery["primary"], rgb_mean),
        "secondary": center_patch_blocks(gallery["secondary"], contour_mean),
        "valid": valid,
    }
    return centered, contour_mean


@torch.inference_mode()
def cache_base_scores(queries, photos, path, metric_batch_size):
    shape = (len(queries), len(photos))
    size = shape[0] * shape[1]
    if not path.is_file() or path.stat().st_size != size * 4:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        scores = torch.from_file(str(temporary), shared=True, size=size, dtype=torch.float32).view(shape)
        start = 0
        for batch in tqdm(
            cosine_batches(queries, photos, metric_batch_size),
            total=(len(queries) + metric_batch_size - 1) // metric_batch_size,
            desc="Cache full-gallery cosine",
        ):
            end = start + len(batch)
            scores[start:end] = batch.cpu()
            start = end
        del scores
        temporary.replace(path)
    return torch.from_file(str(path), shared=False, size=size, dtype=torch.float32).view(shape)


@torch.inference_mode()
def cache_shortlist_scores(
    base_scores, sketch_patches, photo_patches, query_ids, photo_ids, maximum_k, methods, batch_size, device,
):
    ids, local = [], {method: [] for method in methods}
    # Keep paired gallery patches on the scoring device and stream query patches.
    gallery = {
        name: value[photo_ids].to(
            device=device, dtype=torch.float32 if device.type == "cpu" and value.is_floating_point() else value.dtype,
        )
        for name, value in photo_patches.items()
    }
    centered_names = {"patch_centered_ink", "patch_centered_consensus"} & set(methods)
    original_names = set(methods) - centered_names
    if centered_names:
        centered_gallery, contour_mean = centered_patch_gallery(gallery)
    for start in tqdm(range(0, len(base_scores), batch_size), desc="Score CLIP shortlist"):
        end = start + batch_size
        selected = base_scores[start:end].topk(maximum_k, dim=-1).indices.to(device)
        sketch = {
            name: value[query_ids[start:end]].to(
                device=device, dtype=gallery["primary"].dtype if value.is_floating_point() else value.dtype,
            )
            for name, value in sketch_patches.items()
        }
        ids.append(selected.cpu())
        if original_names:
            for name, score in score_shortlist(sketch, gallery, selected, original_names).items():
                local[name].append(score.cpu())
        if centered_names:
            centered_query = {
                "primary": center_patch_blocks(sketch["primary"], contour_mean),
                "valid": sketch["valid"],
            }
            names = {name.replace("patch_centered_", "patch_", 1) for name in centered_names}
            for name, score in score_shortlist(centered_query, centered_gallery, selected, names).items():
                local[name.replace("patch_", "patch_centered_", 1)].append(score.cpu())
    return torch.cat(ids), {name: torch.cat(scores) for name, scores in local.items()}


def reranked_rows(base_scores, ids, local, config):
    for start in range(0, len(base_scores), 128):
        end = start + 128
        rows = base_scores[start:end].clone()
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
