"""Audit the production quadrature on unseen gallery rankings and small score margins."""

import json
import os
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F

from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT, runtime
from .encoding import load_frozen_clip
from .forced_pair_alignment import AnchorGeometry, QUADRATURE_POINTS, profile_terms
from .forced_pair_study import FAMILY
from .gallery_convex_support import SEED
from .gallery_convex_support_audit import prefix
from .quickdraw_locked import KD_FUNCTION_SHA
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import BATCH, SKETCHY, Split, posteriors
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt
from .semantic_boxable_sparse import image_inputs


def barycenter_audit(posteriors, anchors):
    # Subtracting the row mean makes exact uniformity map to zero even with rounded anchors.
    # Compute the subtraction using the first entry so a constant row cancels exactly.
    anchors = anchors.double()
    anchors = anchors - anchors.mean(0)
    scale = anchors.square().sum(1).mean().sqrt()
    deviations, radii, uniform = [], [], []
    for rows in posteriors.split(1024):
        rows = rows.double()
        uniform.append(rows.amax(1) == rows.amin(1))
        normalized = rows / rows.sum(1, keepdim=True)
        deviations.append(.5 * (normalized - 1 / rows.size(1)).abs().sum(1))
        centered = normalized - normalized[:, :1]
        centered -= centered.mean(1, keepdim=True)
        radii.append((centered @ anchors).norm(dim=1) / scale)
    radius, tv = torch.cat(radii), torch.cat(deviations)
    return {
        "count": len(radius), "exactly_uniform_posteriors": int(torch.cat(uniform).sum()),
        "exactly_zero_centered_barycenters": int((radius == 0).sum()),
        "barycenter_radius_divided_by_rms_anchor_norm": {
            "min": float(radius.min()), "p01": float(radius.quantile(.01)), "median": float(radius.median()),
            "counts_at_or_below": {str(t): int((radius <= t).sum()) for t in (1e-6, 1e-3, .01)},
        },
        "total_variation_from_uniform": {
            "min": float(tv.min()), "counts_at_or_below": {str(t): int((tv <= t).sum()) for t in (1e-6, 1e-3, .01)},
        },
    }


def exact_profile(x, y, spectrum):
    left = spectrum.diag() - x[:, None] * (x * spectrum)[None, :]
    projected = left - (left @ y)[:, None] * y[None, :]
    # Use the double-precision QR SVD independently of the quadrature formula.
    return (x * spectrum) @ y + torch.linalg.svdvals(projected, driver="gesvd").sum() - spectrum.sum()


def sign(value):
    return int(value > 0) - int(value < 0)


def audit_nearest_margins(approximate, refined, x, y, spectrum, positions, split):
    records = []
    # Find two distinct photos with the smallest positive production margin in the
    # first 32 ranks of each of four preselected queries. No relevance labels enter.
    for row in range(4):
        values, rank = approximate[row].topk(32)
        gaps = values[:-1] - values[1:]
        if (gaps > 0).any():
            slot = int(gaps.masked_fill(gaps <= 0, torch.inf).argmin())
        else:
            slot = 0
        cols = rank[slot:slot + 2]
        exact = torch.stack([exact_profile(x[row], y[col], spectrum) for col in cols])
        low = approximate[row, cols].double()
        high = refined[row, cols]
        exact_gap = float(exact[0] - exact[1])
        records.append({
            "query_id": int(split.query_ids[positions[row]]), "ranks_at_32_nodes": [slot + 1, slot + 2],
            "gallery_ids": split.gallery_ids[cols.cpu()].tolist(), "direct_svd_gap": exact_gap,
            "production_gap": float(low[0] - low[1]), "refined_gap": float(high[0] - high[1]),
            "production_agrees_with_svd_order": sign(float(low[0] - low[1])) == sign(exact_gap),
            "refined_agrees_with_svd_order": sign(float(high[0] - high[1])) == sign(exact_gap),
            "max_abs_error_vs_svd_32": float((low - exact).abs().max()),
            "max_abs_error_vs_svd_64": float((high - exact).abs().max()),
        })
    return records


def audit(root, split, text, geometry, lock):
    source = root / (split.name + ".pt")
    report = json.loads(source.with_suffix(".json").read_bytes())
    if report["protocol"] != lock or checksum(source) != report["per_query_sha256"]:
        raise ValueError("Primary results failed checksum or protocol verification")
    primary = torch.load(source, map_location="cpu", weights_only=True)
    for name, expected in (("qi", split.query_ids), ("pi", split.gallery_ids), ("labels", split.sketch_labels)):
        if not torch.equal(primary[name], expected):
            raise ValueError("Primary IDs or labels changed")
    queries, photos = posteriors(split, text, [DEFAULT_PROMPTS])
    q, p = queries[DEFAULT_PROMPTS[0]], photos[DEFAULT_PROMPTS[1]].T.contiguous()
    uniformity = {"queries": barycenter_audit(q, geometry.source), "gallery": barycenter_audit(p, geometry.target)}
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    positions = torch.randperm(len(q), generator=generator)[:64]
    x, y = geometry.project(q, p)
    x = x[positions].contiguous()
    low = profile_terms(x, y, geometry.spectrum, QUADRATURE_POINTS)[2]
    x64, y64 = F.normalize(x.double()), F.normalize(y.double())
    s64 = geometry.spectrum.double()
    high = profile_terms(x64, y64, s64, 2 * QUADRATURE_POINTS)[2]
    if not torch.isfinite(low).all() or not torch.isfinite(high).all():
        raise ValueError("Encountered nonfinite profile")
    map_k, k = report["cutoffs"]
    low_ids, high_ids = prefix(low, map_k, k), prefix(high, map_k, k)
    members = torch.zeros_like(low, dtype=torch.bool).scatter_(1, low_ids, True)
    common = members.gather(1, high_ids).sum(1)
    gallery_truth = split.photo_labels.cuda()
    sampled_truth = split.sketch_labels[positions].cuda()[:, None]
    hits = gallery_truth[low_ids] == sampled_truth
    baseline = primary["rows"]["profile"][positions]
    abs_difference = (low.double() - high).abs()
    return {
        "queries": len(x), "gallery": len(y), "primary_protocol": lock,
        "sample_query_ids": split.query_ids[positions].tolist(), "precision_cutoff": k,
        "full_matrix_max_abs_error_32_vs_64": float(abs_difference.max()),
        "full_matrix_mean_abs_error_32_vs_64": float(abs_difference.mean()),
        "top1_agreement": int((low_ids[:, 0] == high_ids[:, 0]).sum()),
        "membership_overlap_mean_fraction": float(common.double().mean() / k),
        "membership_overlap_min_fraction": float(common.min()) / k,
        "identical_membership_count": int((common == k).sum()),
        "production_replay_top1_hit_agreement_count": int((hits[:, 0].cpu() == baseline[:, 3].bool()).sum()),
        "production_replay_max_precision_change": float((hits.double().mean(1).cpu() - baseline[:, 1]).abs().max()),
        "svd_close_pairs": audit_nearest_margins(low, high, x64, y64, s64, positions, split),
        "barycenter_uniformity": uniformity,
    }


@torch.inference_mode()
def run():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    workspace = Path("/content/sbir")
    root = workspace / "results" / FAMILY
    freeze = json.loads((root / "freeze.json").read_bytes())
    vocab = json.loads((DOCS / VOCABULARY_STUDY / "freeze.json").read_bytes())
    vocab_sha = vocab.pop("sha256")
    words = vocab["subsets"]["boxable"]["words"]
    config = freeze["config"]
    if (freeze["sha256"] != digest(config) or len(words) != 601 or digest(vocab) != vocab_sha
            or config["vocabulary_sha256"] != vocab_sha or tuple(config["prompts"]) != DEFAULT_PROMPTS
            or config["clip_sha256"] != CHECKPOINT or config["seed"] != SEED
            or config["evaluator_sha256"] != KD_FUNCTION_SHA):
        raise ValueError("Primary lock does not match evaluation configuration")
    workers = min(4, len(os.sched_getaffinity(0)))
    cfg = runtime(workspace, "sketchy", batch=BATCH, workers=workers)
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = {prompt: encode_prompt(model, tokenizer, words, prompt) for prompt in DEFAULT_PROMPTS}
    geometry = AnchorGeometry(*(text[prompt] for prompt in DEFAULT_PROMPTS))
    results = {}
    for dataset, names in (("sketchy", SKETCHY), ("tuberlin", ("tuberlin",))):
        cfg = runtime(workspace, dataset, batch=BATCH, workers=workers)
        data = image_inputs(cfg, names, model)
        for name in names:
            results[name] = audit(root, Split.from_inputs(data, name), text, geometry, freeze["sha256"])
            print(name, results[name]["top1_agreement"], results[name]["membership_overlap_mean_fraction"], flush=True)
        del data
    atomic_json(root / "numerical-audit.json", {
        "protocol": freeze["sha256"], "primary_commit": config["commit"], "audit_commit": revision(),
        "seed": SEED, "nodes": [QUADRATURE_POINTS, 2 * QUADRATURE_POINTS], "gpu": torch.cuda.get_device_name(),
        "comparison": "production fp32 versus unit-renormalized fp64 coordinates in identical singular basis",
        "results": results,
    })


if __name__ == "__main__":
    run()
