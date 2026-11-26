"""Predeclared numerical checks, full unseen scoring and frozen transfer for depiction flow."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .conditional_completion_study import evaluate
from .crossdomain_inputs import CHECKPOINT
from .depiction_path_diagnostic import class_interval
from .depiction_state_flow import SOURCE, TARGET, DepictionFlow, center, unit
from .encoding import load_frozen_clip, save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE, load_visual_split
from .profiled_study import true_sources
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .replicated_simplex_study import evaluator, read
from .semantic_boxable_prompt_search import SKETCHY, mean_percent


DEPTHS = (1, 3, 6)
SUBSTEPS = (2, 4, 8, 16)
SEED = 42
PARTS = 2
FAMILY = "depiction-state-flow"
BATCH = 384
ROOT = DEFAULT_CACHE.parent / "results" / FAMILY


def model_for(cache):
    manifest = read(cache / "manifest.json")
    if manifest["sha256"] != CHECKPOINT:
        raise ValueError("Checkpoint hash mismatch")
    return DepictionFlow(load_frozen_clip(manifest["checkpoint"], torch.device("cuda")).float())


def protocol(args):
    return save_lock(args.output / "protocol.json", {
        "commit": args.revision, "seed": SEED, "checkpoint": CHECKPOINT, "evaluator": KD_FUNCTION_SHA,
        "source": SOURCE, "target": TARGET, "depths": DEPTHS, "substep_candidates": SUBSTEPS,
        "solver": "spherical RK4, unit time per native sublayer, fixed native terminal-centered radius",
        "numerics": "24 sketches and 24 gallery photos from Ext-1 without labels, seed 42. Each depth must pass. "
                    "Choose the first substep count with target forward/reverse cycle median cosine >= .999, "
                    "p01 >= .995 and p01 cosine against twice the resolution >= .995.",
        "selection": "Equal-weight full unseen Sketchy mAP, break ties by mean precision and then shallower depth. "
                     "Evaluate TU only after the freeze. Two split Sketchy scores are exploratory.",
        "success": "At least +4 absolute mAP on each Sketchy split versus the paired Semantic Dot baseline, "
                   "positive frozen transfer, and evidence for content-specific improvement over a shared shift.",
        "controls": ["raw two-view visual cosine", "projected cosine", "projected fixed generic prompt displacement",
                     "archived paired L4 Semantic Dot 601 reference"],
        "independent_query": True, "primary_external_dictionary": False, "cache_manifest": checksum(args.cache / "manifest.json"),
    })


def numeric(args):
    root = args.output
    split = load_visual_split(args.cache, "sketchy_1")
    flow = model_for(args.cache)
    rng = torch.Generator().manual_seed(SEED)
    qi = torch.randperm(len(split.sketch), generator=rng)[:24]
    pi = torch.randperm(len(split.photo), generator=rng)[:24]
    sample = unit(torch.cat([split.sketch[qi], split.photo[pi]]).float())
    projected = flow.projected(sample)
    recovered = flow.chart.encode(flow.chart.decode(sample)).float()
    origin = (projected - recovered).abs().amax().item()
    caches = {}
    for tag, ctx in (("source", flow.source), ("target", flow.target)):
        native_term = flow.chart.encode(unit(center(ctx.terminal))).float()
        caches[tag] = {"attention_max_abs": ctx.attention_error, "native_block_max_abs": ctx.residual_error,
                       "perturbed_block_max_abs": ctx.perturbation_error,
                       "projected_native_cos": F.cosine_similarity(native_term, flow.projected(ctx.native)).item()}
    if origin > 2e-4 or any(max(v["attention_max_abs"], v["native_block_max_abs"], v["perturbed_block_max_abs"]) > .002
                             or v["projected_native_cos"] < .9999 for v in caches.values()):
        raise ValueError(f"Conditional evaluation does not reconstruct native CLIP: {origin}, {caches}")
    evidence = {}
    selected = None
    previous = None
    for steps in SUBSTEPS:
        approx = previous if previous is not None else flow.transform(sample, args.depth, steps)
        refined = flow.transform(sample, args.depth, 2 * steps)
        back = flow.transform(approx, args.depth, steps, reverse=True)
        cycle = F.cosine_similarity(projected, back)
        resolution = F.cosine_similarity(approx, refined)
        stats = {"cycle_median": cycle.median().item(), "cycle_p01": cycle.quantile(.01).item(),
                 "double_resolution_p01": resolution.quantile(.01).item()}
        evidence[steps] = stats
        if stats["cycle_median"] >= .999 and min(stats["cycle_p01"], stats["double_resolution_p01"]) >= .995:
            selected = steps
            break
        previous = refined
    result = {"protocol": read(root/"protocol.json")["sha256"], "depth": args.depth, "steps": selected,
              "query_sample": qi.tolist(), "gallery_sample": pi.tolist(), "chart_max_abs": origin,
              "context_validation": caches, "convergence": evidence}
    atomic_json(root/f"numerics-{args.depth}.json", result)
    print("numeric", args.depth, selected, evidence, flush=True)


def query_subset(split, part):
    size = len(split.query_ids)
    sl = slice(size * part // PARTS, size * (part + 1) // PARTS)
    return replace(split, query_ids=split.query_ids[sl], sketch_labels=split.sketch_labels[sl],
                   sketch=split.sketch[sl], mirror=split.mirror[sl])


def embed(split, flow, depth, steps):
    results = []
    for start in range(0, len(split.sketch), BATCH):
        sl = slice(start, start + BATCH)
        original = split.sketch[sl].float()
        mirror = split.mirror[sl].float()
        both = flow.transform(torch.cat([original, mirror]), depth, steps)
        first, second = both.chunk(2, dim=0)
        results.append(unit(first + second))
    return torch.cat(results)


def numeric_steps(root, depth):
    record = read(root/f"numerics-{depth}.json")
    if record["protocol"] != read(root/"protocol.json")["sha256"] or record["steps"] is None:
        raise ValueError(f"Depth {depth} did not meet the predeclared numerical gate")
    return record["steps"]


def eligibility(args):
    """Record the descriptive replay decision before seeing any retrieval metrics."""
    root = args.output
    if list(root.glob("grid-*.json")) or (root/"freeze.json").exists():
        raise ValueError("Numerical eligibility must be frozen before retrieval")
    records = {d: read(root/f"numerics-{d}.json") for d in DEPTHS}
    signature = read(root/"protocol.json")["sha256"]
    if any(r["protocol"] != signature for r in records.values()):
        raise ValueError("Numerical validation belongs to a different protocol")
    eligible = [d for d, record in records.items() if record["steps"] is not None]
    if not eligible:
        raise ValueError("No numerically admissible map to score")
    save_lock(root/"numerical-eligibility.json", {
        "commit": args.revision, "original_protocol": signature, "eligible": eligible,
        "all_depths_passed": len(eligible) == len(DEPTHS),
        "numerics": {str(d): checksum(root/f"numerics-{d}.json") for d in DEPTHS},
        "purpose": "Descriptive full-unseen replay of previously specified, numerically valid depths only. "
                   "The original all-depth convergence hypothesis already failed. No retrieval metrics seen.",
    })


def scan(args):
    if args.split not in SKETCHY:
        raise ValueError("Development is restricted to the two Sketchy splits")
    root = args.output
    full = load_visual_split(args.cache, args.split)
    split = query_subset(full, args.part)
    flow = model_for(args.cache)
    gallery = flow.projected(split.photo)
    replay = PrefixReplay(evaluator(args.evaluator), split.photo_labels, split.name)
    embeddings, means, seconds = {}, {}, {}
    for depth in read(root/"numerical-eligibility.json")["config"]["eligible"]:
        since = time.perf_counter()
        vectors = embed(split, flow, depth, numeric_steps(root, depth))
        embeddings[str(depth)] = vectors.cpu()
        means[str(depth)] = mean_percent(evaluate(split, vectors, gallery, replay))
        seconds[str(depth)] = time.perf_counter() - since
        print(args.split, args.part, depth, means[str(depth)], round(seconds[str(depth)], 1), flush=True)
    artifact = root/"features"/f"{args.split}-{args.part}.pt"
    save_tensor({"qi": split.query_ids, "pi": split.gallery_ids, "embeddings": embeddings}, artifact)
    atomic_json(root/f"grid-{args.split}-{args.part}.json", {
        "protocol": read(root/"protocol.json")["sha256"], "queries": len(split.sketch), "metrics": means,
        "device": torch.cuda.get_device_name(), "seconds_inc_score": seconds, "features_sha256": checksum(artifact),
    })


def lock(args):
    root = args.output
    signature = read(root/"protocol.json")["sha256"]
    decision = read(root/"numerical-eligibility.json")
    eligible = decision["config"]["eligible"]
    grids, means = {}, {}
    for split in SKETCHY:
        names = [f"grid-{split}-{part}.json" for part in range(PARTS)]
        records = [read(root/name) for name in names]
        if any(rec["protocol"] != signature for rec in records):
            raise ValueError("Grid protocol mismatch")
        grids.update({name: checksum(root/name) for name in names})
        weights = np.array([rec["queries"] for rec in records])
        means[split] = {str(d): np.average([rec["metrics"][str(d)] for rec in records], axis=0,
                                           weights=weights).tolist() for d in eligible}
    def criterion(d):
        result = np.mean([means[s][str(d)] for s in SKETCHY], axis=0)
        return result[0], result[1], -d
    depth = max(eligible, key=criterion)
    save_lock(root/"freeze.json", {"commit": args.revision, "protocol": signature, "grids": grids,
              "depth": depth, "steps": numeric_steps(root, depth), "selection_scores": means,
              "eligibility": decision["sha256"], "numerics": {str(d): checksum(root/f"numerics-{d}.json") for d in DEPTHS}})
    print("Frozen depth", depth, means, flush=True)


def paired_baseline(split, reference):
    report = read(reference/f"{split.name}.json")
    artifact = reference/f"{split.name}.pt"
    if report["per_query_sha256"] != checksum(artifact):
        raise ValueError("Semantic reference checkpoint changed")
    saved = torch.load(artifact, weights_only=True)
    if any(not torch.equal(saved[k], v) for k, v in
           (("qi", split.query_ids), ("pi", split.gallery_ids), ("labels", split.sketch_labels))):
        raise ValueError("Baseline has a different query or gallery order")
    return saved["rows"]["dot"][:, :2], report["per_query_sha256"]


def source_audit(split, cache, original, transformed, fixed, photos):
    source = true_sources(split, cache)
    valid = source >= 0
    source_ids, group = source[valid].unique(return_inverse=True)
    counts = group.bincount().float()[:, None]
    def per_identity(x):
        out = torch.zeros(len(counts), x.size(1), device=x.device)
        return out.index_add_(0, group, x[valid]) / counts
    known = photos[source_ids]
    images = per_identity(original)
    true_delta = known - images
    classes = split.photo_labels.cuda()[source_ids]
    rng = torch.Generator().manual_seed(SEED)
    halves = []
    for c in classes.unique():
        idx = (classes == c).nonzero().flatten()
        halves.append(idx[torch.randperm(len(idx), generator=rng)].tensor_split(2))
    result = {}
    for name, vector in (("flow", transformed), ("fixed_shift", fixed)):
        prediction = per_identity(vector) - images
        cos = []
        for half in range(2):
            predicted = torch.stack([prediction[h[half]].mean(0) for h in halves])
            observed = torch.stack([true_delta[h[1 - half]].mean(0) for h in halves])
            cos.append(F.cosine_similarity(predicted - predicted.mean(0), observed - observed.mean(0)).mean().item())
        result[name] = {"centered_class_shift_cross_half_cos": sum(cos) / 2,
                        "mean_source_cos_change": ((per_identity(vector) - images) * known).sum(1).mean().item()}
    result["paired_sketches"] = valid.sum().item()
    result["distinct_photos"] = len(counts)
    return result


def mean_other_cos(vectors):
    vectors = unit(vectors.double())
    n = len(vectors)
    return ((vectors.sum(0).square().sum() - n) / (n * (n-1))).item()


def official(args):
    root = args.output
    freeze = read(root/"freeze.json")
    frozen = freeze["config"]
    if frozen["commit"] != args.revision or frozen["protocol"] != read(root/"protocol.json")["sha256"]:
        raise ValueError("Frozen specification mismatch")
    split = load_visual_split(args.cache, args.split)
    flow = model_for(args.cache)
    photos = flow.projected(split.photo)
    original = unit(unit(split.sketch.float()) + unit(split.mirror.float()))
    projected = unit(flow.projected(split.sketch) + flow.projected(split.mirror))
    fixed = unit(flow.fixed_shift(split.sketch) + flow.fixed_shift(split.mirror))
    if args.split in SKETCHY:
        shards = []
        for part in range(PARTS):
            artifact = root/"features"/f"{args.split}-{part}.pt"
            filename = f"grid-{args.split}-{part}.json"
            if checksum(root/filename) != frozen["grids"][filename] or checksum(artifact) != read(root/filename)["features_sha256"]:
                raise ValueError("Unverified development features")
            saved = torch.load(artifact, weights_only=True)
            if not torch.equal(saved["qi"], query_subset(split, part).query_ids) or not torch.equal(saved["pi"], split.gallery_ids):
                raise ValueError("Unverified development order")
            shards.append(saved["embeddings"][str(frozen["depth"])])
        transformed = torch.cat(shards).cuda()
    else:
        transformed = embed(split, flow, frozen["depth"], frozen["steps"])
    replay = PrefixReplay(evaluator(args.evaluator), split.photo_labels, split.name)
    baseline, ref_sha = paired_baseline(split, args.reference)
    rows = {"semantic_dot": baseline}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for arm, query, bank in (("flow", transformed, photos), ("native", original, unit(split.photo.float())),
                                 ("projected", projected, photos), ("fixed_shift", fixed, photos)):
            rows[arm] = evaluate(split, query, bank, replay, pool)
            print(args.split, arm, mean_percent(rows[arm]), flush=True)
    means = {k: mean_percent(v) for k, v in rows.items()}
    if args.split in SKETCHY and max(abs(a-b) for a,b in zip(means["flow"], frozen["selection_scores"][args.split][str(frozen["depth"])], strict=True)) > .015:
        raise ValueError("Official replay differs from the development ranking")
    audit = source_audit(split, args.cache, projected, transformed, fixed, photos) if args.split in SKETCHY else None
    artifact = root/f"{args.split}.pt"
    save_tensor({"qi": split.query_ids, "pi": split.gallery_ids, "labels": split.sketch_labels, "rows": rows}, artifact)
    atomic_json(root/f"{args.split}.json", {
        "commit": args.revision, "selection": freeze["sha256"], "queries": len(original), "gallery": len(photos),
        "classes": len(split.sketch_labels.unique()), "device": torch.cuda.get_device_name(), "cutoffs": replay.cutoffs,
        "metrics": means, "source_identity_audit": audit, "per_query_sha256": checksum(artifact), "semantic_reference_sha256": ref_sha,
        "average_cos_between_distinct_queries_after_ranking": {
            "projected": mean_other_cos(projected), "flow": mean_other_cos(transformed), "fixed_shift": mean_other_cos(fixed)},
        "flow_minus_control_class_macro_ap": {k: class_interval(100*(rows["flow"][:,0]-v[:,0]), split.sketch_labels, seed=SEED)
                                             for k,v in rows.items() if k != "flow"},
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("protocol", "numeric", "eligibility", "scan", "lock", "official"))
    p.add_argument("--revision", required=True)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--output", type=Path, default=ROOT)
    p.add_argument("--evaluator", type=Path, default=DEFAULT_CACHE.parent/"real-sketchy-smoke"/"evaluator.py")
    p.add_argument("--reference", type=Path, default=ROOT.parent/"exchangeable-concept-observations")
    p.add_argument("--split", choices=(*SKETCHY, "tuberlin"))
    p.add_argument("--depth", type=int, choices=DEPTHS)
    p.add_argument("--part", type=int, choices=range(PARTS), default=0)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    with torch.inference_mode():
        globals()[args.stage](args)
