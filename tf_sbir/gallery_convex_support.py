"""Certify gallery lexical evidence in a local tangent chart before query marginalization."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import random
import time
import zipfile

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT, runtime
from .depiction_path_diagnostic import class_interval
from .encoding import load_frozen_clip, save_tensor
from .gallery_diffusion import nearest_gallery, solve_diffusion
from .locked_geometry_words import score_rows
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import BATCH, SKETCHY, Split, mean_percent, posteriors
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt
from .semantic_boxable_sparse import image_inputs


FAMILY = "gallery-convex-support"
SEED = 42
NEIGHBORS = 16
ANGLE_TOLERANCE = 1e-6
DIFFUSION_ALPHA = 0.5
DIFFUSION_STEPS = 16
REFERENCE = DOCS / "semantic-boxable-sparse" / "paired-T4.zip"
ARMS = ("dot", "concept_hull", "shuffled_directions", "pooled_hull", "rgb_diffusion")


def tangent_coordinates(photos, neighbors):
    coordinates = []
    for start in range(0, len(photos), 512):
        anchor = photos[start:start + 512, None]
        nearby = photos[neighbors[start:start + 512]]
        tangent = nearby - (nearby * anchor).sum(-1, keepdim=True) * anchor
        eigval, eigvec = torch.linalg.eigh(tangent @ tangent.transpose(1, 2))
        coordinates.append(eigvec[..., -2:] * eigval[:, None, -2:].clamp_min(0).sqrt())
    return torch.cat(coordinates)


def direction_masks(coordinates):
    """A bit denotes an open halfplane containing a given neighbor."""
    unit = F.normalize(coordinates, dim=-1, eps=ANGLE_TOLERANCE)
    valid = coordinates.norm(dim=-1) > ANGLE_TOLERANCE
    x, y = unit.unbind(dim=-1)
    area = x[:, :, None] * y[:, None, :] - y[:, :, None] * x[:, None, :]
    no_axis = ~valid[:, :, None]
    positive = (area > ANGLE_TOLERANCE) | no_axis
    negative = (area < -ANGLE_TOLERANCE) | no_axis
    sides = torch.cat((positive, negative), dim=1) & valid[:, None, :]
    bits = 1 << torch.arange(sides.size(1), device=coordinates.device, dtype=torch.int64)
    return (sides * bits[None, :, None]).sum(1).to(torch.int32).contiguous()


def shuffle_directions(masks):
    # Slots arrive in descending cosine similarity. Shuffle within each block of four.
    rng = torch.Generator(device="cpu").manual_seed(SEED)
    strata = torch.rand((len(masks), NEIGHBORS // 4, 4), generator=rng).argsort(dim=-1, stable=True)
    order = (strata + 4 * torch.arange(NEIGHBORS // 4)[None, :, None]).flatten(1)
    return masks.gather(1, order.to(masks.device)).contiguous()


@triton.jit
def bit_union(left, right):
    return left | right


@triton.jit
def hull_kernel(scores, neighbors, masks, result, n: tl.constexpr, block: tl.constexpr, k: tl.constexpr):
    row = tl.program_id(1)
    center = tl.program_id(0) * block + tl.arange(0, block)
    offsets = center[:, None] * k + tl.arange(0, k)[None, :]
    valid = center[:, None] < n
    witness = tl.load(neighbors + offsets, valid, other=0)
    value = tl.load(scores + row * n + witness, valid, other=0)
    cover = tl.load(masks + offsets, valid, other=0).to(tl.uint32)
    # The high bits sort nonnegative float32 values exactly. Equal values may
    # order by mask because they all test the same threshold.
    packed = (value.to(tl.uint32, bitcast=True).to(tl.uint64) << 32) | cover.to(tl.uint64)
    ordered = tl.sort(packed, descending=True)
    value = (ordered >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    cover = tl.associative_scan(ordered.to(tl.uint32), 1, bit_union)
    floor = tl.max(tl.where(cover == 0xFFFFFFFF, value, 0.0), 1)
    tl.store(result + row * n + center, floor, center < n)


def support_floor(scores, neighbors, masks):
    scores = scores.contiguous()
    result = torch.empty_like(scores)
    block = 32
    hull_kernel[(triton.cdiv(scores.size(1), block), scores.size(0))](
        scores, neighbors, masks, result, scores.size(1), block, NEIGHBORS, num_warps=4,
    )
    return result


def verify_floor(scores, neighbors, coordinates, result):
    """Compare the GPU kernel to an independent angular-gap sweep on real rows."""
    rng = np.random.default_rng(SEED)
    probes = zip(rng.integers(len(scores), size=64), rng.integers(scores.size(1), size=64), strict=True)
    coords, idx, values, actual = (t.cpu().numpy() for t in (coordinates, neighbors, scores, result))
    for row, center in probes:
        nonzero = np.linalg.norm(coords[center], axis=1) > ANGLE_TOLERANCE
        angles = np.arctan2(coords[center, nonzero, 1], coords[center, nonzero, 0]).astype(np.float64)
        donors = values[row, idx[center, nonzero]]
        expected = 0.0
        for threshold in np.unique(donors)[::-1]:
            keep = np.sort(angles[donors >= threshold])
            gaps = np.diff(keep, append=keep[0] + 2 * np.pi)
            if np.max(gaps) < np.pi - ANGLE_TOLERANCE:
                expected = threshold
                break
        if actual[row, center] != expected:
            raise ValueError(f"Direction certificate disagrees with angular oracle at {row}/{center}")


def verify_equal_dot_example(device):
    """Same scalar evidence at every gallery vertex, different concept feasibility."""
    count = NEIGHBORS + 1
    slots = torch.arange(1, count, dtype=torch.int32, device=device).repeat(count, 1)
    degrees = torch.tensor([0, 60, 120, 180, 240, 300] + [0] * 10, device=device).deg2rad()
    positions = torch.stack((degrees.cos(), degrees.sin()), dim=-1).repeat(count, 1, 1)
    masks = direction_masks(positions)
    photo = torch.tensor([[.01, .01, .98]] + [[.89, .01, .10]] * NEIGHBORS, device=device).T.contiguous()
    split = photo.clone()
    # Two opposite sectors assign nearly all evidence to different words.
    split[:2, 3:6] = split[:2, 3:6].flip(0)
    query = torch.tensor([[.5, .5, 0.]], device=device)
    first, second = query @ photo, query @ split
    hi = query @ photo.maximum(support_floor(photo, slots, masks))
    lo = query @ split.maximum(support_floor(split, slots, masks))
    if not torch.equal(first, second) or not torch.allclose(hi[0, :1], torch.tensor([.45], device=device)):
        raise ValueError("Failed the equal-dot, shared-concept test")
    if not torch.allclose(lo[0, :1], torch.tensor([.01], device=device)):
        raise ValueError("A halfplane with only a different concept must not certify this concept")
    return {"center_dot": first[0, 0].item(), "one_concept": hi[0, 0].item(), "two_sectors": lo[0, 0].item()}


def diffuse_bank(photo_bank, neighbors, similarities):
    weights = similarities.pow(3)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    col, order = neighbors.sort(dim=1)
    offsets = torch.arange(0, col.numel() + 1, NEIGHBORS, device=neighbors.device)
    graph = torch.sparse_csr_tensor(offsets, col.flatten(), weights.gather(1, order).flatten(),
                                    size=(len(col), len(col)), device=neighbors.device)
    return solve_diffusion(graph, photo_bank, DIFFUSION_ALPHA, DIFFUSION_STEPS).contiguous()


def pooled_order(base, support):
    # Lexicographic order of two nonnegative float32 scores without a blending epsilon.
    primary = base.maximum(support).view(torch.int32).long()
    direct = base.view(torch.int32).long()
    return (primary << 32) | direct


def error_audit(base, pooled, concept, shuffled, gallery_labels, query_labels, precision_k):
    cut = base.topk(precision_k).values[:, -1:]
    not_in_base = base < cut
    correct = gallery_labels[None, :] == query_labels[:, None]
    masks = []
    for first, second in ((concept, shuffled), (concept, pooled)):
        first_only = not_in_base & (first >= cut) & (second < cut)
        second_only = not_in_base & (second >= cut) & (first < cut)
        masks.extend((first_only, first_only & correct, second_only, second_only & correct))
    return torch.stack([mask.sum(dim=1) for mask in masks], dim=1).cpu()


def audit_summary(audit):
    result = {}
    for position, comparison in enumerate(("concept_vs_shuffled", "concept_vs_pooled")):
        left_n, left_hit, right_n, right_hit = audit.sum(0)[4 * position:4 * position + 4].tolist()
        result[comparison] = {
            "concept_only": [left_hit, left_n], "control_only": [right_hit, right_n],
            "concept_only_relevance": left_hit / left_n if left_n else None,
            "control_only_relevance": right_hit / right_n if right_n else None,
        }
    return result


def verify_baseline(split, value):
    with zipfile.ZipFile(REFERENCE) as archive:
        saved = torch.load(io.BytesIO(archive.read(f"{split.name}.pt")), weights_only=True)
    ordered = (("qi", split.query_ids), ("pi", split.gallery_ids), ("labels", split.sketch_labels))
    if any(not torch.equal(saved[key], data) for key, data in ordered):
        raise ValueError(f"Different query/gallery order from locked baseline on {split.name}")
    difference = 100 * (value.double().mean(0) - saved["rows"]["dot_601"].double().mean(0))
    if difference.abs().max() > 0.01:
        raise ValueError(f"Dot deviates from locked baseline on {split.name}: {difference.tolist()}")
    return difference.tolist()


def evaluate(root, split, text, kd, pool, signature):
    target = root / f"{split.name}.json"
    artifact = target.with_suffix(".pt")
    if target.is_file():
        report = json.loads(target.read_bytes())
        if report["protocol"] != signature or report["per_query_sha256"] != checksum(artifact):
            raise ValueError(f"Invalid prior report for {split.name}")
        print(split.name, "verified prior report", flush=True)
        return

    print(split.name, "indexing the complete gallery", flush=True)
    query, gallery = posteriors(split, text, [DEFAULT_PROMPTS])
    q, p = query[DEFAULT_PROMPTS[0]], gallery[DEFAULT_PROMPTS[1]]
    started = time.perf_counter()
    photos = F.normalize(split.photo.float(), dim=1)
    index, similarity = nearest_gallery(photos, NEIGHBORS)
    index = index.to(torch.int32).contiguous()
    xy = tangent_coordinates(photos, index)
    directions = direction_masks(xy)
    concept_floor = support_floor(p, index, directions)
    verify_floor(p, index, xy, concept_floor)
    concept_bank = p.maximum(concept_floor)
    shuffled_bank = p.maximum(support_floor(p, index, shuffle_directions(directions)))
    diffused = diffuse_bank(p, index.long(), similarity)
    torch.cuda.synchronize()
    index_seconds = time.perf_counter() - started
    print(split.name, f"indexed in {index_seconds:.2f} s", flush=True)

    labels, truth = split.photo_labels.cuda(), split.sketch_labels.cuda()
    replay = PrefixReplay(kd, split.photo_labels, split.name)
    _, precision_k = replay.cutoffs
    groups = {name: [] for name in ARMS}
    audits = []
    score_seconds = 0.0
    for start in range(0, len(q), BATCH):
        stop = min(start + BATCH, len(q))
        batch = q[start:stop]
        started = time.perf_counter()
        base = batch @ p
        concept = batch @ concept_bank
        shuffled = batch @ shuffled_bank
        pooled = support_floor(base, index, directions)
        scores = (base, concept, shuffled, pooled_order(base, pooled), batch @ diffused)
        torch.cuda.synchronize()
        score_seconds += time.perf_counter() - started
        audits.append(error_audit(base, pooled.maximum(base), concept, shuffled, labels, truth[start:stop], precision_k))
        for name, value in zip(ARMS, scores, strict=True):
            groups[name].append(score_rows(value, split.sketch_labels[start:stop], labels, split.name, replay, pool))
        if start == 0 or stop == len(q) or (start // BATCH + 1) % 10 == 0:
            print(split.name, "original KD-SBIR", stop, "/", len(q), flush=True)

    rows = {name: torch.cat(parts).double() for name, parts in groups.items()}
    if any(value.shape != (len(q), 4) or not value.isfinite().all() for value in rows.values()):
        raise ValueError(f"Incomplete/nonfinite replay on {split.name}")
    baseline_error = verify_baseline(split, rows["dot"][:, :2])
    audit = torch.cat(audits)
    save_tensor({"qi": split.query_ids, "pi": split.gallery_ids, "labels": split.sketch_labels,
                 "rows": rows, "threshold_disagreement": audit}, artifact)
    top1 = rows["dot"][:, 3].bool()
    report = {
        "protocol": signature, "queries": len(q), "gallery": p.size(1), "classes": len(truth.unique()),
        "cutoffs": list(replay.cutoffs), "metrics": {name: mean_percent(value[:, :2]) for name, value in rows.items()},
        "concept_hull_minus_control_macro_ap": {
            name: class_interval(100 * (rows["concept_hull"][:, 0] - value[:, 0]), split.sketch_labels, seed=SEED)
            for name, value in rows.items() if name != "concept_hull"
        },
        "top1_repairs_and_damage": {
            name: [int((~top1 & value[:, 3].bool()).sum()), int((top1 & ~value[:, 3].bool()).sum())]
            for name, value in rows.items()
        },
        "threshold_disagreement": audit_summary(audit), "baseline_drift_points": baseline_error,
        "index_seconds_including_oracle": index_seconds, "batch_scoring_seconds_all_arms": score_seconds,
        "per_query_sha256": checksum(artifact),
    }
    atomic_json(target, report)
    print(split.name, "verified", report["metrics"], flush=True)


@torch.inference_mode()
def run():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    workspace = Path("/content/sbir")
    root = workspace / "results" / FAMILY
    config = json.loads((DOCS / VOCABULARY_STUDY / "freeze.json").read_bytes())
    vocab_sha = config.pop("sha256")
    words = config["subsets"]["boxable"]["words"]
    if len(words) != 601 or digest(config) != vocab_sha:
        raise ValueError("Expected the locked 601 boxable concepts")
    kd, _ = load_kd(workspace)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("KD-SBIR source has changed")
    toy = verify_equal_dot_example(torch.device("cuda"))
    spec = {
        "commit": revision(), "clip_sha256": CHECKPOINT, "evaluator_sha256": KD_FUNCTION_SHA,
        "vocabulary_sha256": vocab_sha, "prompts": list(DEFAULT_PROMPTS), "seed": SEED,
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "triton": triton.__version__,
        "neighbors": NEIGHBORS, "tangent_dimensions": 2, "angle_tolerance": ANGLE_TOLERANCE,
        "shuffle_block": 4, "arms": list(ARMS), "diffusion_alpha": DIFFUSION_ALPHA,
        "diffusion_steps": DIFFUSION_STEPS, "selection": "fixed before reading any split metrics",
        "equal_dot_fixture": toy,
    }
    signature = save_lock(root / "freeze.json", spec)
    workers = min(4, len(os.sched_getaffinity(0)))
    cfg = runtime(workspace, "sketchy", batch=384, workers=workers)
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = {prompt: encode_prompt(model, tokenizer, words, prompt) for prompt in DEFAULT_PROMPTS}
    data = image_inputs(cfg, SKETCHY, model)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name in SKETCHY:
            evaluate(root, Split.from_inputs(data, name), text, kd, pool, signature)
        del data
        cfg = runtime(workspace, "tuberlin", batch=384, workers=workers)
        split = Split.from_inputs(image_inputs(cfg, ("tuberlin",), model), "tuberlin")
        evaluate(root, split, text, kd, pool, signature)


if __name__ == "__main__":
    run()
