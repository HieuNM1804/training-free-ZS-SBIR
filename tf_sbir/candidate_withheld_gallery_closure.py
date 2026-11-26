"""Rank a photo by the first semantic prefix that supports a cycle without its own score."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import inspect
from itertools import combinations
import json
import os
from pathlib import Path
import random

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
from .gallery_convex_support import NEIGHBORS, SEED
from .gallery_convex_support_audit import matched_counts, newcomer_hits, prefix, summarize
from .joint_gallery_witnesses import load_reference
from .locked_geometry_words import score_rows
from .paired_gallery_connectivity import diffusion_bank, order_with_dot_ties, reciprocal_index
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import BATCH, SKETCHY, Split, mean_percent
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt
from .semantic_boxable_sparse import image_inputs
from .semantic_disputed_word_risk import word_bank


FAMILY = "candidate-withheld-gallery-closure"
ARMS = ("closure", "permuted_endpoints", "open_pair", "rgb_diffusion")


@dataclass(frozen=True)
class Paths:
    centers: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    middle: torch.Tensor
    permuted_left: torch.Tensor
    permuted_right: torch.Tensor


def endpoint_permutation(neighbors, rng):
    """Shuffle endpoint identities within the two halves of the RGB neighbor order."""
    first, second = np.array_split(neighbors, 2)
    return dict(zip(neighbors, np.concatenate((rng.permutation(first), rng.permutation(second))), strict=True))


def index_paths(index, linked):
    adjacency = [row[keep].tolist() for row, keep in zip(
        index.cpu().numpy(), linked.cpu().numpy(), strict=True,
    )]
    rng = np.random.default_rng(SEED)
    buckets = {}
    path_counts = np.zeros((len(adjacency), 2), dtype=np.int32)
    for center, neighbors in enumerate(adjacency):
        direct = set(neighbors)
        outside = {}
        paths = []
        for first in neighbors:
            for middle in adjacency[first]:
                if middle in direct and first < middle:
                    paths.append((first, middle, -1))
                elif middle != center and middle not in direct:
                    outside.setdefault(middle, []).append(first)
        triangles = len(paths)
        for middle, endpoints in outside.items():
            paths.extend((a, b, middle) for a, b in combinations(endpoints, 2))
        path_counts[center] = triangles, len(paths) - triangles
        if not paths:
            continue
        alternative = endpoint_permutation(neighbors, rng)
        paths = [(a, b, m, alternative[a], alternative[b]) for a, b, m in paths]
        width = triton.next_power_of_2(len(paths))
        buckets.setdefault(width, []).append((center, paths))

    result = []
    for width, rows in sorted(buckets.items()):
        packed = np.full((5, len(rows), width), -1, dtype=np.int32)
        centers = np.empty(len(rows), dtype=np.int32)
        for row, (center, paths) in enumerate(rows):
            centers[row] = center
            packed[:, row, :len(paths)] = np.asarray(paths, dtype=np.int32).T
        data = torch.as_tensor(packed, device=index.device)
        result.append(Paths(torch.as_tensor(centers, device=index.device), *data.unbind()))
    return result, adjacency, path_counts


@triton.jit
def path_levels(
    scores, centers, left, right, middle, permuted_left, permuted_right, observed, permuted,
    n: tl.constexpr, rows: tl.constexpr, width: tl.constexpr, tile: tl.constexpr,
):
    query = tl.program_id(1)
    row = tl.program_id(0) * tile + tl.arange(0, tile)
    slot = row[:, None] * width + tl.arange(0, width)[None, :]
    inside = row[:, None] < rows
    a = tl.load(left + slot, inside, -1)
    b = tl.load(right + slot, inside, -1)
    m = tl.load(middle + slot, inside, -1)
    a_null = tl.load(permuted_left + slot, inside, -1)
    b_null = tl.load(permuted_right + slot, inside, -1)
    value = scores + query * n
    av = tl.load(value + a, a >= 0, 0.0)
    bv = tl.load(value + b, b >= 0, 0.0)
    mv = tl.load(value + m, m >= 0, float("inf"))
    an = tl.load(value + a_null, a_null >= 0, 0.0)
    bn = tl.load(value + b_null, b_null >= 0, 0.0)
    real_level = tl.max(tl.minimum(tl.minimum(av, bv), mv), 1)
    null_level = tl.max(tl.minimum(tl.minimum(an, bn), mv), 1)
    output = query * n + tl.load(centers + row, row < rows, 0)
    tl.store(observed + output, real_level, row < rows)
    tl.store(permuted + output, null_level, row < rows)


def levels(scores, buckets):
    observed, permuted = torch.zeros_like(scores), torch.zeros_like(scores)
    for paths in buckets:
        count, width = paths.left.shape
        tile = max(1, 256 // width)
        path_levels[(triton.cdiv(count, tile), len(scores))](
            scores, paths.centers, paths.left, paths.right, paths.middle,
            paths.permuted_left, paths.permuted_right, observed, permuted,
            scores.size(1), count, width, tile,
        )
    return observed, permuted


@triton.jit
def open_pair_levels(scores, index, linked, out, n: tl.constexpr, k: tl.constexpr, block: tl.constexpr):
    query = tl.program_id(1)
    center = tl.program_id(0) * block + tl.arange(0, block)
    slot = tl.arange(0, k)
    entry = center[:, None] * k + slot[None, :]
    neighbor = tl.load(index + entry, center[:, None] < n, 0)
    valid = tl.load(linked + entry, center[:, None] < n, False)
    nearby = tl.load(scores + query * n + neighbor, valid, 0.0)
    # Remove one occurrence of the maximum, leaving a tied second maximum intact.
    first = tl.argmax(nearby, 1)
    second = tl.max(tl.where(slot[None, :] == first[:, None], 0.0, nearby), 1)
    tl.store(out + query * n + center, second, center < n)


def open_pair(scores, index, linked):
    output = torch.empty_like(scores)
    open_pair_levels[(triton.cdiv(scores.size(1), 32), len(scores))](
        scores, index, linked, output, scores.size(1), index.size(1), 32,
    )
    return output


def exists_at_level(adjacency, row, center, threshold):
    # An exhaustive leave-out check that does not use the packed paths.
    endpoints = {j for j in adjacency[center] if row[j] >= threshold}
    for first in endpoints:
        for inner in adjacency[first]:
            if inner == center or row[inner] < threshold:
                continue
            if inner in endpoints or any(last != first and last in endpoints for last in adjacency[inner]):
                return True
    return False


def check_levels(scores, observed, adjacency, points):
    scores, observed = scores.cpu().numpy(), observed.cpu().numpy()
    for row, center in points:
        value = observed[row, center]
        if (value > 0 and not exists_at_level(adjacency, scores[row], center, value)) or exists_at_level(
            adjacency, scores[row], center, np.nextafter(value, np.float32(np.inf)),
        ):
            raise ValueError("GPU result differs from a candidate-deleted path search")


def fixture(device):
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (5, 2), (3, 4)]
    index = torch.zeros((6, 4), dtype=torch.int32, device=device)
    linked = torch.zeros_like(index, dtype=torch.bool)
    for node in range(6):
        incident = [v if u == node else u for u, v in edges if node in (u, v)]
        index[node, :len(incident)] = torch.tensor(incident, device=device)
        linked[node, :len(incident)] = True
    paths, adjacency, _ = index_paths(index, linked)
    base = torch.tensor([
        [.07, .90, .80, .20, .10, .85],
        [.07, .90, .20, .80, .10, .85],
        [.99, .90, .80, .20, .10, .85],
        [.07, .90, .80, .20, .10, .15],
    ], dtype=torch.float32, device=device)
    real, _ = levels(base, paths)
    pair = open_pair(base, index, linked)
    check_levels(base, real, adjacency, ((q, i) for q in range(len(base)) for i in range(base.size(1))))
    torch.testing.assert_close(real[:, 0], base.new_tensor([.8, .2, .8, .15]))
    torch.testing.assert_close(pair[:, 0], base.new_full((len(base),), .8))
    return {"closure": real[:, 0].tolist(), "open_pair": pair[:, 0].tolist()}


def repair_diagnostic(base, views, gallery, truth, values):
    relevant = gallery[None, :] == truth[:, None]
    left, right = views
    best_positive = left.maximum(right).masked_fill(~relevant, -torch.inf).amax(1)
    best_negative = left.minimum(right).masked_fill(relevant, -torch.inf).amax(1)
    wrong = ~base[:, 3].bool()
    outside = wrong & (best_positive < best_negative).cpu()
    return torch.stack((wrong, outside, *(values[name][:, 3].bool() for name in ARMS)), 1)


def repair_summary(rows):
    wrong, outside, *top1 = rows.T
    return {
        "dot_wrong": int(wrong.sum()), "wrong_outside_two_view_interval": int(outside.sum()),
        "arms": {
            name: {
                "repaired": int((wrong & hit).sum()), "damaged": int((~wrong & ~hit).sum()),
                "repaired_outside_two_view_interval": int((outside & hit).sum()),
            } for name, hit in zip(ARMS, top1, strict=True)
        },
    }


def evaluate(root, split, text, kd, pool, signature):
    target = root / f"{split.name}.json"
    artifact = target.with_suffix(".pt")
    if target.is_file():
        report = json.loads(target.read_bytes())
        if report["protocol"] != signature or report["per_query_sha256"] != checksum(artifact):
            raise ValueError("Invalid previous report")
        print(split.name, "verified prior report", flush=True)
        return

    reference, _ = load_reference(split)
    p = word_bank(split.photo, text[DEFAULT_PROMPTS[1]], split.scale)
    index, linked, sims = reciprocal_index(F.normalize(split.photo.float(), dim=1))
    buckets, adjacency, counts = index_paths(index, linked)
    diffused = diffusion_bank(p, index, linked, sims)
    gallery, truth = split.photo_labels.cuda(), split.sketch_labels.cuda()
    replay = PrefixReplay(kd, split.photo_labels, split.name)
    map_k, precision_k = replay.cutoffs
    rows, comparisons = {arm: [] for arm in ARMS}, {arm: [] for arm in ARMS[1:]}
    repairs, times = [], dict(dot=0.0, closure_and_permutation=0.0)
    rng = np.random.default_rng(SEED)
    print(split.name, "scoring", len(truth), "queries with", int(counts.sum()), "gallery paths", flush=True)
    for start in range(0, len(truth), BATCH):
        sl = slice(start, min(start + BATCH, len(truth)))
        with torch.cuda.device(p.device):
            before = torch.cuda.Event(enable_timing=True)
            after_dot = torch.cuda.Event(enable_timing=True)
            after_paths = torch.cuda.Event(enable_timing=True)
            before.record()
            images = torch.stack((split.sketch[sl], split.mirror[sl].float()), dim=1)
            posterior = (split.scale * (images @ text[DEFAULT_PROMPTS[0]].T)).softmax(2)
            q = posterior.mean(1)
            base = q @ p
            after_dot.record()
            real, null = levels(base, buckets)
            after_paths.record()
        open_levels = open_pair(base, index, linked)
        if torch.any(real > open_levels) or torch.any(null > open_levels):
            raise ValueError("Path support must be bounded by the open-pair support")
        times["dot"] += before.elapsed_time(after_dot) / 1000
        times["closure_and_permutation"] += after_dot.elapsed_time(after_paths) / 1000
        if start == 0:
            check_levels(base[:8], real[:8], adjacency, zip(
                rng.integers(8, size=128), rng.integers(p.size(1), size=128), strict=True,
            ))
        scores = (*(order_with_dot_ties(s, base) for s in (real, null, open_levels)), q @ diffused)
        ordered = prefix(base, map_k, precision_k)
        in_base = torch.zeros_like(base, dtype=torch.bool).scatter_(1, ordered, True)
        base_rows = reference["dot"][sl]
        _, baseline_hits = newcomer_hits(ordered, in_base, gallery, truth[sl], base_rows[:, 1])
        if not torch.equal(baseline_hits[:, 0].cpu(), base_rows[:, 3].bool()):
            raise ValueError("Replayed Dot top-1 differs from the locked run")
        matched, batch_rows = {}, {}
        for arm, score in zip(ARMS, scores, strict=True):
            result = score_rows(score, split.sketch_labels[sl], gallery, split.name, replay, pool)
            batch_rows[arm] = result
            rows[arm].append(result)
            matched[arm] = newcomer_hits(
                prefix(score, map_k, precision_k), in_base, gallery, truth[sl], result[:, 1],
            )
        for arm in comparisons:
            comparisons[arm].append(matched_counts(matched["closure"], matched[arm]))
        view_scores = (view.contiguous() @ p for view in posterior.unbind(1))
        repairs.append(repair_diagnostic(base_rows, view_scores, gallery, truth[sl], batch_rows))
        if start == 0 or sl.stop == len(truth) or (start // BATCH + 1) % 10 == 0:
            print(split.name, sl.stop, "/", len(truth), "queries", flush=True)

    results = {arm: torch.cat(blocks).double() for arm, blocks in rows.items()}
    results["dot"] = reference["dot"]
    matched = {arm: torch.cat(blocks) for arm, blocks in comparisons.items()}
    repair = torch.cat(repairs)
    save_tensor({
        "qi": split.query_ids, "pi": split.gallery_ids, "labels": split.sketch_labels,
        "rows": results, "matched": matched, "top1_repair": repair,
    }, artifact)
    report = {
        "protocol": signature, "queries": len(truth), "gallery": p.size(1), "classes": len(truth.unique()),
        "cutoffs": list(replay.cutoffs),
        "metrics": {arm: mean_percent(result[:, :2]) for arm, result in results.items()},
        "closure_minus_macro_ap": {
            arm: class_interval(100 * (results["closure"][:, 0] - result[:, 0]), split.sketch_labels, seed=SEED)
            for arm, result in results.items() if arm != "closure"
        },
        "matched_promotions": {arm: summarize(value, split.sketch_labels) for arm, value in matched.items()},
        "top1_repair": repair_summary(repair),
        "paths": {
            "triangles": int(counts[:, 0].sum()), "four_cycles": int(counts[:, 1].sum()),
            "photos_without_either": int((counts.sum(1) == 0).sum()),
            "reciprocal_edges": int(linked.sum()) // 2,
        },
        "scoring_seconds": times, "per_query_sha256": checksum(artifact),
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
        raise ValueError("Unexpected 601-word vocabulary")
    kd, _ = load_kd(workspace)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("KD-SBIR source has changed")
    workers = min(4, len(os.sched_getaffinity(0)))
    cfg = runtime(workspace, "sketchy", batch=BATCH, workers=workers)
    signature = save_lock(root / "freeze.json", {
        "commit": revision(), "clip_sha256": CHECKPOINT, "evaluator_sha256": KD_FUNCTION_SHA,
        "vocabulary_sha256": vocab_sha, "prompts": list(DEFAULT_PROMPTS), "seed": SEED,
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "triton": triton.__version__,
        "neighbors": NEIGHBORS, "graph": "reciprocal RGB cosine neighborhoods",
        "path": "the shortest two nontrivial simple-path lengths after withholding the candidate",
        "views": ["original", "horizontal_flip"], "diffusion": {"alpha": .5, "steps": 16},
        "selection": "fixed before running any split",
        "primary_test": "class-macro matched newcomer precision versus permuted endpoints and open pair",
        "fixture": fixture(torch.device("cuda")),
    })
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = {prompt: encode_prompt(model, tokenizer, words, prompt) for prompt in DEFAULT_PROMPTS}
    data = image_inputs(cfg, SKETCHY, model)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name in SKETCHY:
            evaluate(root, Split.from_inputs(data, name), text, kd, pool, signature)
        del data
        cfg = runtime(workspace, "tuberlin", batch=BATCH, workers=workers)
        split = Split.from_inputs(image_inputs(cfg, ("tuberlin",), model), "tuberlin")
        evaluate(root, split, text, kd, pool, signature)


if __name__ == "__main__":
    run()
