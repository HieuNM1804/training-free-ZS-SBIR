"""Recover gallery components that jointly support two readings of one query."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import time

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
from .gallery_diffusion import nearest_gallery, solve_diffusion
from .joint_gallery_witnesses import load_reference
from .locked_geometry_words import score_rows
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import BATCH, SKETCHY, Split, mean_percent
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt
from .semantic_boxable_sparse import image_inputs
from .semantic_disputed_word_risk import word_bank


FAMILY = "paired-gallery-connectivity"
ARMS = ("connected_views", "swapped_identity", "same_photo_seeds", "rgb_diffusion")


def reciprocal_index(photos):
    neighbors, similarities = nearest_gallery(photos, NEIGHBORS)
    rows = torch.arange(len(neighbors), device=photos.device)
    linked = (neighbors[neighbors] == rows[:, None, None]).any(-1)
    index = torch.where(linked, neighbors, rows[:, None]).int().contiguous()
    return index, linked, similarities


def diffusion_bank(p, index, linked, similarity):
    # The matrix uses the same reciprocal edges as the connectivity operators.
    weights = similarity.pow(3) * linked
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
    row, slot = linked.nonzero(as_tuple=True)
    isolates = torch.where(~linked.any(1))[0]
    edges = torch.stack((torch.cat((row, isolates)), torch.cat((index[row, slot].long(), isolates))))
    values = torch.cat((weights[row, slot], weights.new_ones(len(isolates))))
    graph = torch.sparse_coo_tensor(edges, values, (p.size(1), p.size(1))).coalesce().to_sparse_csr()
    return solve_diffusion(graph, p, alpha=.5, iterations=16).contiguous()


@triton.jit
def neighbor_step(values, ceiling, index, out, n: tl.constexpr, batch: tl.constexpr,
                  k: tl.constexpr, block: tl.constexpr, clip: tl.constexpr):
    row = tl.program_id(1)
    center = tl.program_id(0) * block + tl.arange(0, block)
    slot = tl.arange(0, k)
    neighbors = tl.load(index + center[:, None] * k + slot[None, :], center[:, None] < n, other=0)
    nearby = tl.load(values + row * n + neighbors, center[:, None] < n, other=0.0)
    current = tl.load(values + row * n + center, center < n, other=0.0)
    updated = tl.maximum(current, tl.max(nearby, axis=1))
    if clip:
        capacity = tl.load(ceiling + (row % batch) * n + center, center < n, other=0.0)
        updated = tl.minimum(updated, capacity)
    tl.store(out + row * n + center, updated, center < n)


def step(values, ceiling, index, result, clip):
    n, block = ceiling.size(1), 32
    neighbor_step[(triton.cdiv(n, block), len(values))](
        values, ceiling, index, result, n, len(ceiling), index.size(1), block, clip, num_warps=4,
    )


def reconstruct(seeds, capacity, index):
    a, b = seeds.contiguous(), torch.empty_like(seeds)
    for count in range(1, capacity.size(1) + 1):
        step(a, capacity, index, b, clip=True)
        if (count % 8 == 0 or count == capacity.size(1)) and torch.equal(a, b):
            return b, count
        a, b = b, a
    raise ValueError("Reconstruction did not reach a fixed point in N iterations")


def connectivity(a, b, index, swap):
    n = len(a)
    ceiling = a.maximum(b)
    left = torch.where(swap, b, a)
    right = torch.where(swap, a, b)
    all_views = torch.cat((a, b, left, right))
    maximum = torch.empty_like(all_views)
    step(all_views, ceiling, index, maximum, clip=False)
    a_near, b_near, left_near, right_near = maximum.split(n)
    direct = a.minimum(b)
    # At each threshold a component can have both colors at a vertex or
    # across an edge. A one-hop cross-color seed identifies that component.
    edge = a.minimum(b_near).maximum(b.minimum(a_near))
    swapped = left.minimum(right_near).maximum(right.minimum(left_near))
    values, rounds = reconstruct(torch.cat((edge, swapped, direct)), ceiling, index)
    return values.split(n), ceiling, direct, rounds


def certified_at(a, b, index, item, level):
    ceiling = np.maximum(a, b)
    if ceiling[item] < level:
        return False
    seen, queue = {item}, [item]
    has_first = has_second = False
    for current in queue:
        has_first |= a[current] >= level
        has_second |= b[current] >= level
        if has_first and has_second:
            return True
        for neighbor in index[current]:
            if neighbor not in seen and ceiling[neighbor] >= level:
                seen.add(neighbor)
                queue.append(neighbor)
    return False


def verify_components(a, b, index, obtained, *, every_vertex=False):
    values = tuple(t.cpu().numpy() for t in (a, b, index, obtained))
    first, second, graph, measured = values
    rng = np.random.default_rng(SEED)
    if every_vertex:
        points = ((row, i) for row in range(len(first)) for i in range(len(graph)))
    else:
        points = zip(rng.integers(len(first), size=16), rng.integers(len(graph), size=16), strict=True)
    for row, i in points:
        threshold = measured[row, i]
        next_value = np.nextafter(threshold, np.float32(np.inf))
        if not certified_at(first[row], second[row], graph, i, threshold) or certified_at(
            first[row], second[row], graph, i, next_value,
        ):
            raise ValueError("Threshold certificate differs from explicit component search")


def test_fixture(device):
    index = torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.int32, device=device)
    a = torch.tensor([[.90, .80, .60]], device=device)
    b = torch.tensor([[.10, .20, .40]], device=device)
    swap = torch.tensor([False, True, False], device=device)
    (actual, alternative, _), _, _, _ = connectivity(a, b, index, swap)
    aa, bb = torch.where(swap, b, a), torch.where(swap, a, b)
    if not torch.equal(a + b, aa + bb) or not torch.equal(a.maximum(b), aa.maximum(bb)):
        raise ValueError("Counterfactual changed the scalar score or capacity")
    verify_components(a, b, index, actual, every_vertex=True)
    verify_components(aa, bb, index, alternative, every_vertex=True)
    torch.testing.assert_close(actual, torch.full_like(actual, .4))
    torch.testing.assert_close(alternative, torch.tensor([[.8, .8, .6]], device=device))
    return {"actual": actual.tolist()[0], "swapped": alternative.tolist()[0]}


def order_with_dot_ties(levels, base):
    return (levels.contiguous().view(torch.int32).long() << 32) | base.view(torch.int32).long()


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
    photos = F.normalize(split.photo.float(), dim=1)
    index, linked, sims = reciprocal_index(photos)
    diffused = diffusion_bank(p, index, linked, sims)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    swap = torch.randint(2, (p.size(1),), generator=generator).bool().cuda()
    truth, gallery = split.sketch_labels.cuda(), split.photo_labels.cuda()
    replay = PrefixReplay(kd, split.photo_labels, split.name)
    map_k, precision_k = replay.cutoffs
    rows, audit = {name: [] for name in ARMS}, {name: [] for name in ARMS[1:3]}
    incidence = []
    seconds = 0.0
    rounds = 0
    print(split.name, "scoring every unseen query", flush=True)
    for start in range(0, len(truth), BATCH):
        sl = slice(start, min(start + BATCH, len(truth)))
        tick = time.perf_counter()
        images = torch.stack((split.sketch[sl], split.mirror[sl].float()), dim=1)
        posterior = (split.scale * (images @ text[DEFAULT_PROMPTS[0]].T)).softmax(2)
        q = posterior.mean(1)
        base = q @ p
        a, b = (view.contiguous() @ p for view in posterior.unbind(1))
        levels, cap, direct, steps = connectivity(a, b, index, swap)
        reconstructed, shuffled, node = levels
        scores = (*[order_with_dot_ties(value, base) for value in levels], q @ diffused)
        torch.cuda.synchronize()
        seconds += time.perf_counter() - tick
        rounds = max(rounds, steps)
        if torch.any(reconstructed > cap) or torch.any(reconstructed < node) or torch.any(node < direct):
            raise ValueError("Found a reconstruction outside its certified bounds")
        if start == 0:
            verify_components(a, b, index, reconstructed)
            verify_components(torch.where(swap, b, a), torch.where(swap, a, b), index, shuffled)

        base_top = prefix(base, map_k, precision_k)
        in_base = torch.zeros_like(base, dtype=torch.bool).scatter_(1, base_top, True)
        newcomer_hits(base_top, in_base, gallery, truth[sl], reference["dot"][sl, 1])
        ranked = {}
        for name, score in zip(ARMS, scores, strict=True):
            values = score_rows(score, split.sketch_labels[sl], gallery, split.name, replay, pool)
            rows[name].append(values)
            if name != "rgb_diffusion":
                ordered = prefix(score, map_k, precision_k)
                ranked[name] = newcomer_hits(ordered, in_base, gallery, truth[sl], values[:, 1])
        for name in audit:
            audit[name].append(matched_counts(ranked["connected_views"], ranked[name]))

        cutoff = base.gather(1, base_top[:, -1:])
        only_across_edge = (base < cutoff) & (reconstructed >= cutoff) & (node < cutoff)
        equivalent_null = (base < cutoff) & (shuffled >= cutoff) & (node < cutoff)
        hit = gallery[None, :] == truth[sl, None]
        incidence.append(torch.stack([mask.sum(1) for mask in (
            reconstructed > node, only_across_edge, only_across_edge & hit,
            equivalent_null, equivalent_null & hit,
        )], dim=1).cpu())
        if start == 0 or sl.stop == len(truth) or (start // BATCH + 1) % 10 == 0:
            print(split.name, sl.stop, "/", len(truth), "queries", rounds, "sweeps max", flush=True)

    measured = {name: torch.cat(parts).double() for name, parts in rows.items()}
    measured["dot"] = reference["dot"]
    combined = {name: torch.cat(parts) for name, parts in audit.items()}
    diagnostics = torch.cat(incidence)
    save_tensor({"qi": split.query_ids, "pi": split.gallery_ids, "labels": split.sketch_labels,
                 "rows": measured, "matched": combined, "threshold_counts": diagnostics}, artifact)
    table = {
        "protocol": signature, "queries": len(truth), "gallery": p.size(1), "classes": len(truth.unique()),
        "cutoffs": list(replay.cutoffs), "metrics": {name: mean_percent(value[:, :2]) for name, value in measured.items()},
        "primary_ap_vs_control": {name: class_interval(
            100 * (measured["connected_views"][:, 0] - value[:, 0]), split.sketch_labels, seed=SEED,
        ) for name, value in measured.items() if name != "connected_views"},
        "matched_promotions": {name: summarize(value, split.sketch_labels) for name, value in combined.items()},
        "component_effects": dict(zip(("raised_over_same_photo_scores", "edge_only_promotions", "edge_only_hits",
                                       "swapped_edge_only_promotions", "swapped_edge_only_hits"),
                                      diagnostics.sum(0).tolist(), strict=True)),
        "reciprocal_edge_count": int(linked.sum()) // 2, "scoring_seconds_all_arms": seconds,
        "max_fixed_point_sweeps": rounds, "per_query_sha256": checksum(artifact),
    }
    atomic_json(target, table)
    print(split.name, "verified", table["metrics"], flush=True)


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
    cfg = runtime(workspace, "sketchy", batch=384, workers=workers)
    signature = save_lock(root / "freeze.json", {
        "commit": revision(), "clip_sha256": CHECKPOINT, "evaluator_sha256": KD_FUNCTION_SHA,
        "vocabulary_sha256": vocab_sha, "prompts": list(DEFAULT_PROMPTS), "seed": SEED,
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "triton": triton.__version__,
        "neighbors": NEIGHBORS, "graph": "reciprocal RGB cosine neighborhoods", "views": ["original", "horizontal_flip"],
        "selection": "fixed before running any split", "diffusion": {"alpha": .5, "steps": 16},
        "fixture": test_fixture(torch.device("cuda")),
    })
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
