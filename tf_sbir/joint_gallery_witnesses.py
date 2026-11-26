"""Retrieve with a single common lexical subdistribution supported around each photo."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import io
from itertools import combinations
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

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT, runtime
from .depiction_path_diagnostic import class_interval
from .encoding import load_frozen_clip, save_tensor
from .gallery_convex_support import (
    ANGLE_TOLERANCE, NEIGHBORS, SEED, direction_masks, pooled_order,
    support_floor, tangent_coordinates,
)
from .gallery_convex_support_audit import matched_counts, newcomer_hits, prefix, summarize
from .gallery_diffusion import nearest_gallery
from .locked_geometry_words import score_rows
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import BATCH, SKETCHY, Split, mean_percent, posteriors
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt
from .semantic_boxable_sparse import image_inputs


FAMILY = "joint-gallery-witnesses"
REFERENCE = DOCS / "gallery-convex-support"
GALLERY_BLOCK = 1024
GEMM_BATCH = 768


def combinations_index(device):
    return tuple(torch.tensor(list(combinations(range(NEIGHBORS), size)), device=device)
                 for size in (3, 4))


def minimal_groups(masks, triples, quadruples):
    """Return the inclusion-minimal covers of all open halfplanes in owner order."""
    a, b, c = masks[:, triples].unbind(dim=-1)
    triangles = (a | b | c) == -1
    a, b, c, d = masks[:, quadruples].unbind(dim=-1)
    has_triangle = ((a | b | c) == -1) | ((a | b | d) == -1) | ((a | c | d) == -1) | ((b | c | d) == -1)
    quadrangles = ((a | b | c | d) == -1) & ~has_triangle
    ti, tj = triangles.nonzero(as_tuple=True)
    qi, qj = quadrangles.nonzero(as_tuple=True)
    tri_slots = triples[tj]
    slots = torch.cat((torch.cat((tri_slots, tri_slots[:, :1]), dim=1), quadruples[qj]), dim=0)
    owner = torch.cat((ti, qi))
    order = owner.argsort(stable=True)
    sizes = triangles.sum(1) + quadrangles.sum(1)
    return owner[order], slots[order], sizes, len(qi)


def verify_minimal_groups(masks, triples, quadruples):
    """Compare the planar bound above with exhaustive subsets on 32 actual charts."""
    rng = np.random.default_rng(SEED)
    selections = rng.choice(len(masks), size=min(32, len(masks)), replace=False)
    sample = masks[torch.from_numpy(selections).to(masks.device)]
    owner, slots, _, _ = minimal_groups(sample, triples, quadruples)
    bits = np.bitwise_or.reduce(1 << slots.cpu().numpy(), axis=1)
    union = np.zeros((len(sample), 1), dtype=np.int32)
    for cover in sample.cpu().numpy().T:
        union = np.concatenate((union, union | cover[:, None]), axis=1)
    valid = union == -1
    minimal = valid.copy()
    for bit in range(NEIGHBORS):
        width = 1 << bit
        minimal.reshape(len(sample), -1, 2, width)[:, :, 1] &= ~valid.reshape(
            len(sample), -1, 2, width,
        )[:, :, 0]
    known_owner, known_bits = minimal.nonzero()
    key = owner.cpu().numpy() * (1 << NEIGHBORS) + bits
    expected = known_owner * (1 << NEIGHBORS) + known_bits
    if not np.array_equal(np.sort(key), expected):
        raise ValueError("Minimal witness sets disagree with exhaustive gallery subsets")


def all_joint_support(q, p, neighbors, masks):
    """Compute the maximum common-mass score for each query and gallery image."""
    triples, quadruples = combinations_index(masks.device)
    verify_minimal_groups(masks, triples, quadruples)
    photo_words = p.T.contiguous()
    support = p.new_zeros((len(q), p.size(1)))
    stats = {"triangles": 0, "quadrangles": 0, "photos_with_witnesses": 0}
    for start in range(0, p.size(1), GALLERY_BLOCK):
        stop = min(start + GALLERY_BLOCK, p.size(1))
        owner, slots, sizes, n_quad = minimal_groups(masks[start:stop], triples, quadruples)
        if owner.numel() == 0:
            continue
        stats["triangles"] += len(owner) - n_quad
        stats["quadrangles"] += n_quad
        stats["photos_with_witnesses"] += int((sizes > 0).sum())
        ids = neighbors[start + owner[:, None], slots]
        shared = photo_words[ids[:, 0]]
        for witness in ids.T[1:]:
            torch.minimum(shared, photo_words[witness], out=shared)
        codebook = shared.T
        for offset in range(0, len(q), GEMM_BATCH):
            qs = q[offset:offset + GEMM_BATCH]
            evidence = qs @ codebook
            counts = sizes[None, :].expand(len(qs), -1).contiguous()
            support[offset:offset + len(qs), start:stop] = torch.segment_reduce(
                evidence, "max", lengths=counts, axis=1, initial=0.0, unsafe=True,
            )
    return support, stats


def verify_toy(device):
    """Alternating explanations can each enclose a point without one shared signature."""
    count = NEIGHBORS + 1
    ids = torch.arange(1, count, dtype=torch.int32, device=device).repeat(count, 1)
    direction = torch.tensor([0, 60, 120, 180, 240, 300] + [0] * 10, device=device).deg2rad()
    masks = direction_masks(torch.stack((direction.cos(), direction.sin()), dim=1).repeat(count, 1, 1))
    one_word = torch.tensor([[.01, .01, .98]] + [[.89, .01, .10]] * NEIGHBORS, device=device).T.contiguous()
    two_sectors = one_word.clone()
    two_sectors[:2, 3:6] = two_sectors[:2, 3:6].flip(0)
    alternating = one_word.clone()
    alternating[:2, 2:7:2] = alternating[:2, 2:7:2].flip(0)
    q = torch.tensor([[.5, .5, 0.]], device=device)
    dot = q @ one_word
    pooled = support_floor(dot, ids, masks)
    independent = q @ alternating.maximum(support_floor(alternating, ids, masks))
    a, b, c = (all_joint_support(q, p, ids, masks)[0] for p in (one_word, two_sectors, alternating))
    if not torch.equal(dot, q @ two_sectors) or not torch.equal(dot, q @ alternating):
        raise ValueError("Toy galleries must have exactly the same scalar Dot scores")
    values = torch.stack((dot[0, 0], pooled[0, 0], independent[0, 0], a[0, 0], b[0, 0], c[0, 0]))
    if not torch.allclose(values, torch.tensor([.01, .45, .89, .45, .01, .45], device=device)):
        raise ValueError(f"Failed the common-mass counterexample: {values.tolist()}")
    return dict(zip(("dot", "pooled", "independent_alternating", "joint_one_word",
                     "joint_two_sectors", "joint_alternating"), values.tolist(), strict=True))


def load_reference(split):
    report = json.loads((REFERENCE / f"{split.name}.json").read_bytes())
    with zipfile.ZipFile(REFERENCE / "paired-T4.zip") as archive:
        blob = archive.read(f"{split.name}.pt")
    if hashlib.sha256(blob).hexdigest() != report["per_query_sha256"]:
        raise ValueError("Reference artifact checksum mismatch")
    saved = torch.load(io.BytesIO(blob), weights_only=True)
    if any(not torch.equal(saved[key], expected) for key, expected in (
        ("qi", split.query_ids), ("pi", split.gallery_ids), ("labels", split.sketch_labels),
    )):
        raise ValueError("Reference artifact used a different gallery or query order")
    return saved["rows"], report


def compare(root, split, text, kd, pool, signature):
    target = root / f"{split.name}.json"
    if target.is_file():
        report = json.loads(target.read_bytes())
        if report["protocol"] != signature or report["per_query_sha256"] != checksum(target.with_suffix(".pt")):
            raise ValueError("Invalid previous result")
        print(split.name, "verified prior result", flush=True)
        return
    rows, reference = load_reference(split)
    query, gallery = posteriors(split, text, [DEFAULT_PROMPTS])
    q, p = query[DEFAULT_PROMPTS[0]], gallery[DEFAULT_PROMPTS[1]]
    started = time.perf_counter()
    photos = F.normalize(split.photo.float(), dim=1)
    index, _ = nearest_gallery(photos, NEIGHBORS)
    index = index.int().contiguous()
    coordinates, _, _ = tangent_coordinates(photos, index)
    masks = direction_masks(coordinates)
    independent = p.maximum(support_floor(p, index, masks))
    print(split.name, "scoring common witness groups", flush=True)
    support, groups = all_joint_support(q, p, index, masks)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(split.name, "joint scores ready", groups, round(elapsed, 2), "seconds", flush=True)

    labels, truth = split.photo_labels.cuda(), split.sketch_labels.cuda()
    replay = PrefixReplay(kd, split.photo_labels, split.name)
    map_k, precision_k = replay.cutoffs
    primary, contrasts = [], {name: [] for name in ("pooled_hull", "concept_hull")}
    intervention = []
    worst_violation = 0.0
    for start in range(0, len(q), BATCH):
        qs = q[start:start + BATCH]
        sl = slice(start, start + len(qs))
        base = qs @ p
        pooled = support_floor(base, index, masks)
        joint = support[sl]
        worst_violation = max(worst_violation, (joint - pooled).max().item())
        score = pooled_order(base, joint)
        values = score_rows(score, split.sketch_labels[sl], labels, split.name, replay, pool)
        primary.append(values)
        top = prefix(base, map_k, precision_k)
        in_base = torch.zeros_like(base, dtype=torch.bool).scatter_(1, top, True)
        newcomer_hits(top, in_base, labels, truth[sl], rows["dot"][sl, 1])
        first = newcomer_hits(prefix(score, map_k, precision_k), in_base, labels, truth[sl], values[:, 1])
        controls = (pooled_order(base, pooled), qs @ independent)
        for name, data in zip(contrasts, controls, strict=True):
            second = newcomer_hits(prefix(data, map_k, precision_k), in_base, labels, truth[sl], rows[name][sl, 1])
            contrasts[name].append(matched_counts(first, second))
        cutoff = base.gather(1, top[:, -1:])
        outside = base < cutoff
        accepted = outside & (joint >= cutoff)
        scalar_only = outside & (pooled >= cutoff) & (joint < cutoff)
        hit = labels[None, :] == truth[sl, None]
        intervention.append(torch.stack([mask.sum(1) for mask in (
            accepted, accepted & hit, scalar_only, scalar_only & hit,
        )], dim=1).cpu())
        if start == 0 or sl.stop == len(q) or (start // BATCH + 1) % 10 == 0:
            print(split.name, "original KD-SBIR", sl.stop, "/", len(q), flush=True)

    if worst_violation > 2e-6:
        raise ValueError("Common lexical mass exceeded the scalar support upper bound")
    result = torch.cat(primary).double()
    counts = torch.cat(intervention)
    a, ah, b, bh = counts.sum(0).tolist()
    matched = {name: torch.cat(parts) for name, parts in contrasts.items()}
    artifact = target.with_suffix(".pt")
    save_tensor({"qi": split.query_ids, "pi": split.gallery_ids, "labels": split.sketch_labels,
                 "joint": result, "threshold_counts": counts, "matched_promotions": matched}, artifact)
    table = {
        "protocol": signature, "queries": len(q), "gallery": p.size(1), "classes": len(truth.unique()),
        "cutoffs": list(replay.cutoffs), "metrics": mean_percent(result[:, :2]),
        "reference_metrics": reference["metrics"],
        "delta_macro_ap": {name: class_interval(100 * (result[:, 0] - previous[:, 0]),
                                                split.sketch_labels, seed=SEED) for name, previous in rows.items()},
        "witness_groups": groups, "wall_seconds_before_evaluation": elapsed,
        "matched_promotions": {name: summarize(value, split.sketch_labels) for name, value in matched.items()},
        "threshold_retained_and_rejected": {"joint": [ah, a], "scalar_only": [bh, b]},
        "max_joint_minus_scalar_floor": worst_violation,
        "per_query_sha256": checksum(artifact),
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
        raise ValueError("Expected the locked 601 boxable concepts")
    kd, _ = load_kd(workspace)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("KD-SBIR source has changed")
    workers = min(4, len(os.sched_getaffinity(0)))
    cfg = runtime(workspace, "sketchy", batch=384, workers=workers)
    signature = save_lock(root / "freeze.json", {
        "commit": revision(), "clip_sha256": CHECKPOINT, "evaluator_sha256": KD_FUNCTION_SHA,
        "vocabulary_sha256": vocab_sha, "prompts": list(DEFAULT_PROMPTS), "seed": SEED,
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "neighbors": NEIGHBORS,
        "angle_tolerance": ANGLE_TOLERANCE, "gallery_block": GALLERY_BLOCK, "gemm_batch": GEMM_BATCH,
        "control_reference_sha256": checksum(REFERENCE / "paired-T4.zip"),
        "selection": "All settings fixed from the mechanism before evaluating this branch",
        "toy_fixture": verify_toy(torch.device("cuda")),
    })
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = {prompt: encode_prompt(model, tokenizer, words, prompt) for prompt in DEFAULT_PROMPTS}
    data = image_inputs(cfg, SKETCHY, model)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name in SKETCHY:
            compare(root, Split.from_inputs(data, name), text, kd, pool, signature)
        del data
        cfg = runtime(workspace, "tuberlin", batch=384, workers=workers)
        split = Split.from_inputs(image_inputs(cfg, ("tuberlin",), model), "tuberlin")
        compare(root, split, text, kd, pool, signature)


if __name__ == "__main__":
    run()
