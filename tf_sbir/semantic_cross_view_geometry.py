"""Check if cross-view geometric verification adds evidence to boxable S."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import random

import numpy as np
import open_clip
import torch

from .benchmark import merge_inputs
from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .cross_view_geometry import audit_geometry, evidence, grid_controls
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT, runtime
from .depiction_path_diagnostic import class_interval
from .depiction_pivot_geometry import native_patches
from .encoding import load_frozen_clip, save_tensor
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_boxable import DEFAULT_PROMPTS, FAMILY as BASELINE
from .semantic_boxable_prompt_search import BATCH, PAIRS, SKETCHY, Split, mean_percent, posteriors
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt
from .semantic_disputed_word_risk import SEED, inputs


FAMILY = "semantic-cross-view-geometry"
HEAD = 256
PAIR_BATCH = 16
WEIGHTS = (.125, .25, .5, 1., 2.)
SIGNALS = ("appearance", "distance", "within_view", "cross_view",
           "coordinate_contrast", "wrong_frame", "radial_shuffle")
PRIMARY = ("cross_view", "coordinate_contrast")
VARIANTS = {"S": None}
VARIANTS.update({f"{name}/{weight:g}": (name, weight) for name in SIGNALS for weight in WEIGHTS})


def standardized(values):
    return (values - values.mean(1, keepdim=True)) / values.std(1, correction=0, keepdim=True).clamp_min(1e-6)


def head_order(logits, spatial, arm):
    if arm is None:
        return torch.arange(logits.shape[1], device=logits.device).expand_as(logits)
    name, weight = arm
    scores = logits + weight * standardized(spatial[name])
    return scores.argsort(dim=1, descending=True, stable=True)


class RankedBatch:
    """Measure the original rank metrics after permuting a fixed candidate head."""

    def __init__(self, scores, labels, photo_labels, cutoffs):
        self.ranks = scores.topk(scores.shape[1]).indices
        self.logits = standardized(scores.gather(1, self.ranks[:, :HEAD]).clamp_min(1e-30).log())
        hits = photo_labels[self.ranks] == labels[:, None]
        self.head_hits = hits[:, :HEAD]
        map_k, self.precision_k = cutoffs
        self.ap_head = min(map_k or HEAD, HEAD)
        self.tail_ap = 0.
        self.count = None
        self.positions = torch.arange(1, hits.size(1) + 1, device=scores.device)
        if map_k == 0:
            cumulative = hits.cumsum(1, dtype=torch.int32)
            tail = hits[:, HEAD:] * (cumulative[:, HEAD:] / self.positions[HEAD:])
            self.tail_ap = tail.sum(1)
            self.count = cumulative[:, -1].clamp_min(1)
        self.limit = map_k or hits.shape[1]

    def metrics(self, order):
        hits = self.head_hits.gather(1, order)
        ap_hits = hits[:, :self.ap_head]
        cumulative = ap_hits.cumsum(1, dtype=torch.int32)
        count = self.count if self.count is not None else cumulative[:, -1].clamp_min(1)
        ap = ((ap_hits * (cumulative / self.positions[:self.ap_head])).sum(1) + self.tail_ap) / count
        p = hits[:, :self.precision_k].float().mean(1)
        return torch.stack((ap, p), dim=1)

    def original(self, order, labels, replay, pool):
        re_ranked = self.ranks[:, :HEAD].gather(1, order)
        rank = torch.cat((re_ranked, self.ranks[:, HEAD:]), dim=1)[:, :self.limit]
        exact = replay.score(rank, labels, pool)
        if (exact - self.metrics(order).cpu()).abs().max() > 2e-6:
            raise ValueError("Head permutation metrics disagree with original KD-SBIR")
        return exact


def load_baseline(workspace, split):
    base = workspace / "results" / BASELINE / f"{split.name}.pt"
    saved = torch.load(base, weights_only=True, map_location="cpu")
    expected = {"qi": split.query_ids, "pi": split.gallery_ids, "labels": split.sketch_labels}
    if any(not torch.equal(value, saved[name]) for name, value in expected.items()):
        raise ValueError(f"S was measured on different samples in {split.name}")
    return saved["rows"][f"pair_{PAIRS.index(DEFAULT_PROMPTS)}"]


def prepare_patches(workspace, dataset, names, model):
    cfg = runtime(workspace, dataset, batch=384, workers=min(4, len(os.sched_getaffinity(0))))
    sketches, _ = merge_inputs(cfg.root, names, "sketch")
    photos, _ = merge_inputs(cfg.root, names, "photo")
    folder = workspace / "cache/depiction-pivot-geometry"
    print(dataset, "encoding native CLIP patch grids", flush=True)
    query = native_patches(sketches, model.visual, folder).cuda()
    mirror = native_patches(sketches, model.visual, folder, reflected=True).cuda()
    gallery = native_patches(photos, model.visual, folder).cuda()
    return query, mirror, gallery


def patch_signals(q, q_flip, photos, candidates, scale, grid):
    coords, flip, shuffled, distance = grid
    result = {name: [] for name in SIGNALS}
    for i in range(0, len(candidates), PAIR_BATCH):
        sl = slice(i, i + PAIR_BATCH)
        values = evidence(q[sl], q_flip[sl], photos[candidates[sl]], scale, coords, flip, shuffled, distance)
        for name, value in values.items():
            result[name].append(value)
    return {name: torch.cat(parts) for name, parts in result.items()}


def blocks(root, split, banks, patches, replay, recipe, grid):
    source, target = banks
    photo_labels, sketch_labels = split.photo_labels.cuda(), split.sketch_labels.cuda()
    if patches is not None:
        query, mirror, gallery = patches
        split_photos = gallery[split.gallery_ids.cuda()]
    folder = root / split.name / recipe
    for first in range(0, len(source), BATCH):
        sl = slice(first, min(first+BATCH, len(source)))
        batch = RankedBatch(source[sl] @ target, sketch_labels[sl], photo_labels, replay.cutoffs)
        candidates = batch.ranks[:, :HEAD]
        path = folder / f"{first:05d}.pt"
        if path.is_file():
            shard = torch.load(path, weights_only=True)
            if not torch.equal(shard["ids"], split.query_ids[sl]) or not torch.equal(shard["head"], candidates.cpu()):
                raise ValueError(f"Candidate order changed in {split.name}")
            spatial = {name: value.cuda() for name, value in shard["signals"].items()}
        else:
            if patches is None:
                raise FileNotFoundError(path)
            ids = split.query_ids[sl].cuda()
            spatial = patch_signals(query[ids], mirror[ids], split_photos, candidates, split.scale, grid)
            if any(not value.isfinite().all() for value in spatial.values()):
                raise ValueError(f"Nonfinite geometry in {split.name}")
            save_tensor({"ids": split.query_ids[sl], "head": candidates.cpu(),
                         "signals": {name: value.cpu() for name, value in spatial.items()}}, path)
        yield sl, batch, spatial
        if first % (5 * BATCH) == 0 or sl.stop == len(source):
            print(split.name, sl.stop, "/", len(source), "queries", flush=True)


def measure(root, workspace, split, banks, patches, kd, pool, grid, protocol, *, official=None, selection=None):
    query, gallery = posteriors(split, banks, [DEFAULT_PROMPTS])
    source, target = query[DEFAULT_PROMPTS[0]], gallery[DEFAULT_PROMPTS[1]]
    replay = PrefixReplay(kd, split.photo_labels, split.name)
    recipe = digest({"protocol": protocol, "ids": split.query_ids.tolist(), "gallery": split.gallery_ids.tolist()})
    options = VARIANTS if official is None else official
    collected = {key: [] for key in options}
    for sl, batch, spatial in blocks(root, split, (source, target), patches, replay, recipe, grid):
        for key, config in options.items():
            order = head_order(batch.logits, spatial, config)
            if official is not None and key != "S":
                values = batch.original(order, split.sketch_labels[sl], replay, pool)
            else:
                values = batch.metrics(order).cpu()
            collected[key].append(values)
    rows = {key: torch.cat(parts).double() for key, parts in collected.items()}
    baseline = load_baseline(workspace, split)
    error = 100 * (rows["S"] - baseline).abs().mean(0)
    if error.max() > .04:
        raise ValueError(f"S replay differs from the archived boxable baseline: {error.tolist()}")
    rows["S"] = baseline
    result = {
        "commit": revision(), "protocol_sha256": protocol, "selection_sha256": selection, "recipe": recipe,
        "queries": len(split.query_ids), "gallery": len(split.gallery_ids), "classes": len(split.sketch_labels.unique()),
        "cutoffs": {"mAP": replay.cutoffs[0], "precision": replay.cutoffs[1]},
        "metrics": {key: mean_percent(value) for key, value in rows.items()},
        "base_mae_points": error.tolist(),
    }
    if official is None:
        atomic_json(root / f"grid-{split.name}.json", result)
    else:
        result["vs_S"] = {
            key: class_interval(100 * (value[:, 0] - baseline[:, 0]), split.sketch_labels, seed=SEED)
            for key, value in rows.items() if key != "S"
        }
        artifact = root / f"{split.name}.pt"
        save_tensor({"qi": split.query_ids, "pi": split.gallery_ids, "labels": split.sketch_labels, "rows": rows}, artifact)
        result["per_query_sha256"] = checksum(artifact)
        atomic_json(artifact.with_suffix(".json"), result)
        print(split.name, "official", result["metrics"], flush=True)
    return result["metrics"]


def choose(tables):
    def mean(key):
        return tuple(sum(table[key][j] for table in tables) / len(tables) for j in (0, 1))
    choices = {name: max((key for key in VARIANTS if key.startswith(name+"/")), key=mean) for name in SIGNALS}
    winner = max((choices[name] for name in PRIMARY), key=mean)
    return winner, choices, mean(winner)


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
    lock = json.loads((DOCS / VOCABULARY_STUDY / "freeze.json").read_bytes())
    word_sha = lock.pop("sha256")
    words = lock["subsets"]["boxable"]["words"]
    if len(words) != 601 or digest(lock) != word_sha:
        raise ValueError("Expected the frozen boxable vocabulary")
    kd, _ = load_kd(workspace)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("KD-SBIR evaluator has changed")
    cfg = runtime(workspace, "sketchy")
    coord, flip, shuffled = grid_controls("cuda", SEED)
    sanity = audit_geometry(coord, flip)
    print("Geometry sanity scores", sanity, flush=True)
    grid = (coord, flip, shuffled, torch.cdist(coord, coord))
    spec = {
        "commit": revision(), "clip_sha256": CHECKPOINT, "kd_sha256": KD_FUNCTION_SHA,
        "vocab_sha256": word_sha, "words": len(words), "prompts": list(DEFAULT_PROMPTS),
        "device": torch.cuda.get_device_name(), "torch": str(torch.__version__), "seed": SEED,
        "candidates": HEAD, "weights": list(WEIGHTS), "signals": list(SIGNALS),
        "sanity": sanity, "selection": "Mean full unseen Sketchy mAP, then mean precision",
    }
    protocol = save_lock(root / "protocol.json", spec)
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = {prompt: encode_prompt(model, tokenizer, words, prompt) for prompt in DEFAULT_PROMPTS}
    data = inputs(workspace, "sketchy")
    if data[-1] != lock["full_vocab_sha"]:
        raise ValueError("Sketchy text input changed")
    patch_cache = prepare_patches(workspace, "sketchy", SKETCHY, model)
    workers = min(4, len(os.sched_getaffinity(0)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tables = [measure(root, workspace, Split.from_inputs(data, name), text, patch_cache, kd, pool, grid, protocol)
                  for name in SKETCHY]
        winner, alternatives, mean = choose(tables)
        weight = VARIANTS[winner][1]
        official = {
            "S": None, "S+geometry": VARIANTS[winner], "S+appearance": VARIANTS[alternatives["appearance"]],
            "S+within_view": ("within_view", weight), "S+wrong_frame": ("wrong_frame", weight),
            "S+radial_shuffle": ("radial_shuffle", weight),
        }
        selected = {
            "protocol_sha256": protocol, "commit": revision(), "winner": winner,
            "choices_each_signal": alternatives, "mean_sketchy_metrics": mean,
            "official": {name: list(value) if value is not None else None for name, value in official.items()},
        }
        selection = save_lock(root / "freeze.json", selected)
        print("Frozen for TU-Berlin", selected, flush=True)
        del patch_cache
        for name in SKETCHY:
            measure(root, workspace, Split.from_inputs(data, name), text, None, kd, pool, grid,
                    protocol, official=official, selection=selection)
        del data
        data = inputs(workspace, "tuberlin")
        if data[-1] != lock["full_vocab_sha"]:
            raise ValueError("TU-Berlin text input changed")
        patches = prepare_patches(workspace, "tuberlin", ("tuberlin",), model)
        measure(root, workspace, Split.from_inputs(data, "tuberlin"), text, patches, kd, pool, grid,
                protocol, official=official, selection=selection)


if __name__ == "__main__":
    run()
