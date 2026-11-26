"""Compare only the vocabulary in the existing global S ranker."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .depiction_path_diagnostic import class_interval
from .encoding import save_tensor
from .foundation import DEFAULT_DOMAIN_TEMPLATES
from .geometry_word_flat import COUNTS, SHARD
from .locked_geometry_words import score_rows
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic import read_words
from .semantic_disputed_word_risk import SEED, archived_semantic, inputs, query_posteriors, word_bank


FAMILY = "semantic-open-images-vocabulary"
ARMS = ("full", "boxable", "random")
BATCH = 96
ASSETS = {
    "open-images-v7-classes.csv": "84a4373a0efb7fd6d93fe19b0e7ceb6c1b855c233d13b9b78a9a33655c9fdce3",
    "open-images-v7-boxable.csv": "1839e0e7e84130ae281f7f67413768601b031581c0c42e7fc17527b8e2a99aa9",
}


def freeze(workspace, root):
    full_file, boxable_file = [workspace / "assets" / name for name in ASSETS]
    for file in (full_file, boxable_file):
        if checksum(file) != ASSETS[file.name]:
            raise ValueError(f"Unexpected vocabulary file: {file.name}")
    words = read_words(full_file)
    boxable = read_words(boxable_file)
    positions = {word: i for i, word in enumerate(words)}
    if (len(words), len(boxable)) != (20931, 601) or not positions.keys() >= set(boxable):
        raise ValueError("Expected the full bank and its official boxable subset")
    boxable_ids = [positions[word] for word in boxable]
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    random_ids = torch.randperm(len(words), generator=generator)[:len(boxable)].sort().values.tolist()
    subsets = {"boxable": boxable_ids, "random": random_ids}
    spec = {
        "commit": revision(),
        "seed": SEED,
        "rng": "torch.randperm on CPU with a dedicated generator, sampled without replacement",
        "assets": ASSETS,
        "clip_sha": CHECKPOINT,
        "evaluator_sha": KD_FUNCTION_SHA,
        "prompts": [list(group) for group in DEFAULT_DOMAIN_TEMPLATES],
        "full_vocab_sha": digest(words),
        "word_counts": {"full": len(words), **{name: len(ids) for name, ids in subsets.items()}},
        "subsets": {
            name: {"indices": ids, "words": [words[i] for i in ids]} for name, ids in subsets.items()
        },
        "subset_overlap": len(set(boxable_ids) & set(random_ids)),
        "score": "Dot of sketch pose-averaged and photo posteriors, each softmax within the vocabulary",
    }
    lock = spec | {"sha256": digest(spec)}
    file = root / "freeze.json"
    if file.is_file():
        if json.loads(file.read_bytes()) != lock:
            raise ValueError("The experiment differs from the vocabulary lock")
    else:
        atomic_json(file, lock)
    return lock, {name: torch.tensor(ids, dtype=torch.long, device="cuda") for name, ids in subsets.items()}


def contrasts(rows, labels):
    paired = {}
    for a, b in (("boxable", "full"), ("random", "full"), ("boxable", "random")):
        diff = 100 * (rows[a].double() - rows[b].double())
        paired[f"{a}_minus_{b}"] = {
            "points": diff.mean(0).tolist(),
            "macro_ap_delta": class_interval(diff[:, 0], labels, seed=SEED),
        }
    return paired


def score_split(root, split, queries, gallery, tensors, banks, scale, lock, kd):
    qi, labels = queries
    pi, photo_labels = gallery
    if (len(qi), len(pi), len(labels.unique())) != COUNTS[split] or len(photo_labels.unique()) != COUNTS[split][2]:
        raise ValueError(f"Unexpected unseen split {split}")
    reference = archived_semantic(split, qi, pi, labels)
    device = torch.cuda.get_device_name()
    signature = digest({
        "freeze": lock["sha256"], "gpu": device, "torch": str(torch.__version__),
        "split": split, "queries": qi.tolist(), "photos": pi.tolist(),
    })
    file = root / f"{split}.json"
    archive = file.with_suffix(".pt")
    if file.exists() and archive.exists():
        result = json.loads(file.read_bytes())
        if result["recipe"] == signature and result["per_query_sha256"] == checksum(archive):
            print(split, "already complete", flush=True)
            return

    started = time.perf_counter()
    q, p, mirror = tensors
    query, reflected, photos = q[qi].cuda(), mirror[qi].cuda(), p[pi].cuda()
    posteriors = {name: word_bank(photos, bank[1], scale) for name, bank in banks.items()}
    truth = photo_labels.cuda()
    replay = PrefixReplay(kd, photo_labels, split)
    chunks = root / split / signature
    rows = {name: [] for name in ARMS}
    cpu_workers = min(4, len(os.sched_getaffinity(0)))
    print(split, "ranking", len(qi), "queries against", len(pi), "photos", flush=True)
    with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
        for start in range(0, len(qi), SHARD):
            stop = min(start + SHARD, len(qi))
            shard = chunks / f"{start:05d}.pt"
            if shard.exists():
                record = torch.load(shard, weights_only=True)
            else:
                block = {name: [] for name in ARMS}
                for lo in range(start, stop, BATCH):
                    batch = slice(lo, min(lo + BATCH, stop))
                    for name in ARMS:
                        reading = query_posteriors(query[batch], reflected[batch], banks[name][0], scale)
                        scores = reading @ posteriors[name]
                        block[name].append(score_rows(scores, labels[batch], truth, split, replay, pool)[:, :2])
                record = {"ids": qi[start:stop], "rows": {name: torch.cat(parts) for name, parts in block.items()}}
                save_tensor(record, shard)
            if not torch.equal(qi[start:stop], record["ids"]):
                raise ValueError(f"Unexpected query order in {shard}")
            for name in ARMS:
                rows[name].append(record["rows"][name])
            if stop == len(qi) or start % (4 * SHARD) == 0:
                print(split, stop, "/", len(qi), flush=True)
    metrics = {name: torch.cat(pieces).double() for name, pieces in rows.items()}
    for name, values in metrics.items():
        if values.shape != (len(qi), 2) or not values.isfinite().all():
            raise ValueError(f"Incomplete metrics for {split}/{name}")
    replay_mae = 100 * (metrics["full"] - reference.double()).abs().mean(0)
    if replay_mae.max() > .04:
        raise ValueError(f"Full S differs from the archived control: {replay_mae.tolist()}")

    save_tensor({"recipe": signature, "qi": qi, "pi": pi, "labels": labels, "rows": metrics}, archive)
    result = {
        "split": split, "recipe": signature, "freeze_sha256": lock["sha256"], "commit": lock["commit"],
        "device": device, "torch": str(torch.__version__), "seed": SEED,
        "clip_sha": CHECKPOINT, "evaluator_sha": KD_FUNCTION_SHA,
        "queries": len(qi), "gallery": len(pi), "classes": len(labels.unique()),
        "cutoffs": dict(zip(("mAP", "precision"), replay.cutoffs, strict=True)),
        "vocabulary_sizes": lock["word_counts"],
        "metrics": {name: (100 * values.mean(0)).tolist() for name, values in metrics.items()},
        "paired_delta": contrasts(metrics, labels),
        "full_control_mae_points": replay_mae.tolist(),
        "per_query_sha256": checksum(archive),
        "ranking_seconds": time.perf_counter() - started,
    }
    atomic_json(file, result)
    print(split, result["metrics"], flush=True)


@torch.inference_mode()
def run(dataset):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    workspace = Path("/content/sbir")
    root = workspace / "results" / FAMILY
    lock, subsets = freeze(workspace, root)
    print("Vocabulary frozen", lock["word_counts"], "overlap", lock["subset_overlap"], lock["sha256"], flush=True)
    kd, _ = load_kd(workspace)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("KD-SBIR evaluator has changed")
    qs, gs, q, p, mirror, source, target, scale, vocab = inputs(workspace, dataset)
    if vocab != lock["full_vocab_sha"]:
        raise ValueError("Text embeddings differ from the frozen vocabulary")
    banks = {"full": (source, target)}
    for name, ids in subsets.items():
        banks[name] = (source.index_select(0, ids), target.index_select(0, ids))
    for split, queries in qs.items():
        score_split(root, split, queries, gs[split], (q, p, mirror), banks, scale, lock, kd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("sketchy", "tuberlin"))
    run(parser.parse_args().dataset)
