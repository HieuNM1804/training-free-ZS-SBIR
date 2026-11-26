"""Ablate the text prompt ensemble of S on the locked 601-word vocabulary."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import random
import zipfile

import numpy as np
import open_clip
import torch
from torch.nn import functional as F

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT, runtime
from .depiction_path_diagnostic import class_interval
from .encoding import load_frozen_clip, save_tensor
from .foundation import DEFAULT_DOMAIN_TEMPLATES
from .geometry_word_flat import COUNTS
from .locked_geometry_words import score_rows
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_disputed_word_risk import SEED, inputs, query_posteriors, word_bank
from .semantic_open_images_vocabulary import BATCH


FAMILY = "semantic-boxable-single-prompt"
REFERENCE = Path(__file__).resolve().parents[1] / "docs/results/semantic-open-images-vocabulary"
SKETCH_PROMPT = DEFAULT_DOMAIN_TEMPLATES[0][1]
PHOTO_PROMPT = DEFAULT_DOMAIN_TEMPLATES[1][1]


def encode_prompt(model, tokenizer, words, prompt):
    batches = []
    for start in range(0, len(words), 256):
        text = tokenizer([prompt.format(word) for word in words[start:start + 256]]).cuda()
        batches.append(F.normalize(model.encode_text(text).float(), dim=-1))
    return torch.cat(batches)


def previous_rows(split, qi, pi, labels):
    with zipfile.ZipFile(REFERENCE / "paired-T4.zip") as archive:
        prior = torch.load(io.BytesIO(archive.read(f"{split}.pt")), weights_only=True)
    if not all(torch.equal(prior[name], value) for name, value in (("qi", qi), ("pi", pi), ("labels", labels))):
        raise ValueError(f"Query or gallery order changed for {split}")
    return prior["rows"]["boxable"]


def score_split(root, split, queries, gallery, images, single_sketch, photo_ensemble, single_photo, scale, lock, kd):
    qi, labels = queries
    pi, photo_labels = gallery
    if (len(qi), len(pi), len(labels.unique())) != COUNTS[split] or len(photo_labels.unique()) != COUNTS[split][2]:
        raise ValueError(f"Unexpected unseen split {split}")
    baseline = previous_rows(split, qi, pi, labels)
    device = torch.cuda.get_device_name()
    recipe = digest({"freeze": lock["sha256"], "split": split, "gpu": device, "torch": str(torch.__version__)})
    report = root / f"{split}.json"
    archive = report.with_suffix(".pt")
    if report.is_file() and archive.is_file():
        prior = json.loads(report.read_bytes())
        if prior["recipe"] == recipe and prior["per_query_sha256"] == checksum(archive):
            print(split, "already complete", flush=True)
            return

    query, mirror, photos = (tensor[index].cuda() for tensor, index in zip(images, (qi, qi, pi), strict=True))
    gallery_words = {
        "sketch_only": word_bank(photos, photo_ensemble, scale),
        "both_single": word_bank(photos, single_photo, scale),
    }
    labels_cuda = photo_labels.cuda()
    replay = PrefixReplay(kd, photo_labels, split)
    rows = {name: [] for name in gallery_words}
    workers = min(4, len(os.sched_getaffinity(0)))
    print(split, "ranking", len(qi), "sketches against", len(pi), "photos", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(qi), BATCH):
            batch = slice(start, min(start + BATCH, len(qi)))
            reading = query_posteriors(query[batch], mirror[batch], single_sketch, scale)
            for name, bank in gallery_words.items():
                rows[name].append(score_rows(reading @ bank, labels[batch], labels_cuda, split, replay, pool)[:, :2])
            if start % (20 * BATCH) == 0 or batch.stop == len(qi):
                print(split, batch.stop, "/", len(qi), flush=True)

    metrics = {"two_each": baseline, **{name: torch.cat(parts).double() for name, parts in rows.items()}}
    if any(value.shape != (len(qi), 2) or not value.isfinite().all() for value in metrics.values()):
        raise ValueError(f"Incomplete or nonfinite metrics for {split}")
    deltas = {}
    for name in rows:
        diff = 100 * (metrics[name] - baseline)
        deltas[name] = {"points": diff.mean(0).tolist(), "macro_ap": class_interval(diff[:, 0], labels, seed=SEED)}
    save_tensor({"qi": qi, "pi": pi, "labels": labels, "rows": metrics, "recipe": recipe}, archive)
    result = {
        "split": split, "queries": len(qi), "gallery": len(pi), "classes": len(labels.unique()),
        "cutoffs": dict(zip(("mAP", "precision"), replay.cutoffs, strict=True)),
        "metrics": {name: (100 * value.mean(0)).tolist() for name, value in metrics.items()},
        "versus_ensemble": deltas, "recipe": recipe, "commit": lock["commit"], "device": device,
        "torch": str(torch.__version__), "seed": SEED, "freeze_sha256": lock["sha256"],
        "per_query_sha256": checksum(archive),
    }
    atomic_json(report, result)
    print(split, result["metrics"], flush=True)


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
    prior = json.loads((REFERENCE / "freeze.json").read_bytes())
    prior_hash = prior.pop("sha256")
    if digest(prior) != prior_hash:
        raise ValueError("The boxable word list has changed")
    boxable = prior["subsets"]["boxable"]
    if len(boxable["words"]) != 601:
        raise ValueError("Expected the locked 601 boxable words")
    protocol = {
        "commit": revision(), "seed": SEED, "clip_sha": CHECKPOINT, "evaluator_sha": KD_FUNCTION_SHA,
        "vocabulary_count": 601, "vocabulary_lock_sha256": prior_hash,
        "reference_commit": prior["commit"], "reference_archive_sha256": checksum(REFERENCE / "paired-T4.zip"),
        "prompts": {
            "two_each": [list(group) for group in DEFAULT_DOMAIN_TEMPLATES],
            "sketch_only": [[SKETCH_PROMPT], list(DEFAULT_DOMAIN_TEMPLATES[1])],
            "both_single": [[SKETCH_PROMPT], [PHOTO_PROMPT]],
        },
    }
    lock = protocol | {"sha256": digest(protocol)}
    freeze = root / "freeze.json"
    if freeze.is_file() and json.loads(freeze.read_bytes()) != lock:
        raise ValueError("The prompt protocol differs from the previous run")
    atomic_json(freeze, lock)
    kd, _ = load_kd(workspace)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("KD-SBIR evaluator has changed")

    cfg = runtime(workspace, "sketchy")
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    single_sketch = encode_prompt(model, tokenizer, boxable["words"], SKETCH_PROMPT)
    single_photo = encode_prompt(model, tokenizer, boxable["words"], PHOTO_PROMPT)
    del model
    ids = torch.tensor(boxable["indices"], device="cuda")
    for dataset in ("sketchy", "tuberlin"):
        qs, gs, q, p, mirror, _, target, scale, vocab = inputs(workspace, dataset)
        if vocab != prior["full_vocab_sha"]:
            raise ValueError("The prompt banks do not use the same vocabulary")
        photo_ensemble = target.index_select(0, ids)
        for split, queries in qs.items():
            score_split(root, split, queries, gs[split], (q, mirror, p), single_sketch,
                        photo_ensemble, single_photo, scale, lock, kd)


if __name__ == "__main__":
    run()
