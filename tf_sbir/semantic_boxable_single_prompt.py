"""Compare separate and shared text prompts for S on the locked 601-word vocabulary."""

import argparse
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
SHARED_FAMILY = "semantic-boxable-shared-photo-prompt"
DOCS = Path(__file__).resolve().parents[1] / "docs/results"
VOCABULARY_STUDY = "semantic-open-images-vocabulary"
SKETCH_PROMPT = DEFAULT_DOMAIN_TEMPLATES[0][1]
PHOTO_PROMPT = DEFAULT_DOMAIN_TEMPLATES[1][1]


def encode_prompt(model, tokenizer, words, prompt):
    batches = []
    for start in range(0, len(words), 256):
        text = tokenizer([prompt.format(word) for word in words[start:start + 256]]).cuda()
        batches.append(F.normalize(model.encode_text(text).float(), dim=-1))
    return torch.cat(batches)


def previous_rows(split, qi, pi, labels, lock):
    reference = DOCS / lock["reference_study"] / "paired-T4.zip"
    with zipfile.ZipFile(reference) as archive:
        prior = torch.load(io.BytesIO(archive.read(f"{split}.pt")), weights_only=True)
    if not all(torch.equal(prior[name], value) for name, value in (("qi", qi), ("pi", pi), ("labels", labels))):
        raise ValueError(f"Query or gallery order changed for {split}")
    return {name: prior["rows"][original] for name, original in lock["reference_arms"].items()}


def score_split(root, split, queries, gallery, images, sketch_bank, photo_banks, scale, lock, kd):
    qi, labels = queries
    pi, photo_labels = gallery
    if (len(qi), len(pi), len(labels.unique())) != COUNTS[split] or len(photo_labels.unique()) != COUNTS[split][2]:
        raise ValueError(f"Unexpected unseen split {split}")
    archived = previous_rows(split, qi, pi, labels, lock)
    baseline = archived[lock["comparison_arm"]]
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
    gallery_words = {name: word_bank(photos, text, scale) for name, text in photo_banks.items()}
    labels_cuda = photo_labels.cuda()
    replay = PrefixReplay(kd, photo_labels, split)
    rows = {name: [] for name in gallery_words}
    workers = min(4, len(os.sched_getaffinity(0)))
    print(split, "ranking", len(qi), "sketches against", len(pi), "photos", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(qi), BATCH):
            batch = slice(start, min(start + BATCH, len(qi)))
            reading = query_posteriors(query[batch], mirror[batch], sketch_bank, scale)
            for name, bank in gallery_words.items():
                rows[name].append(score_rows(reading @ bank, labels[batch], labels_cuda, split, replay, pool)[:, :2])
            if start % (20 * BATCH) == 0 or batch.stop == len(qi):
                print(split, batch.stop, "/", len(qi), flush=True)

    metrics = archived | {name: torch.cat(parts).double() for name, parts in rows.items()}
    if any(value.shape != (len(qi), 2) or not value.isfinite().all() for value in metrics.values()):
        raise ValueError(f"Incomplete or nonfinite metrics for {split}")
    deltas = {}
    for name in rows:
        diff = 100 * (metrics[name] - baseline)
        deltas[name] = {"points": diff.mean(0).tolist(), "macro_ap": class_interval(diff[:, 0], labels, seed=SEED)}
    save_tensor({"qi": qi, "pi": pi, "labels": labels, "rows": metrics, "recipe": recipe}, archive)
    delta_key = "paired_delta" if lock["comparison_arm"] == "both_single" else "versus_ensemble"
    result = {
        "split": split, "queries": len(qi), "gallery": len(pi), "classes": len(labels.unique()),
        "cutoffs": dict(zip(("mAP", "precision"), replay.cutoffs, strict=True)),
        "metrics": {name: (100 * value.mean(0)).tolist() for name, value in metrics.items()},
        "comparison_arm": lock["comparison_arm"], delta_key: deltas,
        "recipe": recipe, "commit": lock["commit"], "device": device,
        "torch": str(torch.__version__), "seed": SEED, "freeze_sha256": lock["sha256"],
        "per_query_sha256": checksum(archive),
    }
    atomic_json(report, result)
    print(split, result["metrics"], flush=True)


@torch.inference_mode()
def run(shared_photo=False):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    workspace = Path("/content/sbir")
    root = workspace / "results" / (SHARED_FAMILY if shared_photo else FAMILY)
    prior = json.loads((DOCS / VOCABULARY_STUDY / "freeze.json").read_bytes())
    prior_hash = prior.pop("sha256")
    if digest(prior) != prior_hash:
        raise ValueError("The boxable word list has changed")
    boxable = prior["subsets"]["boxable"]
    if len(boxable["words"]) != 601:
        raise ValueError("Expected the locked 601 boxable words")
    reference_study = FAMILY if shared_photo else VOCABULARY_STUDY
    reference = DOCS / reference_study
    reference_lock = json.loads((reference / "freeze.json").read_bytes())
    prompts = {
        "two_each": [list(group) for group in DEFAULT_DOMAIN_TEMPLATES],
        "sketch_only": [[SKETCH_PROMPT], list(DEFAULT_DOMAIN_TEMPLATES[1])],
        "both_single": [[SKETCH_PROMPT], [PHOTO_PROMPT]],
    }
    reference_arms = {"two_each": "boxable"}
    if shared_photo:
        if (reference_lock["vocabulary_lock_sha256"] != prior_hash
                or reference_lock["prompts"] != prompts):
            raise ValueError("The archived prompt comparison must use the same words and templates")
        reference_arms = {name: name for name in prompts}
        prompts["shared_photo"] = [[PHOTO_PROMPT], [PHOTO_PROMPT]]
    protocol = {
        "commit": revision(), "seed": SEED, "clip_sha": CHECKPOINT, "evaluator_sha": KD_FUNCTION_SHA,
        "vocabulary_count": 601, "vocabulary_lock_sha256": prior_hash,
        "reference_study": reference_study, "reference_commit": reference_lock["commit"],
        "reference_archive_sha256": checksum(reference / "paired-T4.zip"),
        "reference_arms": reference_arms,
        "comparison_arm": "both_single" if shared_photo else "two_each",
        "prompts": prompts,
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
    single_photo = encode_prompt(model, tokenizer, boxable["words"], PHOTO_PROMPT)
    sketch_bank = single_photo if shared_photo else encode_prompt(model, tokenizer, boxable["words"], SKETCH_PROMPT)
    del model
    ids = torch.tensor(boxable["indices"], device="cuda")
    for dataset in ("sketchy", "tuberlin"):
        qs, gs, q, p, mirror, _, target, scale, vocab = inputs(workspace, dataset)
        if vocab != prior["full_vocab_sha"]:
            raise ValueError("The prompt banks do not use the same vocabulary")
        if shared_photo:
            photo_banks = {"shared_photo": single_photo}
        else:
            photo_banks = {"sketch_only": target.index_select(0, ids), "both_single": single_photo}
        for split, queries in qs.items():
            score_split(root, split, queries, gs[split], (q, mirror, p), sketch_bank, photo_banks, scale, lock, kd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-photo", action="store_true",
                        help="Use the single photo prompt bank for both sketch and photo")
    run(shared_photo=parser.parse_args().shared_photo)
