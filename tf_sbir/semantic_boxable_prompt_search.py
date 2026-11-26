"""Search one sketch and one photo template using the locked 601-word S ranker."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import inspect
import io
from itertools import product
import json
import os
from pathlib import Path
import random
import time
import zipfile

import numpy as np
import open_clip
import torch

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT, runtime
from .encoding import load_frozen_clip, save_tensor
from .geometry_word_flat import COUNTS
from .locked_geometry_words import score_rows
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_boxable_single_prompt import DOCS, PHOTO_PROMPT, SKETCH_PROMPT, VOCABULARY_STUDY, encode_prompt
from .semantic_disputed_word_risk import SEED, inputs, query_posteriors, word_bank


FAMILY = "semantic-boxable-prompt-search"
REFERENCE = DOCS / "semantic-boxable-shared-photo-prompt" / "paired-T4.zip"
SKETCHY = ("sketchy_1", "sketchy_2")
BATCH = 384
AUDIT_QUERIES = 8
MAX_METRIC_ERROR = 2e-6


def article_templates(prefixes):
    return tuple(f"{prefix} {article} {{}}." for prefix, article in product(prefixes, ("a", "the")))


SKETCH_TEMPLATES = article_templates((
    "a sketch of", "a drawing of", "a doodle of", "a line drawing of", "a pencil sketch of",
    "a hand-drawn sketch of", "a black and white drawing of", "a black and white sketch of",
    "an outline drawing of", "a simple sketch of", "an illustration of", "a photo of",
)) + ("a sketch of {}.", "a drawing of {}.")
PHOTO_TEMPLATES = article_templates((
    "a photo of", "a photograph of", "a picture of", "a close-up photo of", "a cropped photo of",
    "an image of", "a good photo of",
)) + ("a photo of one {}.", "a photo of {}.")
PAIRS = tuple(product(SKETCH_TEMPLATES, PHOTO_TEMPLATES))
REFERENCE_PAIR = PAIRS.index((SKETCH_PROMPT, PHOTO_PROMPT))
SHARED_PAIR = PAIRS.index((PHOTO_PROMPT, PHOTO_PROMPT))


@dataclass(frozen=True)
class Split:
    name: str
    query_ids: torch.Tensor
    gallery_ids: torch.Tensor
    sketch_labels: torch.Tensor
    photo_labels: torch.Tensor
    sketch: torch.Tensor
    mirror: torch.Tensor
    photo: torch.Tensor
    scale: float

    @classmethod
    def from_inputs(cls, data, name):
        qs, gs, query, gallery, mirror, _, _, scale, _ = data
        qi, ql = qs[name]
        pi, pl = gs[name]
        if (len(qi), len(pi), len(ql.unique())) != COUNTS[name] or len(pl.unique()) != COUNTS[name][2]:
            raise ValueError(f"Unexpected unseen split {name}")
        return cls(name, qi, pi, ql, pl, query[qi].cuda(), mirror[qi].cuda(), gallery[pi].cuda(), scale)


def encode_banks(workspace, words):
    cfg = runtime(workspace, "sketchy")
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    templates = dict.fromkeys((*SKETCH_TEMPLATES, *PHOTO_TEMPLATES))
    return {template: encode_prompt(model, tokenizer, words, template) for template in templates}


def posteriors(split, text, pairs):
    sketches = dict.fromkeys(sketch for sketch, _ in pairs)
    photos = dict.fromkeys(photo for _, photo in pairs)
    query_banks = {
        template: torch.cat([
            query_posteriors(split.sketch[i:i+BATCH], split.mirror[i:i+BATCH], text[template], split.scale)
            for i in range(0, len(split.query_ids), BATCH)
        ])
        for template in sketches
    }
    gallery_banks = {template: word_bank(split.photo, text[template], split.scale) for template in photos}
    return query_banks, gallery_banks


def metric_rows(scores, labels, gallery, positions, precision_k):
    """Apply the original KD-SBIR cutoffs and AP denominator to ranked hits."""
    rank = scores.topk(len(positions)).indices
    hits = gallery[rank] == labels[:, None]
    cumulative = hits.cumsum(1, dtype=torch.int32)
    ap = (hits * (cumulative / positions)).sum(1) / cumulative[:, -1].clamp_min(1)
    return torch.stack((ap, cumulative[:, precision_k - 1] / precision_k), dim=1)


def archived(split):
    with zipfile.ZipFile(REFERENCE) as archive:
        saved = torch.load(io.BytesIO(archive.read(f"{split.name}.pt")), weights_only=True)
    ordered = (("qi", split.query_ids), ("pi", split.gallery_ids), ("labels", split.sketch_labels))
    if any(not torch.equal(saved[k], v) for k, v in ordered):
        raise ValueError(f"Query or gallery order changed for {split.name}")
    return saved["rows"]


def mean_percent(rows):
    return (100 * rows.double().mean(dim=0)).tolist()


def audit_candidate(q, p, audit, split, labels, gallery, positions, replay, pool):
    scores = q[audit] @ p
    exact = score_rows(scores, split.sketch_labels[audit], gallery, split.name, replay, pool)[:, :2]
    fast = metric_rows(scores, labels[audit], gallery, positions, replay.cutoffs[1]).cpu()
    error = (exact - fast).abs().max().item()
    if error > MAX_METRIC_ERROR:
        raise ValueError(f"The vectorized KD-SBIR result differs from the original by {error}")
    return error


def sweep(root, split, banks, kd, pool, protocol_sha):
    query_banks, gallery_banks = banks
    labels, gallery = split.sketch_labels.cuda(), split.photo_labels.cuda()
    replay = PrefixReplay(kd, split.photo_labels, split.name)
    map_k, precision_k = replay.cutoffs
    positions = torch.arange(1, (map_k or len(gallery)) + 1, device="cuda")
    audit = torch.randperm(len(labels), generator=torch.Generator().manual_seed(SEED))[:AUDIT_QUERIES]
    audit = audit.sort().values
    folder = root / split.name / protocol_sha
    totals = []
    errors = []
    start_time = time.perf_counter()
    for index, sketch_template in enumerate(SKETCH_TEMPLATES):
        file = folder / f"{index:02d}.json"
        if file.exists():
            record = json.loads(file.read_bytes())
            if record["sketch_template"] != sketch_template or len(record["metrics"]) != len(PHOTO_TEMPLATES):
                raise ValueError(f"Unexpected saved grid row {file}")
        else:
            q = query_banks[sketch_template]
            metrics, max_error = [], 0.
            for photo_template in PHOTO_TEMPLATES:
                p = gallery_banks[photo_template]
                per_query = []
                for start in range(0, len(q), BATCH):
                    batch = slice(start, start + BATCH)
                    per_query.append(metric_rows(q[batch] @ p, labels[batch], gallery, positions, precision_k))
                values = torch.cat(per_query)
                metrics.append(mean_percent(values))
                max_error = max(max_error, audit_candidate(q, p, audit, split, labels, gallery, positions, replay, pool))
            record = {"sketch_template": sketch_template, "metrics": metrics, "max_kd_error": max_error}
            atomic_json(file, record)
        totals.extend(record["metrics"])
        errors.append(record["max_kd_error"])
        print(split.name, "full-unseen pairs", len(totals), "/", len(PAIRS),
              "elapsed seconds", round(time.perf_counter() - start_time), flush=True)

    prior = archived(split)
    for index, control in ((REFERENCE_PAIR, "both_single"), (SHARED_PAIR, "shared_photo")):
        if max(abs(a - b) for a, b in zip(totals[index], mean_percent(prior[control]), strict=True)) > .04:
            raise ValueError(f"The archived {control} control has changed on {split.name}")
    data = {
        "split": split.name, "protocol_sha256": protocol_sha, "commit": revision(),
        "queries": len(labels), "gallery": len(gallery), "classes": len(labels.unique()),
        "cutoffs": {"mAP": map_k, "precision": precision_k},
        "audit_query_ids": split.query_ids[audit].tolist(), "max_kd_error": max(errors),
        "metrics": totals,
    }
    atomic_json(root / f"grid-{split.name}.json", data)
    return totals


def original_rows(split, banks, candidates, kd, pool):
    query_banks, gallery_banks = banks
    gallery_labels = split.photo_labels.cuda()
    replay = PrefixReplay(kd, split.photo_labels, split.name)
    rows = {}
    for idx in candidates:
        sketch, photo = PAIRS[idx]
        source, target = query_banks[sketch], gallery_banks[photo]
        parts = []
        for start in range(0, len(source), BATCH):
            batch = slice(start, start + BATCH)
            scores = source[batch] @ target
            parts.append(score_rows(scores, split.sketch_labels[batch], gallery_labels, split.name, replay, pool)[:, :2])
        rows[idx] = torch.cat(parts).double()
        print(split.name, "original KD-SBIR", idx, (sketch, photo), mean_percent(rows[idx]), flush=True)
    return rows


def write_official(root, split, scores, protocol_sha, selection_sha=None):
    reference = archived(split)
    rows = {f"pair_{k}": v for k, v in scores.items()}
    for name in ("two_each", "both_single", "shared_photo"):
        rows[name] = reference[name]
    tensor = root / f"{split.name}.pt"
    save_tensor({"qi": split.query_ids, "pi": split.gallery_ids, "labels": split.sketch_labels, "rows": rows}, tensor)
    result = {
        "split": split.name, "commit": revision(), "protocol_sha256": protocol_sha,
        "selection_sha256": selection_sha, "device": torch.cuda.get_device_name(),
        "queries": len(split.query_ids), "gallery": len(split.gallery_ids), "classes": len(split.sketch_labels.unique()),
        "cutoffs": {"mAP": 200 if split.name == "sketchy_2" else 0,
                    "precision": 200 if split.name == "sketchy_2" else 100},
        "metrics": {name: mean_percent(values) for name, values in rows.items()},
        "per_query_sha256": checksum(tensor),
    }
    atomic_json(tensor.with_suffix(".json"), result)
    return result


def pair_key(tables, idx):
    return tuple((tables[0][idx][i] + tables[1][idx][i]) / 2 for i in (0, 1))


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
    vocabulary = json.loads((DOCS / VOCABULARY_STUDY / "freeze.json").read_bytes())
    word_sha = vocabulary.pop("sha256")
    words = vocabulary["subsets"]["boxable"]["words"]
    if len(words) != 601 or digest(vocabulary) != word_sha:
        raise ValueError("Expected the locked 601 boxable words")
    kd, _ = load_kd(workspace)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("KD-SBIR evaluator has changed")
    spec = {
        "commit": revision(), "vocabulary_lock_sha256": word_sha, "vocabulary_count": len(words),
        "clip_sha": CHECKPOINT, "evaluator_sha": KD_FUNCTION_SHA, "seed": SEED,
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "batch": BATCH,
        "reference_sha256": checksum(REFERENCE), "audit_queries_per_pair": AUDIT_QUERIES,
        "sketch_templates": list(SKETCH_TEMPLATES), "photo_templates": list(PHOTO_TEMPLATES),
        "selection": "Highest mean of the two complete unseen Sketchy mAPs; mean precision as tie-break",
    }
    protocol_sha = save_lock(root / "protocol.json", spec)
    print("Locked", len(SKETCH_TEMPLATES), "sketch prompts x", len(PHOTO_TEMPLATES), "photo prompts", flush=True)
    text = encode_banks(workspace, words)
    data = inputs(workspace, "sketchy")
    if data[-1] != vocabulary["full_vocab_sha"]:
        raise ValueError("Image and text inputs use a different vocabulary")

    workers = min(4, len(os.sched_getaffinity(0)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        grids = []
        for name in SKETCHY:
            split = Split.from_inputs(data, name)
            banks = posteriors(split, text, PAIRS)
            grids.append(sweep(root, split, banks, kd, pool, protocol_sha))
            del banks, split
        ranked = sorted(range(len(PAIRS)), key=lambda idx: pair_key(grids, idx), reverse=True)
        best_mean = pair_key(grids, ranked[0])[0]
        # Replay both leaders and any contenders within twice the maximal allowed metric error.
        shortlist = [idx for rank, idx in enumerate(ranked)
                     if rank < 2 or best_mean - pair_key(grids, idx)[0] <= 200 * MAX_METRIC_ERROR]
        print("Replaying top candidates on both full Sketchy splits", shortlist, flush=True)
        comparisons = []
        short_pairs = [PAIRS[idx] for idx in shortlist]
        for name, scan in zip(SKETCHY, grids, strict=True):
            split = Split.from_inputs(data, name)
            exact = original_rows(split, posteriors(split, text, short_pairs), shortlist, kd, pool)
            pairs = {idx: mean_percent(value) for idx, value in exact.items()}
            for idx in shortlist:
                if max(abs(a - b) for a, b in zip(pairs[idx], scan[idx], strict=True)) > 100 * MAX_METRIC_ERROR:
                    raise ValueError("Full original KD-SBIR replay disagrees with the template sweep")
            comparisons.append(pairs)
            write_official(root, split, exact, protocol_sha)
        best = max(shortlist, key=lambda idx: pair_key(comparisons, idx))
        lock = {
            "protocol_sha256": protocol_sha, "commit": spec["commit"], "winner": best,
            "sketch_prompt": PAIRS[best][0], "photo_prompt": PAIRS[best][1],
            "mean_sketchy_metrics": pair_key(comparisons, best),
            "replayed": shortlist, "pair_count": len(PAIRS),
            "sketchy_grid_sha256": {s: checksum(root / f"grid-{s}.json") for s in SKETCHY},
        }
        selection_sha = save_lock(root / "freeze.json", lock)
        print("Locked for TU-Berlin", lock, flush=True)
        del data, split

        data = inputs(workspace, "tuberlin")
        if data[-1] != vocabulary["full_vocab_sha"]:
            raise ValueError("TU-Berlin uses a different vocabulary")
        split = Split.from_inputs(data, "tuberlin")
        exact = original_rows(split, posteriors(split, text, [PAIRS[best]]), [best], kd, pool)
        write_official(root, split, exact, protocol_sha, selection_sha)


if __name__ == "__main__":
    run()
