"""Measure what existing gallery-neighbor averaging does to category and subject evidence."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import normalize
from torch.utils.data import DataLoader

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .conditional_completion_study import evaluate
from .crossdomain_inputs import CHECKPOINT
from .data import CLIP_MEAN, CLIP_STD, SKETCHY_SPLITS, configure_image_worker
from .encoding import load_frozen_clip, save_tensor
from .gallery_diffusion import nearest_gallery
from .kaggle_unseen_features import DEFAULT_CACHE, index_file, load_split
from .native_block_probe import SEED, deterministic
from .native_choice_observations import PhotoPairs
from .profiled_study import true_sources
from .replicated_simplex_study import evaluator
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_disputed_word_risk import query_posteriors, word_bank
from .semantic_operator_audit import K, SPLITS


OUT = DEFAULT_CACHE.parent / "results" / "semantic-donor-specificity"


def donor_folds(dataset, ids):
    bits = [hashlib.sha256(f"{SEED}/{dataset}/{i}".encode()).digest()[0] % 2 for i in ids.tolist()]
    return torch.tensor(bits)


def restricted_neighbors(photos, pool):
    references = photos[pool].T.contiguous()
    rows = []
    for ids in torch.arange(len(photos), device=pool.device).split(384):
        scores = photos[ids] @ references
        scores.masked_fill_(ids[:, None] == pool[None, :], -torch.inf)
        rows.append(pool[scores.topk(256, dim=1).indices[:, :K]])
    return torch.cat(rows)


@torch.inference_mode()
def prepare(split_name, revision):
    split, _, _ = load_split(DEFAULT_CACHE, split_name)
    dataset = "sketchy" if split_name in SKETCHY_SPLITS else "tuberlin"
    manifest = json.loads((DEFAULT_CACHE / "manifest.json").read_bytes())
    if manifest["sha256"] != CHECKPOINT:
        raise ValueError("Unexpected CLIP checkpoint")
    record = manifest["datasets"][dataset]
    order = torch.load(index_file(DEFAULT_CACHE, dataset, record), weights_only=True)
    names = [order["photo"]["paths"][i] for i in split.gallery_ids.tolist()]
    loader = DataLoader(PhotoPairs(Path(record["root"]), names), batch_size=384, shuffle=False,
                        num_workers=9, prefetch_factor=2, pin_memory=True, worker_init_fn=configure_image_worker,
                        generator=torch.Generator().manual_seed(SEED))
    model = load_frozen_clip(manifest["checkpoint"], torch.device("cuda"))
    mean = torch.tensor(CLIP_MEAN, device="cuda")[None, :, None, None]
    std = torch.tensor(CLIP_STD, device="cuda")[None, :, None, None]
    gray, replay, start = [], [], 0
    for color, mono in loader:
        # Eight original photos in every batch check path order, preprocessing and the existing cache.
        n = min(8, len(color))
        image = torch.cat((color[:n].cuda(non_blocking=True), mono.cuda(non_blocking=True).expand(-1, 3, -1, -1)))
        pixels = image.float().div_(255).sub_(mean).div_(std).to(model.visual.conv1.weight.dtype)
        features = normalize(model.encode_image(pixels).float(), dim=1)
        replay.append((features[:n] * normalize(split.photo[start:start+n].float(), dim=1)).sum(1).min().item())
        gray.append(features[n:].cpu())
        start += len(color)
    if start != len(names) or min(replay) < .999:
        raise ValueError("Gray-image source indexing or original RGB replay mismatch")
    photos = normalize(split.photo.float(), dim=1)
    folds = donor_folds(dataset, split.gallery_ids)
    ids, _ = nearest_gallery(photos, 256)
    neighbors = {"full64": ids[:, :K].contiguous().cpu()}
    for f in (0, 1):
        pool = torch.where(folds != f)[0].cuda()
        ix = restricted_neighbors(photos, pool).cpu()
        if (folds[ix] == f).any():
            raise ValueError("Withheld photo was used as a donor")
        neighbors[f"without_{f}"] = ix
    for ix in neighbors.values():
        if (ix == torch.arange(len(names))[:, None]).any() or (ix.sort(1).values.diff(dim=1) == 0).any():
            raise ValueError("A donor list includes repeated photos or the candidate itself")
    path = OUT / f"input-{split_name}.pt"
    save_tensor({"gallery_ids": split.gallery_ids, "gray": torch.cat(gray), "folds": folds, "neighbors": neighbors}, path)
    atomic_json(path.with_suffix(".json"), {"split": split_name, "revision": revision, "seed": SEED,
                "gray_images": start, "minimum_replay_cosine": min(replay), "donor_groups": folds.bincount().tolist(),
                "sha256": checksum(path), "device": torch.cuda.get_device_name(), "checkpoint": CHECKPOINT})
    print(split_name, "gray and donor identities verified", start, flush=True)


def average_donors(p, ix):
    return torch.cat([p[rows].double().mean(1).float() for rows in ix.split(96)])


def identity_ranks(q, bank, source, classes):
    gallery = bank.T.contiguous()
    harmonic = torch.cat((torch.zeros(1, device=q.device, dtype=torch.float64),
                          torch.arange(1, len(bank)+1, device=q.device, dtype=torch.float64).reciprocal().cumsum(0)))
    rows = []
    for qs, ids in zip(q.split(96), source.split(96), strict=True):
        scores = qs @ gallery
        own = scores.gather(1, ids[:, None])
        same = classes[None, :] == classes[ids, None]
        better = ((scores > own) & same).sum(1)
        tied = ((scores == own) & same).sum(1)
        if (tied < 1).any():
            raise ValueError("Each exact source must tie itself")
        r1 = (better == 0).double() / tied
        rr = (harmonic[better + tied] - harmonic[better]) / tied
        rows.append(torch.stack((r1, rr), dim=1).cpu())
    return torch.cat(rows)


def slot_expectation(p, donors, all_labels):
    """Expectation under within-block permutation of donor edge slots, conditional on class counts."""
    labels = all_labels[donors]
    used, frequencies = donors.flatten().unique(return_counts=True)
    expected = torch.zeros(len(donors), p.shape[1], device=p.device, dtype=torch.float64)
    for cls in all_labels[used].unique().tolist():
        choose = all_labels[used] == cls
        weights = frequencies[choose].double()
        mean = (weights / weights.sum()) @ p[used[choose]].double()
        fraction = (labels == cls).double().mean(1)
        expected += fraction[:, None] * mean
    return expected


def pair_accuracy(q, b, group, weights):
    panel = (q @ b.T)[:, group]
    difference = panel.diag()[:, None] - panel
    exchange = difference + difference.T
    valid = group[:, None] != group[None, :]
    pair_weight = weights[:, None] * weights[None, :] * valid
    return ((exchange.gt(0).double() + .5 * exchange.eq(0)) * pair_weight).sum().item() / pair_weight.sum().item()


def subject_diagnostics(query, source, p, indices, banks, folds, gallery_labels):
    source = source.cuda()
    groups = folds[source]
    classes = gallery_labels[source]
    raw = identity_ranks(query, p, source, gallery_labels)
    all_neighbors = identity_ranks(query, banks["full64"], source, gallery_labels)
    held = torch.zeros_like(raw)
    paired = []
    for fold in (0, 1):
        key = f"without_{fold}"
        take = groups == fold
        held[take.cpu()] = identity_ranks(query[take], banks[key], source[take], gallery_labels)
        for cls in classes[take].unique().tolist():
            pick = take & (classes == cls)
            ids, group, number = source[pick].unique(return_inverse=True, return_counts=True)
            if len(ids) < 2:
                raise ValueError("At least two subjects per class and fold are required")
            weights = number[group].double().reciprocal()
            q = query[pick].double()
            # Accumulate independently annotated drawings before giving each subject equal influence.
            by_id = torch.zeros(len(ids), q.shape[1], device=q.device, dtype=q.dtype)
            by_id.index_add_(0, group, q * weights[:, None])
            own_minus_other = (by_id - by_id.mean(0)) * (len(ids) / (len(ids)-1))
            slots = indices[key][ids]
            # All candidate subjects in this panel are in the withheld fold, including the rival in every pair.
            if torch.isin(slots, ids).any():
                raise ValueError("Subject identity leaked into the donor reference pool")
            ref = slot_expectation(p, slots, gallery_labels)
            real = banks[key][ids].double()
            conserved = ((ref - real).sum(0).abs() / len(ids)).max().item()
            if conserved > 2e-7:
                raise ValueError(f"Donor slot total failed to replay: {conserved}")
            base = p[ids].double()
            linear = {name: (own_minus_other * field).sum(1).mean().item()
                      for name, field in (("native", base), ("withheld64", real), ("class_slots", ref))}
            paired.append({"class": cls, "fold": fold, "queries": len(q), "subjects": len(ids),
                           "possible_source_pairs": len(ids)*(len(ids)-1)//2, "linear_contrast": linear,
                           "two_choice": {"native": pair_accuracy(q, base, group, weights),
                                          "withheld64": pair_accuracy(q, real, group, weights)},
                           "slot_conservation_max_abs": conserved})
    return {"native": raw, "full64": all_neighbors, "withheld64": held}, paired


def boot(values):
    x = np.asarray(values, dtype=np.float64)
    draws = np.random.default_rng(SEED).integers(0, len(x), size=(10000, len(x)))
    return {"mean": x.mean(0).tolist(), "class_bootstrap95": np.quantile(x[draws].mean(1), [.025, .975], axis=0).tolist()}


def summarize_subjects(metrics, pairs, sources, labels):
    subjects, group, counts = sources.unique(return_counts=True, return_inverse=True)
    weights = counts[group].double().reciprocal()
    classes = labels.unique().tolist()
    means = {}
    for key, matrix in metrics.items():
        means[key] = np.array([((matrix[labels == c] * weights[labels == c, None]).sum(0)
                                / weights[labels == c].sum()).mul(100).tolist() for c in classes])
    outputs = {"queries": len(labels), "subjects": len(subjects), "metrics_R1_MRR": {k: boot(v) for k, v in means.items()},
               "withheld64_minus_native_R1_MRR": boot(means["withheld64"] - means["native"])}
    by_stat = {}
    for key in ("two_choice", "linear_contrast"):
        rows = {}
        for variant in pairs[0][key]:
            per_class = []
            for cls in classes:
                folds = [r for r in pairs if r["class"] == cls]
                per_class.append(np.average([r[key][variant] for r in folds],
                                            weights=[r["possible_source_pairs"] for r in folds]))
            rows[variant] = np.asarray(per_class) * (100 if key == "two_choice" else 1)
        data = {k: boot(v) for k, v in rows.items()}
        data["withheld64_minus_native"] = boot(rows["withheld64"] - rows["native"])
        if "class_slots" in rows:
            data["withheld64_minus_class_slots"] = boot(rows["withheld64"] - rows["class_slots"])
        by_stat[key] = data
    outputs.update(by_stat)
    return outputs


@torch.inference_mode()
def measure(split_name, readout, revision):
    split, text, _ = load_split(DEFAULT_CACHE, split_name)
    path = OUT / f"input-{split_name}.pt"
    manifest = json.loads(path.with_suffix(".json").read_bytes())
    if checksum(path) != manifest["sha256"]:
        raise ValueError("Input archive checksum mismatch")
    inputs = torch.load(path, weights_only=True, mmap=True)
    if not torch.equal(inputs["gallery_ids"], split.gallery_ids):
        raise ValueError("Gallery order differs from the donor cache")
    folds = inputs["folds"].cuda()
    ix = {k: v.cuda() for k, v in inputs["neighbors"].items()}
    photos = normalize(split.photo.float(), dim=1)
    gray_rgb = inputs["gray"].cuda()
    if readout == "semantic":
        p = word_bank(photos, text[DEFAULT_PROMPTS[1]], split.scale).T.contiguous().double()
        p = (p / p.sum(1, keepdim=True)).float()
        q = torch.cat([query_posteriors(a, b, text[DEFAULT_PROMPTS[0]], split.scale)
                       for a, b in zip(split.sketch.split(384), split.mirror.split(384), strict=True)])
        gray = word_bank(gray_rgb, text[DEFAULT_PROMPTS[1]], split.scale).T.contiguous()
    else:
        p = photos
        q = normalize(normalize(split.sketch.float(), dim=1) + normalize(split.mirror.float(), dim=1), dim=1)
        gray = normalize(gray_rgb.float(), dim=1)
    banks = {k: average_donors(p, j) for k, j in ix.items()}
    source = true_sources(split, DEFAULT_CACHE).cpu() if split_name in SKETCHY_SPLITS else None
    if source is not None and ((source < 0).any() or not torch.equal(split.photo_labels[source], split.sketch_labels)):
        raise ValueError("Unmatched human drawing/source identity")
    kd = evaluator(DEFAULT_CACHE.parent / "real-sketchy-smoke/evaluator.py")
    replay = PrefixReplay(kd, split.photo_labels, split.name)
    category = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for key, bank in (("native", p), ("full64", banks["full64"])):
            category[key] = evaluate(split, q, bank, replay, pool)[:, :2].double()
        held = torch.zeros_like(category["native"])
        for f in (0, 1):
            active = torch.arange(len(q)) if source is None else torch.where(inputs["folds"][source] == f)[0]
            sample = replace(split, sketch_labels=split.sketch_labels[active])
            result = evaluate(sample, q[active], banks[f"without_{f}"], replay, pool)[:, :2].double()
            if source is None:
                held += result / 2
            else:
                held[active] = result
        category["withheld64"] = held
    output, raw = {}, {"category": category, "query_ids": split.query_ids, "gallery_ids": split.gallery_ids}
    labels = split.photo_labels.cuda()
    tests = [("gray_photo", gray, torch.arange(len(p)))]
    if source is not None:
        tests.append(("human_sketch", q, source))
    for kind, queries, source_rows in tests:
        values, pairs = subject_diagnostics(queries, source_rows, p, ix, banks, folds, labels)
        output[kind] = summarize_subjects(values, pairs, source_rows, split.photo_labels[source_rows])
        raw[kind] = {"sources": source_rows, "rows": values, "class_fold_summaries": pairs}
    change = 100 * (category["withheld64"] - category["native"])
    by_class = [change[split.sketch_labels == c].mean(0).tolist() for c in split.sketch_labels.unique()]
    official = {k: (100*v.mean(0)).tolist() for k, v in category.items()}
    if readout == "semantic":
        benchmark_path = Path(__file__).resolve().parents[1] / "docs/results/semantic-operator-identifiability/operator-summary.json"
        archived = json.loads(benchmark_path.read_bytes())["splits"][split_name]["metrics"]
        discrepancy = max(abs(x-y) for key, old in (("native","dot"),("full64","donor64"))
                          for x,y in zip(official[key], archived[old], strict=True))
        if discrepancy > .025:
            raise ValueError(f"Original full semantic benchmark not reproduced: {discrepancy}")
    dest = OUT / f"{split_name}-{readout}.pt"
    save_tensor(raw, dest)
    atomic_json(dest.with_suffix(".json"), {"split": split_name, "reader": readout, "revision": revision,
                "input_revision": manifest["revision"], "input_sha256": manifest["sha256"], "seed": SEED, "k": K,
                "queries": len(q), "gallery": len(p), "classes": int(labels.max()) + 1, "category_mAP_P": official,
                "withheld_category_change_macro": boot(by_class), "subject": output, "per_query_sha256": checksum(dest),
                "device": torch.cuda.get_device_name()})
    print(split_name, readout, official, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "measure"))
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--reader", choices=("semantic", "visual"))
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    deterministic()
    if args.stage == "prepare":
        prepare(args.split, args.revision)
    elif args.reader:
        measure(args.split, args.reader, args.revision)
    else:
        parser.error("A reader is required for the measurement stage")
