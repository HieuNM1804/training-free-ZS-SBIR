"""Audit real hard choices under an exactly bias-cancelling exchange of two queries."""

import argparse
import json

import numpy as np
import torch
from torch.nn.functional import normalize

from .concept_benchmark import atomic_json, checksum
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .encoding import save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE, load_split
from .native_block_probe import SEED, deterministic
from .native_choice_observations import OUT
from .paired_patch_interaction_probe import DST as ENDPOINTS
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_disputed_word_risk import query_posteriors, word_bank


def interval(class_means):
    values = np.asarray(class_means, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(values), size=(10000, len(values)))
    return {"class_mean": float(values.mean()), "interval": np.quantile(values[samples].mean(1), [.025, .975]).tolist()}


def grouped_summary(values, keep, classes, sources):
    picked = torch.where(keep)[0]
    if not len(picked):
        return {"queries": 0}
    unique, group = sources[picked].unique(return_inverse=True)
    weights = 1 / group.bincount().double()[group]
    tags = classes[picked]
    measure = {}
    for key, vector in values.items():
        averaged = []
        for c in tags.unique().tolist():
            take = tags == c
            averaged.append(((vector[picked][take] * weights[take]).sum() / weights[take].sum()).item())
        measure[key] = interval(averaged)
    return {"queries": len(picked), "sources": len(unique), "classes": len(tags.unique()), "results": measure}


def matched_choices(q, p, endpoints, src, labels, gallery, unique_classes):
    weight = src.bincount()[src].double().reciprocal()
    group = torch.zeros(len(q), int(labels.max()) + 1, device=q.device, dtype=torch.float64)
    group.scatter_(1, labels[:, None], weight[:, None])
    class_totals = group.sum(0)
    all_classes = torch.ones_like(class_totals, dtype=torch.bool)
    saves = []
    for start in range(0, len(q), 96):
        stop = min(start + 96, len(q))
        sl = slice(start, stop)
        left, right = endpoints[sl].T
        axis = p[left] - p[right]
        own = (q[sl] * axis).sum(1)
        other = axis @ q.T
        crossed = (own[:, None] > other).double()
        both = (own > 0).double()
        right_class = gallery[right]
        right_mask = (labels[None, :] == right_class[:, None]) & (src[None, :] != right[:, None])
        if (labels[sl] == right_class).any() or not torch.equal(gallery[left], labels[sl]):
            raise ValueError("Selected candidates disagree with recorded classes")
        weights = right_mask * weight[None, :]
        if (weights.sum(1) <= 0).any():
            raise ValueError("No independent opposing sources")
        weights /= weights.sum(1)[:, None]
        rival_success = (weights * crossed).sum(1)
        rival_correct = (weights * (other < 0)).sum(1)
        by_class = (crossed @ group) / class_totals[None, :]
        rec = {"own": own.cpu(), "rival_exchange": rival_success.cpu(), "rival_direct": rival_correct.cpu(),
               "both_direct": (both * rival_correct).cpu()}
        for subset, allowed in (("all", all_classes), ("disjoint", unique_classes)):
            plausible = allowed.expand(len(axis), -1).clone()
            plausible.scatter_(1, labels[sl, None], False)
            plausible.scatter_(1, right_class[:, None], False)
            unrelated = (by_class * plausible).sum(1) / plausible.sum(1)
            rec[f"unrelated_{subset}"] = unrelated.cpu()
            rec[f"excess_{subset}"] = (rival_success - unrelated).cpu()
        saves.append(rec)
    return {key: torch.cat([part[key] for part in saves]) for key in saves[0]}


@torch.inference_mode()
def run(name, revision):
    path = ENDPOINTS / f"real-errors-{name}.pt"
    baseline = json.loads(path.with_suffix(".json").read_bytes())
    if checksum(path) != baseline["per_query_sha256"]:
        raise ValueError("Archived hard-choice checksum differs")
    old = torch.load(path, mmap=True, weights_only=True)
    split, text, _ = load_split(DEFAULT_CACHE, name)
    if not torch.equal(split.gallery_ids, old["gallery_ids"]) or not torch.equal(split.query_ids, old["query_ids"]):
        raise ValueError("Archived candidate or sketch order differs")
    record = old["measurements"]["native"]
    ep = record["endpoints"].cuda()
    source = old["source"].cuda()
    truth = split.sketch_labels.cuda()
    classes = split.photo_labels.cuda()
    if not torch.equal(old["labels"], split.sketch_labels) or (ep == source[:, None]).any() or not torch.equal(classes[source], truth):
        raise ValueError("Self-match or class mismatch")
    unique_names = set(UNSEEN_CLASSES[name]) - set(UNSEEN_CLASSES[SKETCHY_SPLITS[1 - SKETCHY_SPLITS.index(name)]])
    eligible = torch.tensor([c in unique_names for c in UNSEEN_CLASSES[name]], device="cuda")
    vision = normalize(normalize(split.sketch.float(), dim=1) + normalize(split.mirror.float(), dim=1), dim=1)
    photo = normalize(split.photo.float(), dim=1)
    words_q = torch.cat([query_posteriors(a, b, text[DEFAULT_PROMPTS[0]], split.scale)
                         for a, b in zip(split.sketch.split(384), split.mirror.split(384), strict=True)])
    words_p = word_bank(split.photo, text[DEFAULT_PROMPTS[1]], split.scale).T
    readers = {"native": (vision, photo), "semantic": (words_q, words_p)}
    summaries, saved, checks = {}, {}, {}
    event = record["own_selected_margin"]
    for key, (query, bank) in readers.items():
        back = matched_choices(query.double(), bank.double(), ep, source, truth, classes, eligible)
        replay = (back["own"] - record[f"{key}_margin"]).abs().max().item()
        if replay > 2e-5:
            raise ValueError(f"Cannot replay the same pair margin for {key}: {replay}")
        checks[key] = replay
        saved[key] = back
        summaries[key] = {}
        for tag, subset in (("all", torch.ones_like(event, dtype=torch.bool)),
                            ("disjoint", (eligible[truth] & eligible[classes[ep[:, 1]]]).cpu())):
            metrics = {"exchange": back["rival_exchange"], "foil_direct": back["rival_direct"],
                       "both_direct": back["both_direct"], "unrelated": back[f"unrelated_{tag}"],
                       "exchange_minus_unrelated": back[f"excess_{tag}"]}
            for state, mask in (("native_wrong", event < 0), ("native_right", event > 0), ("all", event == event)):
                summaries[key][f"{tag}/{state}"] = grouped_summary(metrics, subset & mask, truth.cpu(), source.cpu())
    output = OUT / f"choices-{name}.pt"
    save_tensor({"queries": split.query_ids, "source": old["source"], "gallery_ids": split.gallery_ids,
                 "endpoints": record["endpoints"], "native_wrong": event < 0, "rows": saved}, output)
    atomic_json(output.with_suffix(".json"), {"split": name, "revision": revision, "seed": SEED,
                "reads": summaries, "queries": len(event), "gallery": len(classes), "native_wrong": int((event < 0).sum()),
                "archived_pair_sha256": baseline["per_query_sha256"], "margin_replay_max_abs": checks,
                "sha256": checksum(output), "device": torch.cuda.get_device_name()})
    print(name, "paired choices complete", len(event), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SKETCHY_SPLITS, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    deterministic()
    run(args.split, args.revision)
