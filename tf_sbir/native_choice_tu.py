"""Transfer the locked two-choice test to the full unpaired TU-Berlin data."""

import argparse

import torch
from torch.nn.functional import normalize

from .concept_benchmark import atomic_json, checksum
from .encoding import save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE, load_split
from .native_block_probe import SEED, deterministic
from .native_choice_observations import OUT
from .native_choice_pairs import grouped_summary, matched_choices
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_disputed_word_risk import query_posteriors, word_bank


@torch.inference_mode()
def run(revision):
    split, text, _ = load_split(DEFAULT_CACHE, "tuberlin")
    labels = split.sketch_labels.cuda()
    classes = split.photo_labels.cuda()
    query = normalize(normalize(split.sketch.float(), dim=1) + normalize(split.mirror.float(), dim=1), dim=1)
    photos = normalize(split.photo.float(), dim=1)
    paths, margins = [], []
    for part, tag in zip(query.split(128), labels.split(128), strict=True):
        scores = part @ photos.T
        same = tag[:, None] == classes[None, :]
        positive, a = scores.masked_fill(~same, -torch.inf).max(1)
        negative, b = scores.masked_fill(same, -torch.inf).max(1)
        paths.append(torch.stack((a, b), dim=1))
        margins.append(positive - negative)
    ep = torch.cat(paths)
    own = torch.cat(margins).cpu()
    # Unpaired data have no ground-truth source-photo identity to exclude.
    units = torch.arange(len(query), device=query.device) + len(photos)
    allow = torch.ones(len(classes.unique()), device=query.device, dtype=torch.bool)
    word_q = torch.cat([query_posteriors(a, b, text[DEFAULT_PROMPTS[0]], split.scale)
                        for a, b in zip(split.sketch.split(384), split.mirror.split(384), strict=True)])
    word_p = word_bank(split.photo, text[DEFAULT_PROMPTS[1]], split.scale).T
    sums, per_query = {}, {}
    for name, q, p in (("native", query, photos), ("semantic", word_q, word_p)):
        rows = matched_choices(q.double(), p.double(), ep, units, labels, classes, allow)
        if name == "native" and (rows["own"] - own).abs().max() > 2e-5:
            raise ValueError("Native TU leaders did not replay")
        per_query[name] = rows
        metrics = {"exchange": rows["rival_exchange"], "foil_direct": rows["rival_direct"],
                   "unrelated": rows["unrelated_all"], "exchange_minus_unrelated": rows["excess_all"],
                   "both_direct": rows["both_direct"]}
        sums[name] = {k: grouped_summary(metrics, keep, labels.cpu(), units.cpu()) for k, keep in
                      (("native_wrong", own < 0), ("native_right", own > 0), ("all", torch.ones_like(own, dtype=torch.bool)))}
    dst = OUT / "choices-tuberlin.pt"
    save_tensor({"queries": split.query_ids, "gallery_ids": split.gallery_ids, "endpoints": ep.cpu(),
                 "native_wrong": own < 0, "rows": per_query}, dst)
    atomic_json(dst.with_suffix(".json"), {"revision": revision, "seed": SEED, "reads": sums, "queries": len(query),
                "gallery": len(photos), "classes": len(allow), "native_wrong": int((own < 0).sum()),
                "source_policy": "unpaired, equal query weight within class", "device": torch.cuda.get_device_name(),
                "sha256": checksum(dst)})
    print("TU paired-choice check complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    deterministic()
    run(parser.parse_args().revision)
