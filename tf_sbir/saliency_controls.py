"""Control crop position with the same extent and weights, and evaluate the frozen full-frame controls."""

import argparse
import json
from pathlib import Path

import torch

from .benchmark import merge_inputs
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT, runtime
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .encoding import load_global_inputs
from .foundation import load_words
from .saliency_benchmark import LOCK, checked_row, folder
from .saliency_transfer import Setting, evaluate_split
from .saliency_views import bank
from .semantic import read_words

SELECTION_REVISION = "5b0f2b4614d6a4bd9b0f251f13d03a95780dc2cb"


@torch.inference_mode()
def run(workspace, dataset, at_center):
    held = dataset == "tuberlin"
    if held and not at_center:
        raise ValueError("Full-frame and shuffled TU controls are already part of the primary study")
    lock = json.loads((workspace / "results" / LOCK).read_text())
    data = lock.copy()
    sig = data.pop("freeze_sha256")
    if (
        digest(data) != sig
        or lock["revision"] != SELECTION_REVISION
        or lock["tu_split_sha256"] != SPLIT_HASHES["tuberlin"]
        or lock["selection_splits"] != list(SKETCHY_SPLITS)
    ):
        raise ValueError("Controls need a valid frozen Sketchy selection")
    splits = ("tuberlin",) if held else SKETCHY_SPLITS
    names = lock["winners"].values() if at_center else lock["controls"].values()
    args = runtime(workspace, dataset, batch=64, workers=2, crop_batch=192)
    args.score_batch = 192
    args.crop_workers = 6
    out = "saliency-" + dataset + ("-center" if at_center else "-controls")
    args.output = workspace / "results" / out
    args.output.mkdir(exist_ok=True, parents=True)
    qp, qi = merge_inputs(args.root, splits, "sketch")
    pp, pi = merge_inputs(args.root, splits, "photo")
    qs, photos, _ = load_global_inputs(args, qp, pp, False)
    qv = bank(args, qp, "sketch", at_center=at_center)
    pv = bank(args, pp, "photo", at_center=at_center)
    words = read_words(args.vocabulary)
    texts = tuple(t.cuda() for t in load_words(args, words))
    for split in splits:
        q, ql = qi[split]
        p, pl = pi[split]
        if digest(UNSEEN_CLASSES[split]) != SPLIT_HASHES[split]:
            raise ValueError(split)
        if held and (len(set(ql.tolist())), len(set(pl.tolist()))) != (30, 30):
            raise ValueError("Full 30-class TU required")
        catalog = {}
        for n in names:
            v = lock["heldout"][n]
            c, mode = Setting(**v["config"]), v["mode"]
            full = n if not at_center else "center-" + n
            if not at_center and (folder(workspace, False) / (split + "-" + n + ".json")).exists():
                continue
            if (args.output / (split + "-" + full + ".json")).exists():
                checked_row(args.output, split, full, c, mode, sig if held else None)
            else:
                catalog[full] = (c, mode)
        if not catalog:
            continue
        args.metric_workers = 2 if split == "sketchy_2" else 4
        meta = {
            "backbone": "OpenAI CLIP ViT-B/32",
            "checkpoint_sha256": CHECKPOINT,
            "gpu": torch.cuda.get_device_name(),
            "torch": str(torch.__version__),
            "revision": revision(),
            "selection_revision": SELECTION_REVISION,
            "exploratory_unseen": not held,
            "public_terms": len(words),
            "study": "saliency-location-control",
            "region_location": "image-center" if at_center else "saliency",
            "freeze_sha256": sig if held else None,
            "classes": len(UNSEEN_CLASSES[split]),
        }
        evaluate_split(
            args,
            split,
            qs[q].cuda(),
            photos[p].cuda(),
            qv[q].cuda(),
            pv[p].cuda(),
            texts,
            catalog,
            (ql, pl),
            meta,
        )
        for name, (c, mode) in catalog.items():
            checked_row(args.output, split, name, c, mode, sig if held else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--dataset", choices=("sketchy", "tuberlin"), required=True)
    parser.add_argument("--at-center", action="store_true")
    opts = parser.parse_args()
    run(opts.workspace, opts.dataset, opts.at_center)
