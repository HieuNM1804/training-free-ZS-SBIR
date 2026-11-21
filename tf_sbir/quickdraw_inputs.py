"""Load the cached, full 30-class QuickDraw inputs on the Colab GPU."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES, collect_paths


def paths(workspace, batch=448, workers=10):
    if not Path("/content").is_dir():
        raise RuntimeError("Run this evaluation in Google Colab")
    manifest = json.loads((workspace / "quickdraw-data.json").read_text())
    classes = UNSEEN_CLASSES["quickdraw"]
    if manifest["classes_sha256"] != digest(classes):
        raise ValueError("QuickDraw class list differs from the prepared dataset")

    checkpoint = workspace / "assets" / "ViT-B-32.pt"
    with checkpoint.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != CHECKPOINT:
            raise ValueError("Unexpected OpenAI CLIP ViT-B/32 checkpoint")

    args = SimpleNamespace(
        root=Path(manifest["root"]),
        cache=workspace / "features",
        clip_checkpoint=str(checkpoint),
        device="cuda",
        workers=workers,
        batch_size=batch,
        memory_map=True,
        vocabulary=workspace / "assets" / "open-images-v7-classes.csv",
    )
    queries, query_labels = collect_paths(args.root, classes, "sketch")
    photos, photo_labels = collect_paths(args.root, classes, "photo")
    for mode, labels in (("sketch", query_labels), ("photo", photo_labels)):
        counts = [manifest["counts"][mode][category] for category in classes]
        if labels.bincount().tolist() != counts:
            raise ValueError("The cached QuickDraw files differ from the prepared split")
    return args, queries, query_labels, photos, photo_labels
