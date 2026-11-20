"""Frozen-input preparation for a Sketchy-selected, TU-Berlin-held-out experiment."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from .benchmark import merge_inputs
from .data import SKETCHY_SPLITS
from .encoding import load_global_inputs, save_tensor
from .isoclip import isoclip_maps
from .mta import load_view_features


CHECKPOINT = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"


class FrameViews(Dataset):
    """One full-frame reflection and two standard central crops in both orientations."""
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as src:
            image = src.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        frames = [ImageOps.mirror(image)]
        for side in (180, 144):
            margin = (224 - side) // 2
            crop = image.crop((margin, margin, margin + side, margin + side)).resize((224, 224), Image.Resampling.BILINEAR)
            frames.extend((crop, ImageOps.mirror(crop)))
        return torch.stack([pil_to_tensor(view) for view in frames])


def runtime(workspace, dataset, batch=192, workers=6, crop_batch=384, metric_workers=4):
    if sys.platform != "linux" or not Path("/content").is_dir() or not torch.cuda.is_available():
        raise RuntimeError("Run this experiment on the connected Google Colab GPU")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    data = "ready.json" if dataset == "sketchy" else "tuberlin-data.json"
    root = Path(json.loads((workspace / data).read_text())["root"])
    cache = workspace / "features"
    checkpoint = str(workspace / "assets/ViT-B-32.pt")
    with open(checkpoint, "rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != CHECKPOINT:
            raise ValueError("Unexpected OpenAI CLIP ViT-B/32 weights")
    return SimpleNamespace(root=root, cache=cache, clip_checkpoint=checkpoint, device="cuda", workers=workers, batch_size=batch,
                           crop_batch_size=crop_batch, crop_workers=workers, metric_workers=metric_workers,
                           score_batch=192, memory_map=True, vocabulary=workspace / "assets/open-images-v7-classes.csv")


def projection(args):
    gpu = torch.cuda.get_device_name().replace(" ", "-")
    file = args.cache / f"iso-b32-150-50-{CHECKPOINT[:12]}-{torch.__version__}-{gpu}.pt"
    if not file.is_file():
        save_tensor(isoclip_maps(args, torch.device("cuda"), bands=((150, 50),))[(150, 50)].cpu(), file)
    return torch.load(file, map_location="cuda", weights_only=True)


def inputs(args, dataset, *, prepare_sketch_controls=False):
    splits = SKETCHY_SPLITS if dataset == "sketchy" else ("tuberlin",)
    q_paths, queries = merge_inputs(args.root, splits, "sketch")
    p_paths, gallery = merge_inputs(args.root, splits, "photo")
    raw, photos, contours = load_global_inputs(args, q_paths, p_paths, prepare_sketch_controls)
    frames = load_view_features(args, FrameViews(q_paths), 5, "sketch", "neutral-frame-reflect-center180-144-v1")["features"]
    legacy = None
    if prepare_sketch_controls:
        from .reflection_transfer import load_reflected_views
        from .transfer_benchmark import AnchorSketchViews
        ink = load_view_features(args, AnchorSketchViews(q_paths), 4, "sketch", "ink-scales-mirror-v1")["features"]
        legacy = (contours, ink, load_reflected_views(args, q_paths))
    return queries, gallery, raw, photos, frames, legacy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sketchy", "tuberlin"), required=True)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--prepare-sketch-controls", action="store_true")
    args = parser.parse_args()
    cfg = runtime(args.workspace, args.dataset)
    with torch.inference_mode():
        projection(cfg)
        query, gallery, *_ = inputs(cfg, args.dataset, prepare_sketch_controls=args.prepare_sketch_controls)
        print(json.dumps({s: {"queries": len(q[0]), "gallery": len(gallery[s][0])} for s, q in query.items()}), flush=True)
