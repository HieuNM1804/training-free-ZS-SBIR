"""Read an unmodified CLIP visual block on the registered photo and human-redraw cohort."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .concept_benchmark import atomic_json, checksum
from .data import SKETCHY_SPLITS
from .encoding import load_frozen_clip, save_tensor
from .paired_patch_interaction_probe import DST as PAIRED, SEED, data_loader
from .kaggle_unseen_features import DEFAULT_CACHE


DST = PAIRED.parent / "native-block-depiction"


def deterministic():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def read_block(visual, pixels):
    transformer = visual.transformer
    tokens = visual._embeds(pixels)
    x = tokens if transformer.batch_first else tokens.transpose(0, 1).contiguous()
    for block in transformer.resblocks[:5]:
        x = block(x)
    y = transformer.resblocks[5](x)
    if not transformer.batch_first:
        x, y = x.transpose(0, 1), y.transpose(0, 1)
    if x.shape[1:] != (50, 768):
        raise ValueError(f"Unexpected CLIP token shape {x.shape}")
    return x, y - x


def check_native_path(visual, pixels):
    observed = []

    def hook(block, inputs, output):
        observed.extend((inputs[0], output))

    handle = visual.transformer.resblocks[5].register_forward_hook(hook)
    try:
        visual(pixels)
    finally:
        handle.remove()
    x, delta = read_block(visual, pixels)
    before, after = observed
    if not visual.transformer.batch_first:
        before, after = before.transpose(0, 1), after.transpose(0, 1)
    error = max((x - before).abs().max().item(), (delta - (after - before)).abs().max().item())
    if error > 2e-5:
        raise ValueError(f"Native forward discrepancy {error}")
    return error


@torch.inference_mode()
def extract(args):
    manifest = json.loads((DEFAULT_CACHE / "manifest.json").read_bytes())
    plan = json.loads((PAIRED / f"{args.split}-0.json").read_bytes())["plan"]
    if plan["split"] != args.split or plan["seed"] != SEED:
        raise ValueError("Cohort identity mismatch")
    classes = sorted(plan["classes"])[args.part::2]
    subjects = [row for name in classes for row in plan["classes"][name]["records"]]
    if len(subjects) != 32 * len(classes) or any(len(row) != 3 for row in subjects):
        raise ValueError("Expected one photo and two drawings of each subject")
    root = Path(manifest["datasets"]["sketchy"]["root"])
    paths = [str(root / p) for row in subjects for p in row]
    loader = data_loader(paths, args.workers, args.batch)
    model = load_frozen_clip(manifest["checkpoint"], torch.device("cuda")).float().eval()
    inputs, updates = [], []
    start = time.monotonic()
    for index, pixels in enumerate(loader):
        images = pixels.cuda(non_blocking=True)
        if index == 0:
            error = check_native_path(model.visual, images[:2])
        x, delta = read_block(model.visual, images)
        inputs.append(x.cpu())
        updates.append(delta.cpu())
    dst = args.output / f"{args.split}-{args.part}-states.pt"
    save_tensor({"classes": classes, "subjects": subjects, "revision": args.revision,
                 "x": torch.cat(inputs).reshape(len(classes), 32, 3, 50, 768),
                 "delta": torch.cat(updates).reshape(len(classes), 32, 3, 50, 768)}, dst)
    meta = {"split": args.split, "seed": SEED, "part": args.part, "classes": classes,
            "images": len(paths), "hook_max_abs": error, "revision": args.revision,
            "checkpoint_sha256": manifest["sha256"], "state_sha256": checksum(dst),
            "device": torch.cuda.get_device_name(), "seconds": time.monotonic() - start}
    atomic_json(dst.with_suffix(".json"), meta)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SKETCHY_SPLITS, required=True)
    parser.add_argument("--part", choices=(0, 1), type=int, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, default=DST)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    deterministic()
    extract(parser.parse_args())
