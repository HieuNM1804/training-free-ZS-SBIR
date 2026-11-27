"""Measure two-factor interactions on sampled source photos and their actual human drawings."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json, checksum
from .data import Images, SKETCHY_SPLITS, UNSEEN_CLASSES, configure_image_worker, normal_transform
from .encoding import load_frozen_clip, save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE


SEED = 42
PER_CLASS = 32
MASKS = 4
PARTS = 2
DST = DEFAULT_CACHE.parent / "results" / "paired-patch-interaction-diagnosis"


def read(path):
    return json.loads(path.read_bytes())


def class_generator(label):
    signature = hashlib.sha256(f"{SEED}/{label}".encode()).digest()
    return random.Random(int.from_bytes(signature[:8], "big"))


def cohort(root, name):
    classes, pool = {}, []
    for cls in UNSEEN_CLASSES[name]:
        photos = {p.stem: p for p in (root/"photo"/cls).iterdir() if p.is_file()}
        drawings = {}
        for p in sorted((root/"sketch"/cls).iterdir()):
            key, sep, number = p.stem.rpartition("-")
            if p.is_file() and sep and number.isdigit() and key in photos:
                drawings.setdefault(key, []).append(p)
        eligible = sorted(key for key, items in drawings.items() if len(items) >= 2)
        rng = class_generator(cls)
        ids = rng.sample(eligible, PER_CLASS)
        records = []
        for key in ids:
            members = [photos[key], *rng.sample(drawings[key], 2)]
            records.append([str(p.relative_to(root)) for p in members])
        classes[cls] = {"source_photos_with_two_drawings": len(eligible), "records": records}
        pool.extend(str(photos[k].relative_to(root)) for k in sorted(photos.keys() - drawings.keys()))
    refs = class_generator(name + "/carriers").sample(sorted(pool), 4 * MASKS)
    g = torch.Generator().manual_seed(SEED)
    partitions = []
    for _ in range(MASKS):
        mask = torch.zeros(49, dtype=torch.bool)
        mask[torch.randperm(49, generator=g)[:24]] = True
        partitions.append(mask.tolist())
    return {"split": name, "classes": classes, "refs": refs, "masks": partitions, "seed": SEED}


def embed_tokens(visual, tokens):
    last = visual.transformer(tokens)[:, 0].float()
    centered = last - last.mean(dim=-1, keepdim=True)
    return (centered * visual.ln_post.weight.float()) @ visual.proj.float()


def replace_groups(first, second, mask):
    return torch.where(mask, first, second)


class PanelMeasurement:
    def __init__(self, visual, references, mask_list):
        self.vision = visual
        embeddings = visual._embeds(references)
        self.banks = []
        for p in range(2):
            conditions = []
            for j, patch_mask in enumerate(mask_list):
                start = (p * MASKS + j) * 2
                u, v = embeddings[start:start+1], embeddings[start+1:start+2]
                mask = torch.tensor([True, *patch_mask], device=references.device).reshape(1, 50, 1)
                combined = replace_groups(v, u, mask)
                conditions.append((u, v, mask, embed_tokens(visual, combined), references[start:start+1]))
            self.banks.append(conditions)

    def measure(self, pixels, check=False):
        tokens = self.vision._embeds(pixels)
        full = embed_tokens(self.vision, tokens)
        diagnostics = {}
        if check:
            u, v, mask, _, u_pixels = self.banks[0][0]
            hybrid = replace_groups(tokens[:1], u, mask)
            pixel_mask = mask[:,1:].reshape(1,1,7,7).repeat_interleave(32,2).repeat_interleave(32,3)
            pixel_tokens = self.vision._embeds(torch.where(pixel_mask, pixels[:1], u_pixels))
            diagnostics["pixel_vs_token_max_abs"] = (hybrid-pixel_tokens).abs().amax().item()
            add = lambda t: t[:,1:].mean(dim=1)
            additive_check = add(tokens[:1])-add(hybrid)-add(replace_groups(v,tokens[:1],mask))+add(replace_groups(v,u,mask))
            diagnostics["additive_null_max_abs"] = additive_check.abs().amax().item()
            if max(diagnostics.values()) > 2e-4:
                raise RuntimeError(f"Patch intervention validation failed: {diagnostics}")
        states = {"full": full.cpu()}
        for panel, conditions in enumerate(self.banks):
            total = torch.zeros_like(full)
            for u, v, mask, base, _ in conditions:
                mix = torch.cat([replace_groups(tokens,u,mask), replace_groups(v,tokens,mask)])
                left, right = embed_tokens(self.vision,mix).chunk(2, dim=0)
                total += full - left - right + base
            states[f"interaction_{panel}"] = (total / MASKS).cpu()
        return states, diagnostics


def data_loader(paths, workers, batch):
    ds = Images(paths, normal_transform(), "rgb")
    return DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True,
                      worker_init_fn=configure_image_worker, prefetch_factor=2)


@torch.inference_mode()
def measure(args):
    spec = read(args.cache/"manifest.json")
    root = Path(spec["datasets"]["sketchy"]["root"])
    plan = cohort(root, args.split)
    classes = list(plan["classes"])[args.part::PARTS]
    observed = [member for cls in classes for subject in plan["classes"][cls]["records"] for member in subject]
    model = load_frozen_clip(spec["checkpoint"], torch.device("cuda")).float()
    visual = model.visual
    ds = Images([str(root/n) for n in plan["refs"]], normal_transform(), "rgb")
    references = torch.stack([img for img in ds]).cuda()
    panels = PanelMeasurement(visual, references, plan["masks"])
    outputs, diagnostics = {}, {}
    start = time.perf_counter()
    for batch, images in enumerate(data_loader([str(root/n) for n in observed], args.workers, args.batch)):
        chunks, checks = panels.measure(images.cuda(non_blocking=True), check=batch==0)
        diagnostics.update(checks)
        for key, chunk in chunks.items():
            outputs.setdefault(key, []).append(chunk)
    dest = args.output/f"{args.split}-{args.part}.pt"
    save_tensor({"classes":classes, "ordered_paths":observed, "states":{k:torch.cat(v) for k,v in outputs.items()}}, dest)
    atomic_json(dest.with_suffix(".json"), {"revision":args.revision, "plan":plan, "part":args.part,
                "classes":classes, "subjects":len(observed)//3, "images":len(observed), "device":torch.cuda.get_device_name(),
                "wall_seconds":time.perf_counter()-start, "validation":diagnostics, "sha256":checksum(dest)})
    print(args.split, args.part, len(observed), diagnostics, flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",choices=SKETCHY_SPLITS,required=True)
    parser.add_argument("--part",type=int,choices=range(PARTS),required=True)
    parser.add_argument("--revision",required=True)
    parser.add_argument("--cache",type=Path,default=DEFAULT_CACHE)
    parser.add_argument("--output",type=Path,default=DST)
    parser.add_argument("--batch",type=int,default=96)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(1)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False
    # Match single-image and batched patch convolution when validating token swaps.
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    measure(args)
