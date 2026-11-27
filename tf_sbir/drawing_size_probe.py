"""Swap natural drawing extents while holding the reduced input raster fixed."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .conditional_completion_study import evaluate
from .data import SKETCHY_SPLITS, configure_image_worker, normal_transform
from .encoding import load_frozen_clip, save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE, index_file, load_split
from .native_block_probe import SEED, deterministic
from .paired_patch_interaction_probe import DST as OLD_PAIRS
from .replicated_simplex_study import evaluator
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import mean_percent
from .semantic_disputed_word_risk import query_posteriors, word_bank


SIZE = 224
DST = OLD_PAIRS.parent / "drawing-size-crossover"
AXIS = np.arange(SIZE, dtype=np.float64) + 0.5
RESAMPLE = Image.Resampling.BILINEAR


def raster(path):
    with Image.open(path) as image:
        return image.convert("RGB").resize((SIZE, SIZE), RESAMPLE)


def moments(image):
    # Continuous absorption on a known white drawing canvas. No binary mask is produced.
    mass = 255 - np.asarray(image).mean(axis=2, dtype=np.float64)
    total = mass.sum()
    if total == 0:
        return SIZE / 2, SIZE / 2, 0.0
    px, py = mass.sum(0) / total, mass.sum(1) / total
    x, y = float(px @ AXIS), float(py @ AXIS)
    radius = float(np.sqrt(px @ (AXIS-x)**2 + py @ (AXIS-y)**2))
    return x, y, radius


def original_moments(path):
    return moments(raster(path))


def display(image, center, side):
    if side == SIZE:
        return (image, image)
    reduced = image.resize((side, side), RESAMPLE)
    room = SIZE - side
    offset = tuple(min(room, max(0, round(v * room / SIZE))) for v in center)
    small = Image.new("RGB", (SIZE, SIZE), "white")
    small.paste(reduced, offset)
    return small, reduced.resize((SIZE, SIZE), RESAMPLE)


class DisplayViews(Dataset):
    def __init__(self, paths, xy, sides):
        self.paths, self.xy, self.sides = paths, xy, sides
        self.preprocess = normal_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = raster(self.paths[index])
        small, restored = display(image, self.xy[index], int(self.sides[index]))
        return self.preprocess(small), self.preprocess(restored), torch.tensor(moments(small), dtype=torch.float64)


def source_sizes(source_ids, geometry):
    sides = np.full(len(source_ids), SIZE, dtype=np.int64)
    for sid in np.unique(source_ids):
        ids = np.flatnonzero((source_ids == sid) & (geometry[:, 2] > 0))
        if len(ids):
            radii = geometry[ids, 2]
            sides[ids] = np.rint(SIZE * radii.min() / radii).astype(int).clip(1, SIZE)
    return sides


@torch.inference_mode()
def observations(paths, geometry, sides, model, workers, batch, identity_ids, references):
    ds = DisplayViews(paths, geometry[:, :2], sides)
    batches = DataLoader(ds, shuffle=False, num_workers=workers, batch_size=batch, pin_memory=True,
                         prefetch_factor=2, worker_init_fn=configure_image_worker)
    outputs = {key: [[], []] for key in ("small", "restored")}
    measured = []
    dtype = model.visual.conv1.weight.dtype
    for small, restored, actual in batches:
        measured.append(actual)
        for name, view in (("small", small), ("restored", restored)):
            view = view.to(device="cuda", dtype=dtype, non_blocking=True)
            for flip, pixels in enumerate((view, view.flip((-1,)))):
                unit = F.normalize(model.encode_image(pixels).float(), dim=-1)
                outputs[name][flip].append((unit.half() if flip else unit).cpu())
    fields = {key: tuple(torch.cat(a).cuda().float() for a in pair) for key, pair in outputs.items()}
    agreement = []
    for vectors in fields.values():
        agreement.extend((F.cosine_similarity(a[identity_ids], b[identity_ids].float()).min().item())
                         for a, b in zip(vectors, references, strict=True))
    if min(agreement) < .9999:
        raise ValueError(f"Same-size input failed to reproduce cached CLIP views: {agreement}")
    return fields, torch.cat(measured), min(agreement)


def score_methods(split, fields, text, kd, reference_pairs=None):
    visual_index = F.normalize(split.photo.float(), dim=-1)
    semantic_index = word_bank(split.photo, text[DEFAULT_PROMPTS[1]], split.scale).T.contiguous()
    replay = PrefixReplay(kd, split.photo_labels, split.name)
    rows, margins = {}, {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, (rgb, flip) in fields.items():
            visual = F.normalize(F.normalize(rgb.float(), dim=-1) + F.normalize(flip.float(), dim=-1), dim=-1)
            semantic = query_posteriors(rgb.float(), flip.float(), text[DEFAULT_PROMPTS[0]], split.scale)
            for k, query, gallery in (("native", visual, visual_index), ("semantic", semantic, semantic_index)):
                key = f"{k}_{name}"
                rows[key] = evaluate(split, query, gallery, replay, pool)
                if reference_pairs is not None:
                    i, j = reference_pairs
                    margins[key] = (query @ (gallery[i]-gallery[j])).cpu()
                print(key, len(query), mean_percent(rows[key]), flush=True)
    return rows, margins


@torch.inference_mode()
def measure(args):
    start = time.monotonic()
    name = args.split
    raw_path = OLD_PAIRS / f"real-errors-{name}.pt"
    archive = torch.load(raw_path, weights_only=True)
    if checksum(raw_path) != json.loads(raw_path.with_suffix(".json").read_bytes())["per_query_sha256"]:
        raise ValueError("Source association archive checksum mismatch")
    split, text, _ = load_split(DEFAULT_CACHE, name)
    if not torch.equal(archive["query_ids"], split.query_ids) or not torch.equal(archive["labels"], split.sketch_labels):
        raise ValueError("Original unseen query manifest changed")
    source = archive["source"]
    if source.min() < 0 or not torch.equal(split.photo_labels[source], split.sketch_labels):
        raise ValueError("Unknown or mislabelled source")
    owners = source.unique()[args.part::2]
    rows = torch.where(torch.isin(source, owners))[0]
    all_rows = split.query_ids[rows]
    spec = json.loads((DEFAULT_CACHE / "manifest.json").read_bytes())
    record = spec["datasets"]["sketchy"]
    db = torch.load(index_file(DEFAULT_CACHE, "sketchy", record), weights_only=True)
    paths = [Path(record["root"]) / db["sketch"]["paths"][i] for i in all_rows.tolist()]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        geometry = np.asarray(list(pool.map(original_moments, paths)))
    sides = source_sizes(source[rows].numpy(), geometry)
    model = load_frozen_clip(spec["checkpoint"], torch.device("cuda")).eval()
    orig = (split.sketch[rows], split.mirror[rows])
    fields, measured, replay_cos = observations(paths, geometry, sides, model, args.workers, args.batch,
                                               np.flatnonzero(sides == SIZE), orig)
    fields["original"] = orig
    reference = archive["measurements"]["semantic"]["endpoints"][6602].tolist() if name == "sketchy_2" else None
    sample = replace(split, sketch_labels=split.sketch_labels[rows])
    kd = evaluator(DEFAULT_CACHE.parent / "real-sketchy-smoke/evaluator.py")
    print(name, args.part, "scoring", len(rows), "against", len(split.photo), flush=True)
    result, margins = score_methods(sample, fields, text, kd, reference)
    dest = args.output / f"{name}-{args.part}.pt"
    save_tensor({"rows":rows, "labels":sample.sketch_labels, "sources":source[rows],
                 "geometry":torch.from_numpy(geometry), "display":measured, "sides":torch.from_numpy(sides),
                 "metrics":result, "rhino_margins":margins}, dest)
    info = {"revision":args.revision, "checkpoint":spec["sha256"], "seed":SEED, "split":name, "part":args.part,
            "queries":len(rows), "gallery":len(split.gallery_ids), "subjects":len(owners), "blank":int((geometry[:,2]==0).sum()),
            "controls_at_native_size":int((sides==SIZE).sum()), "min_identity_cosine":replay_cos,
            "metrics":{k:mean_percent(v) for k,v in result.items()}, "seconds":time.monotonic()-start,
            "device":torch.cuda.get_device_name(), "sha256":checksum(dest)}
    atomic_json(dest.with_suffix(".json"), info)
    print(name, args.part, "complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SKETCHY_SPLITS, required=True)
    parser.add_argument("--part", type=int, choices=(0,1), required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, default=DST)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256)
    deterministic()
    measure(parser.parse_args())
