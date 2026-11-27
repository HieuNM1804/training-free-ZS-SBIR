"""Measure real CLIP changes with subject identity fixed and explicit text controls."""

import argparse
import json
from pathlib import Path

import open_clip
from PIL import Image
import torch
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

from .concept_benchmark import atomic_json, checksum
from .crossdomain_inputs import CHECKPOINT
from .data import CLIP_MEAN, CLIP_STD, SKETCHY_SPLITS, UNSEEN_CLASSES, configure_image_worker, normal_transform
from .encoding import load_frozen_clip, save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE, index_file, load_visual_split
from .native_block_probe import SEED, deterministic
from .profiled_study import true_sources
from .semantic_boxable_single_prompt import encode_prompt


OUT = DEFAULT_CACHE.parent / "results" / "native-choice-domain-contrast"
SPLITS = (*SKETCHY_SPLITS, "tuberlin")
DOMAINS = {"drawing_to_photo": ("drawing", "photo"), "grayscale_to_rgb": ("black and white photo", "color photo")}


class PhotoPairs(Dataset):
    def __init__(self, root, names):
        self.root = root
        self.names = names
        self.resize = normal_transform().transforms[0]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        with Image.open(self.root / self.names[i]) as image:
            rgb = self.resize(image.convert("RGB"))
            return pil_to_tensor(rgb), pil_to_tensor(rgb.convert("L"))


def text_conditions(model, classes):
    names = [v.replace("_", " ") for v in classes]
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    banks = {}
    for condition, domains in DOMAINS.items():
        rows = [encode_prompt(model, tokenizer, names, f"a {domain} of {article} {{}}.")
                for domain in domains for article in ("a", "the")]
        matrix = torch.stack(rows).double().cpu()
        banks[condition] = {"meaning": matrix.mean(0), "depiction": (matrix[2:] - matrix[:2]).mean(0),
                            "article": ((matrix[1] - matrix[0]) + (matrix[3] - matrix[2])) / 2}
    return banks


def human_change(split, classes):
    visual = normalize(normalize(split.sketch.float(), dim=1) + normalize(split.mirror.float(), dim=1), dim=1)
    photo = normalize(split.photo.float(), dim=1)
    labels = split.sketch_labels.cuda()
    if split.name in SKETCHY_SPLITS:
        source = true_sources(split, DEFAULT_CACHE)
        if (source < 0).any() or not torch.equal(split.photo_labels.cuda()[source], labels):
            raise ValueError("Invalid drawing/source association")
        ids, group = source.unique(return_inverse=True)
        means = torch.stack([visual[group == i].double().mean(0) for i in range(len(ids))])
        values = photo[ids].double() - means
        row_labels = split.photo_labels[ids.cpu()].cuda()
        y = torch.stack([values[row_labels == c].mean(0) for c in range(len(classes))])
        counts = row_labels.bincount(minlength=len(classes))
        return y.cpu(), {"sampling": "one vote per paired source identity", "source_counts": counts.cpu().tolist()}
    y = torch.stack([photo[split.photo_labels == i].double().mean(0) - visual[labels == i].double().mean(0)
                     for i in range(len(classes))])
    return y.cpu(), {"sampling": "unpaired within-class mean photo minus mean drawing",
                     "queries": len(visual), "photos": len(photo)}


@torch.inference_mode()
def record(args):
    manifest = json.loads((DEFAULT_CACHE / "manifest.json").read_bytes())
    if manifest["sha256"] != CHECKPOINT:
        raise ValueError("Unexpected checkpoint")
    data = "sketchy" if args.split in SKETCHY_SPLITS else "tuberlin"
    entry = manifest["datasets"][data]
    index = torch.load(index_file(DEFAULT_CACHE, data, entry), weights_only=True)
    split = load_visual_split(DEFAULT_CACHE, args.split)
    classes = UNSEEN_CLASSES[args.split]
    positions = torch.arange(args.part, len(split.gallery_ids), args.parts)
    names = [index["photo"]["paths"][i] for i in split.gallery_ids[positions].tolist()]
    ds = PhotoPairs(Path(entry["root"]), names)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, shuffle=False, pin_memory=True,
                        worker_init_fn=configure_image_worker, generator=torch.Generator().manual_seed(SEED),
                        prefetch_factor=2 if args.workers else None)
    model = load_frozen_clip(manifest["checkpoint"], torch.device("cuda"))
    dtype = model.visual.conv1.weight.dtype
    mu = torch.tensor(CLIP_MEAN, device="cuda")[None, :, None, None]
    sigma = torch.tensor(CLIP_STD, device="cuda")[None, :, None, None]
    sums = torch.zeros(len(classes), 512, dtype=torch.float64, device="cuda")
    counts = torch.zeros(len(classes), dtype=torch.long, device="cuda")
    min_replay, total_movement, offset = 1.0, 0.0, 0
    for rgb, gray in loader:
        n = len(rgb)
        both = torch.cat((rgb.cuda(non_blocking=True), gray.cuda(non_blocking=True).expand(-1, 3, -1, -1)))
        image = (both.float().div_(255).sub_(mu).div_(sigma)).to(dtype)
        color, mono = normalize(model.encode_image(image).float(), dim=1).split(n)
        rows = positions[offset:offset + n]
        native = split.photo[rows]
        min_replay = min(min_replay, (color * normalize(native.float(), dim=1)).sum(1).min().item())
        d = color.double() - mono.double()
        total_movement += d.norm(dim=1).sum().item()
        tag = split.photo_labels[rows].cuda()
        for c in tag.unique().tolist():
            take = tag == c
            sums[c] += d[take].sum(0)
            counts[c] += take.sum()
        offset += n
    if offset != len(positions) or min_replay < 0.999:
        raise ValueError(f"Photo baseline replay mismatch {offset}, {min_replay}")
    saved = {"revision": args.revision, "positions": positions, "gallery_ids": split.gallery_ids[positions],
             "sum_color_changes": sums.cpu(), "class_counts": counts.cpu(), "classes": list(classes),
             "min_baseline_replay_cosine": min_replay, "sum_movement": total_movement}
    if args.part == 0:
        human, sampling = human_change(split, classes)
        saved.update({"text": text_conditions(model, classes), "human_change": human, "human_sampling": sampling})
    dest = OUT / f"observations-{args.split}-{args.part}.pt"
    save_tensor(saved, dest)
    atomic_json(dest.with_suffix(".json"), {"revision": args.revision, "seed": SEED, "sha256": checksum(dest),
                "photos": len(positions), "device": torch.cuda.get_device_name(), "parts": args.parts,
                "part": args.part, "split": args.split, "min_replay_cosine": min_replay})
    print(args.split, args.part, len(positions), min_replay, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--part", required=True, type=int)
    parser.add_argument("--parts", type=int, default=2)
    parser.add_argument("--batch", type=int, default=384)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    if not 0 <= args.part < args.parts:
        parser.error("Part outside sharding range")
    deterministic()
    record(args)
