"""Cache the existing frozen baseline inputs from read-only Kaggle mounts on multiple GPUs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .data import Images, SKETCHY_SPLITS, UNSEEN_CLASSES, configure_image_worker, normal_transform
from .encoding import load_frozen_clip, save_tensor
from .geometry_word_flat import COUNTS
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import Split
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt


DEFAULT_CACHE = Path("/kaggle/working/sbir-offline/frozen-unseen")
BLOCK = 1536
SEED = 42


def sorted_members(directory):
    with os.scandir(directory) as entries:
        members = sorted(entry.name for entry in entries if entry.is_file())
    if not members:
        raise ValueError(f"Empty class directory {directory}")
    return [f"{directory.parent.name}/{directory.name}/{name}" for name in members]


def manifest(output, roots, checkpoint):
    with checkpoint.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != CHECKPOINT:
            raise ValueError("Unexpected OpenAI CLIP weights")
    spec = {"checkpoint": str(checkpoint), "sha256": CHECKPOINT, "seed": SEED, "block": BLOCK, "datasets": {}}
    with ThreadPoolExecutor(max_workers=min(32, len(os.sched_getaffinity(0)))) as pool:
        for dataset, root in roots.items():
            names = SKETCHY_SPLITS if dataset == "sketchy" else ("tuberlin",)
            classes = list(dict.fromkeys(c for name in names for c in UNSEEN_CLASSES[name]))
            pack = {"root": str(root), "order": {}, "ranges": {}}
            artifacts = {}
            for mode in ("sketch", "photo"):
                lists = dict(zip(classes, pool.map(sorted_members, [root / mode / c for c in classes]), strict=True))
                paths = [path for cls in classes for path in lists[cls]]
                position = {path: i for i, path in enumerate(paths)}
                views = {}
                for name in names:
                    indices, labels = [], []
                    for label, cls in enumerate(UNSEEN_CLASSES[name]):
                        indices.extend(position[path] for path in lists[cls])
                        labels.extend([label] * len(lists[cls]))
                    if len(indices) != COUNTS[name][int(mode == "photo")]:
                        raise ValueError(f"Unexpected {name} {mode} count {len(indices)}")
                    views[name] = (torch.tensor(indices), torch.tensor(labels))
                artifacts[mode] = {"paths": paths, "views": views}
                pack["order"][mode] = hashlib.sha256(("\n".join(paths)+"\n").encode()).hexdigest()
                pack["ranges"][mode] = len(paths)
            signature = digest(pack)
            save_tensor(artifacts, output / f"index-{dataset}-{signature[:16]}.pt")
            pack["signature"] = signature
            spec["datasets"][dataset] = pack
            print(dataset, pack["ranges"], "validated", flush=True)
    atomic_json(output / "manifest.json", spec)


def text_inputs(model, output):
    vocab = json.loads((DOCS / VOCABULARY_STUDY / "freeze.json").read_bytes())
    checksum = vocab.pop("sha256")
    words = vocab["subsets"]["boxable"]["words"]
    if len(words) != 601 or digest(vocab) != checksum:
        raise ValueError("Invalid locked word list")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    bank = {prompt: encode_prompt(model, tokenizer, words, prompt).cpu() for prompt in DEFAULT_PROMPTS}
    save_tensor({"words": words, "text": bank, "scale": float(model.logit_scale.exp()),
                 "vocabulary_sha": checksum, "checkpoint": CHECKPOINT}, output / "text.pt")


def index_file(root, dataset, record):
    return root / f"index-{dataset}-{record['signature'][:16]}.pt"


def chunk_file(root, dataset, mode, start, record):
    return root / f"{dataset}-{record['signature'][:16]}" / f"{mode}-{start:06d}.pt"


@torch.inference_mode()
def encode(root, rank, world, batch, workers):
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    spec = json.loads((root / "manifest.json").read_bytes())
    model = load_frozen_clip(spec["checkpoint"], device)
    if rank == 0:
        text_inputs(model, root)
    count = 0
    dtype = model.visual.conv1.weight.dtype
    for dataset, record in spec["datasets"].items():
        index = torch.load(index_file(root, dataset, record), weights_only=True)
        for mode in ("sketch", "photo"):
            paths = index[mode]["paths"]
            todo = []
            for start in range(0, len(paths), BLOCK):
                dest = chunk_file(root, dataset, mode, start, record)
                if count % world == rank and not dest.is_file():
                    todo.append((start, min(start + BLOCK, len(paths)), dest))
                count += 1
            if not todo:
                continue
            subset = [str(Path(record["root"]) / name) for start, end, _ in todo for name in paths[start:end]]
            ds = Images(subset, normal_transform(), "rgb")
            batches = DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True,
                                 worker_init_fn=configure_image_worker, prefetch_factor=2)
            stream = iter(batches)
            for start, end, dest in todo:
                original, mirror = [], []
                begun = time.monotonic()
                for off in range(start, end, batch):
                    pixels = next(stream).to(device=device, dtype=dtype, non_blocking=True)
                    if len(pixels) != min(batch, end - off):
                        raise ValueError("Cache block alignment changed")
                    def embed(img):
                        return F.normalize(model.encode_image(img).float(), dim=-1)
                    original.append(embed(pixels).cpu())
                    if mode == "sketch":
                        mirror.append(embed(pixels.flip((-1,))).half().cpu())
                pack = {"start": start, "end": end, "rgb": torch.cat(original),
                        "checkpoint": CHECKPOINT, "signature": record["signature"]}
                if mirror:
                    pack["mirror"] = torch.cat(mirror)
                save_tensor(pack, dest)
                print(f"GPU {rank} {dataset} {mode} rows {start}-{end} in {time.monotonic()-begun:.1f}s", flush=True)


def load_split(root, name):
    spec = json.loads((root / "manifest.json").read_bytes())
    dataset = "sketchy" if name in SKETCHY_SPLITS else "tuberlin"
    record = spec["datasets"][dataset]
    index = torch.load(index_file(root, dataset, record), weights_only=True)
    arrays = {}
    for mode in ("sketch", "photo"):
        blocks = []
        length = len(index[mode]["paths"])
        for start in range(0, length, BLOCK):
            block = torch.load(chunk_file(root, dataset, mode, start, record), weights_only=True, mmap=True)
            if (block["start"], block["end"], block["checkpoint"], block["signature"]) != (
                start, min(start+BLOCK, length), CHECKPOINT, record["signature"]
            ):
                raise ValueError("Invalid feature shard")
            blocks.append(block)
        arrays[mode] = torch.cat([block["rgb"] for block in blocks])
        if mode == "sketch":
            arrays["mirror"] = torch.cat([block["mirror"] for block in blocks])
    lex = torch.load(root / "text.pt", weights_only=True)
    split = Split.from_inputs((index["sketch"]["views"], index["photo"]["views"], arrays["sketch"],
                              arrays["photo"], arrays["mirror"], None, None, lex["scale"], None), name)
    return split, {k: v.cuda() for k, v in lex["text"].items()}, lex["words"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("manifest", "encode"))
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--sketchy", type=Path)
    parser.add_argument("--tuberlin", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--batch", type=int, default=384)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "manifest":
        manifest(args.cache, {"sketchy": args.sketchy, "tuberlin": args.tuberlin}, args.checkpoint)
    else:
        if BLOCK % args.batch:
            raise ValueError("Select a batch size that divides the feature block")
        encode(args.cache, args.rank, args.world, args.batch, args.workers)
