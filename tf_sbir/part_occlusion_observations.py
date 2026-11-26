"""Measure frozen CLIP on single and double region deletions before its first transformer block."""

from collections import defaultdict
from itertools import combinations
import json
import os
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .benchmark import merge_inputs
from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT, runtime
from .data import Images, configure_image_worker, dataset_fingerprint, normal_transform
from .encoding import load_frozen_clip, save_tensor
from .gallery_convex_support import SEED
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import SKETCHY
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt


FAMILY = "heldout-occlusion-correspondence"
SINGLES = tuple((i,) for i in range(4))
PAIRS = tuple(combinations(range(4), 2))
DELETIONS = SINGLES + PAIRS
CACHE_SIZE = 1024
PIXEL_BATCH = 128


def public_words():
    spec = json.loads((DOCS / VOCABULARY_STUDY / "freeze.json").read_bytes())
    checksum_vocab = spec.pop("sha256")
    words = spec["subsets"]["boxable"]["words"]
    if len(words) != 601 or digest(spec) != checksum_vocab:
        raise ValueError("Unexpected public vocabulary")
    return words, checksum_vocab


def seed_job():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")


def masks(device):
    # Four equal regions. The middle row and column stay visible in every view.
    parts = [[] for _ in SINGLES]
    for y in range(7):
        for x in range(7):
            if x != 3 and y != 3:
                parts[2 * (y > 3) + (x > 3)].append(1 + 7 * y + x)
    kept = defaultdict(list)
    for slot, deleted in enumerate(DELETIONS):
        remove = {idx for part in deleted for idx in parts[part]}
        keep = sorted(set(range(50)) - remove)
        kept[len(keep)].append((slot, keep))
    return [(torch.tensor([slot for slot, _ in rows], device=device),
             torch.tensor([indices for _, indices in rows], device=device)) for rows in kept.values()]


def mirror_order(device):
    return torch.tensor([DELETIONS.index(tuple(sorted(i ^ 1 for i in d))) for d in DELETIONS], device=device)


def read_tokens(visual, tokens):
    z = visual.transformer(tokens)
    z, _ = visual._pool(z)
    z = z @ visual.proj
    return F.normalize(z.float(), dim=-1)


def read_masks(model, text, pixels, kept, flip_order, sketch, verify):
    size = len(pixels)
    visual = model.visual
    pixels = pixels.to(device=text.device, dtype=visual.conv1.weight.dtype, non_blocking=True)
    if sketch:
        pixels = torch.cat((pixels, pixels.flip((-1,))))
    tokens = visual._embeds(pixels)
    if tokens.size(1) != 50:
        raise ValueError("Expected class token and 7 x 7 patches")
    if verify:
        reference = F.normalize(model.encode_image(pixels[:2]).float(), dim=-1)
        torch.testing.assert_close(read_tokens(visual, tokens[:2]), reference, atol=3e-4, rtol=2e-3)
    bank = torch.empty((len(tokens), len(DELETIONS), len(text)), device=text.device)
    scale = model.logit_scale.exp().float()
    for positions, indices in kept:
        observed = tokens[:, indices, :].flatten(0, 1).contiguous()
        logits = torch.cat([(scale * (read_tokens(visual, part) @ text.T)).softmax(-1)
                            for part in observed.split(768)])
        bank[:, positions] = logits.reshape(len(tokens), len(positions), len(text))
    if sketch:
        return .5 * (bank[:size] + bank[size:, flip_order])
    return bank


def directory(cfg, dataset, mode, paths, vocab_sha):
    signature = {
        "feature_code_sha256": checksum(Path(__file__)), "vocabulary_sha256": vocab_sha,
        "ordered_paths": dataset_fingerprint(cfg.root, paths), "checkpoint": CHECKPOINT,
        "prompt": DEFAULT_PROMPTS[int(mode == "photo")], "mode": mode, "images": len(paths),
        "seed": SEED, "torch": str(torch.__version__), "open_clip": open_clip.__version__,
        "gpu": torch.cuda.get_device_name(), "regions": "four corner 3x3 grids; central cross fixed",
        "sketch_views": ["original", "horizontal_flip_physical_regions_registered"],
    }
    key = digest(signature)
    folder = cfg.cache / FAMILY / f"{dataset}-{mode}-{key[:16]}"
    return folder, signature


def read_cache(folder):
    cfg = json.loads((folder / "complete.json").read_bytes())
    if not cfg["chunks"]:
        raise ValueError("Empty masked readout cache")
    parts = []
    for record in cfg["chunks"]:
        path = folder / record["file"]
        if checksum(path) != record["sha256"]:
            raise ValueError("Incorrect probability cache hash")
        parts.append(torch.load(path, weights_only=True, mmap=True))
    bank = torch.cat(parts)
    if bank.shape != (cfg["specification"]["images"], len(DELETIONS), 601):
        raise ValueError("Wrong number or dimensions of masked outputs")
    return bank


def encode_dataset(cfg, dataset, mode, paths, model, text, vocab_sha):
    folder, spec = directory(cfg, dataset, mode, paths, vocab_sha)
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / "complete.json"
    if done.is_file():
        recorded = json.loads(done.read_bytes())
        if recorded["specification"] != spec:
            raise ValueError("Readout cache collision")
        if all((folder / row["file"]).is_file() and checksum(folder / row["file"]) == row["sha256"]
               for row in recorded["chunks"]):
            print(dataset, mode, "verified", len(paths), flush=True)
            return
    configure_image_worker(None)
    kept, flip_order = masks(text.device), mirror_order(text.device)
    records = []
    for start in range(0, len(paths), CACHE_SIZE):
        chunk_paths = paths[start:start + CACHE_SIZE]
        path = folder / f"{start:06}.pt"
        stamp = path.with_suffix(".json")
        if stamp.exists() and path.exists():
            info = json.loads(stamp.read_bytes())
            if info["sha256"] == checksum(path) and info["size"] == len(chunk_paths):
                records.append(info)
                continue
        data = Images(chunk_paths, normal_transform(), "rgb")
        loader = DataLoader(data, batch_size=PIXEL_BATCH, num_workers=cfg.workers,
                            pin_memory=True, worker_init_fn=configure_image_worker)
        bank = []
        for j, batch in enumerate(loader):
            posterior = read_masks(model, text, batch, kept, flip_order, mode == "sketch", start == j == 0)
            if not torch.isfinite(posterior).all() or (posterior.sum(2) - 1).abs().max() > 2e-5:
                raise ValueError("Invalid concept posterior")
            bank.append(posterior.cpu())
        save_tensor(torch.cat(bank), path)
        record = {"file": path.name, "size": len(chunk_paths), "sha256": checksum(path)}
        atomic_json(stamp, record)
        records.append(record)
        print(dataset, mode, start + len(chunk_paths), "/", len(paths), flush=True)
    atomic_json(done, {"specification": spec, "chunks": records})


@torch.inference_mode()
def run():
    seed_job()
    w = Path("/content/sbir")
    words, vocab_sha = public_words()
    workers = min(8, len(os.sched_getaffinity(0)))
    cfg = runtime(w, "sketchy", workers=workers)
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = [encode_prompt(model, tokenizer, words, template) for template in DEFAULT_PROMPTS]
    for dataset, splits in (("sketchy", SKETCHY), ("tuberlin", ("tuberlin",))):
        cfg = runtime(w, dataset, workers=workers)
        for mode, bank in zip(("sketch", "photo"), text, strict=True):
            paths, _ = merge_inputs(cfg.root, splits, mode)
            encode_dataset(cfg, dataset, mode, paths, model, bank, vocab_sha)


if __name__ == "__main__":
    run()
