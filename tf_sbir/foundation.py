"""Cache OpenAI CLIP ViT-B/32 text embeddings for the fixed public vocabulary."""

import hashlib
from pathlib import Path

import open_clip
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from .encoding import load_frozen_clip, save_tensor


def cache_file(args, key):
    model = "b32"
    if args.clip_checkpoint:
        stat = Path(args.clip_checkpoint).stat()
        model = f"b32/{stat.st_size}/{stat.st_mtime_ns}"
    device = torch.device(args.device).type
    digest = hashlib.sha256(f"foundation-v2/{model}/{device}/{key}".encode()).hexdigest()[:20]
    return args.cache / f"b32-{key.split('/')[0]}-{digest}.pt"


@torch.inference_mode()
def load_words(args, words):
    signatures = hashlib.sha256("\0".join(words).encode()).hexdigest()[:20]
    file = cache_file(args, f"text/{signatures}")
    if file.is_file():
        return torch.load(file, map_location="cpu", weights_only=True, mmap=getattr(args, "memory_map", False))
    device = torch.device(args.device)
    model = load_frozen_clip(args.clip_checkpoint, device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    views = [[], []]
    for start in tqdm(range(0, len(words), 256), desc="CLIP B/32 public text vocabulary", mininterval=1.0):
        terms = words[start:start+256]
        templates = (
            ("a drawing of a {}.", "a sketch of the {}."),
            ("a photo of a {}.", "a photo of the {}."),
        )
        for result, (first, second) in zip(views, templates, strict=True):
            a = F.normalize(model.encode_text(tokenizer([first.format(w) for w in terms]).to(device)).float(), dim=-1)
            b = F.normalize(model.encode_text(tokenizer([second.format(w) for w in terms]).to(device)).float(), dim=-1)
            result.append(F.normalize(a + b, dim=-1).cpu())
    embeddings = [torch.cat(view) for view in views]
    save_tensor(embeddings, file)
    return embeddings
