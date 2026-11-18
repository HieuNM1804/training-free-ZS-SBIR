"""Frozen encoders and embedding caches for backbone-matched experiments."""

import hashlib
from pathlib import Path

import open_clip
import torch
from torch.nn import functional as F
from torchvision import transforms
from tqdm.auto import tqdm

from .data import Images, dataset_fingerprint, threaded_batches
from .encoding import load_frozen_clip, load_global_inputs, save_tensor


BACKBONES = {
    "b32": "OpenAI CLIP ViT-B/32",
    "dfn2b": "Apple DFN2B CLIP ViT-L/14",
    "siglip": "Google SigLIP ViT-B/16",
}
SIGLIP = "google/siglip-base-patch16-224"
SIGLIP_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"


def make_model(args, device):
    precision = "fp16" if device.type == "cuda" else "fp32"
    if args.backbone == "b32":
        model = load_frozen_clip(args.clip_checkpoint, device)
        return model, open_clip.get_tokenizer("ViT-B-32"), model.visual.conv1.weight.dtype
    if args.backbone == "dfn2b":
        model = open_clip.create_model(
            "ViT-L-14", pretrained="dfn2b", precision=precision, device=device, force_quick_gelu=True,
        ).eval().requires_grad_(False)
        return model, open_clip.get_tokenizer("ViT-L-14"), model.visual.conv1.weight.dtype
    from transformers import AutoTokenizer, SiglipModel

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = SiglipModel.from_pretrained(SIGLIP, revision=SIGLIP_REVISION, dtype=dtype)
    model = model.to(device).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(SIGLIP, revision=SIGLIP_REVISION)
    return model, tokenizer, dtype


def vision_features(model, pixels, backbone):
    if backbone == "siglip":
        return model.vision_model(pixel_values=pixels).pooler_output
    return model.encode_image(pixels)


def text_features(model, tokenizer, prompts, backbone, device):
    if backbone == "siglip":
        batch = tokenizer(prompts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        return model.text_model(input_ids=batch.input_ids.to(device)).pooler_output
    return model.encode_text(tokenizer(prompts).to(device))


def model_key(args):
    if args.backbone == "b32" and args.clip_checkpoint:
        stat = Path(args.clip_checkpoint).stat()
        return f"{args.backbone}/{stat.st_size}/{stat.st_mtime_ns}"
    return args.backbone


def cache_file(args, key):
    device = torch.device(args.device).type
    digest = hashlib.sha256(f"foundation-v2/{model_key(args)}/{device}/{key}".encode()).hexdigest()[:20]
    return args.cache / f"{args.backbone}-{key.split('/')[0]}-{digest}.pt"


@torch.inference_mode()
def load_vision(args, query_paths, photo_paths, needs_contours=False):
    if args.backbone == "b32":
        return load_global_inputs(args, query_paths, photo_paths, needs_contours)
    device = torch.device(args.device)
    items = [(query_paths, "sketch", "rgb"), (photo_paths, "photo", "rgb")]
    if needs_contours:
        items.append((photo_paths, "photo", "contours"))
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5) if args.backbone == "siglip" else (0.48145466, 0.4578275, 0.40821073),
            (0.5, 0.5, 0.5) if args.backbone == "siglip" else (0.26862954, 0.26130258, 0.27577711),
        ),
    ])
    results = []
    model = None
    for paths, domain, view in items:
        key = dataset_fingerprint(args.root, paths)
        cache = cache_file(args, f"{domain}-{view}/{key}")
        if cache.is_file():
            print(f"Cache {cache.name}", flush=True)
            results.append(torch.load(cache, map_location="cpu", weights_only=True))
            continue
        if model is None:
            model, _, dtype = make_model(args, device)
        batches = threaded_batches(
            Images(paths, transform, view), args.batch_size, args.workers, device.type == "cuda",
        )
        codes = []
        for batch in tqdm(
            batches, total=(len(paths) + args.batch_size - 1) // args.batch_size,
            desc=f"{args.backbone} {domain} {view}", mininterval=1.0,
        ):
            if view == "contours":
                batch = batch.flatten(0, 1)
            pixels = batch.to(device=device, dtype=dtype, non_blocking=True)
            encoding = F.normalize(vision_features(model, pixels, args.backbone).float(), dim=-1)
            if view == "contours":
                encoding = encoding.reshape(-1, 2, encoding.size(-1))
            codes.append(encoding.cpu())
        features = torch.cat(codes)
        save_tensor(features, cache)
        results.append(features)
    return results[0], results[1], results[2] if needs_contours else None


@torch.inference_mode()
def load_words(args, words):
    signatures = hashlib.sha256("\0".join(words).encode()).hexdigest()[:20]
    file = cache_file(args, f"text/{signatures}")
    if file.is_file():
        return torch.load(file, map_location="cpu", weights_only=True)
    device = torch.device(args.device)
    model, tokenizer, _ = make_model(args, device)
    views = [[], []]
    for start in tqdm(range(0, len(words), 256), desc=f"{args.backbone} public text vocabulary", mininterval=1.0):
        terms = words[start:start+256]
        templates = (
            ("a drawing of a {}.", "a sketch of the {}."),
            ("a photo of a {}.", "a photo of the {}."),
        )
        for result, (first, second) in zip(views, templates, strict=True):
            a = F.normalize(text_features(model, tokenizer, [first.format(w) for w in terms], args.backbone, device).float(), dim=-1)
            b = F.normalize(text_features(model, tokenizer, [second.format(w) for w in terms], args.backbone, device).float(), dim=-1)
            result.append(F.normalize(a + b, dim=-1).cpu())
    embeddings = [torch.cat(view) for view in views]
    save_tensor(embeddings, file)
    return embeddings
