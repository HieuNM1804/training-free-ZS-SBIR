"""Encode and cache the frozen global CLIP inputs."""

import hashlib
import os
from pathlib import Path

import open_clip
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import Images, configure_image_worker, dataset_fingerprint, normal_transform, threaded_batches


def global_cache_name(root, paths, mode, view, precision, checkpoint):
    source = dataset_fingerprint(root, paths)
    model = checkpoint or "ViT-B-32/openai"
    if checkpoint:
        stat = Path(checkpoint).stat()
        model = f"{Path(checkpoint).resolve()}/{stat.st_size}/{stat.st_mtime_ns}"
    key = hashlib.sha256(f"clip-v1/{model}/{precision}/{source}".encode()).hexdigest()[:20]
    return f"clip-{mode}-{view}-{key}.pt"


def save_tensor(tensor, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    torch.save(tensor, temporary)
    temporary.replace(output)


@torch.inference_mode()
def encode_clip(model, paths, view, batch_size, workers, device):
    configure_image_worker(None)
    dataset = Images(paths, normal_transform(), view)
    if os.name == "nt" and workers:
        batches = threaded_batches(dataset, batch_size, workers, device.type == "cuda")
    else:
        batches = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                             pin_memory=device.type == "cuda", worker_init_fn=configure_image_worker)
    output = []
    dtype = model.visual.conv1.weight.dtype
    for batch in tqdm(batches, desc=f"CLIP {view}", total=(len(paths) + batch_size - 1) // batch_size):
        if view == "contours":
            batch = batch.flatten(0, 1)
        embeddings = F.normalize(model.encode_image(batch.to(device=device, dtype=dtype, non_blocking=True)).float(), dim=-1)
        if view == "contours":
            embeddings = embeddings.reshape(-1, 2, embeddings.shape[-1])
        output.append(embeddings.cpu())
    return torch.cat(output)


def load_frozen_clip(checkpoint, device):
    precision = "fp16" if device.type == "cuda" else "fp32"
    model = open_clip.load_openai_model(checkpoint or "ViT-B-32", precision=precision, device=device)
    visual = model.visual
    if (
        visual.conv1.kernel_size != (32, 32)
        or visual.conv1.out_channels != 768
        or len(visual.transformer.resblocks) != 12
    ):
        raise ValueError("All experiment checkpoints must use the OpenAI CLIP ViT-B/32 architecture.")
    return model.eval().requires_grad_(False)


def load_global_inputs(args, query_paths, photo_paths, need_contours):
    device = torch.device(args.device)
    precision = "fp16" if device.type == "cuda" else "fp32"
    encodings = {}
    items = [("sketch", "rgb", query_paths), ("photo", "rgb", photo_paths)]
    if need_contours:
        items.append(("photo", "contours", photo_paths))
    model = None
    for mode, view, paths in items:
        cache = args.cache / global_cache_name(args.root, paths, mode, view, precision, args.clip_checkpoint)
        if cache.is_file():
            print(f"Cache {cache.name}", flush=True)
            encodings[(mode, view)] = torch.load(cache, map_location="cpu", weights_only=True)
            continue
        if model is None:
            model = load_frozen_clip(args.clip_checkpoint, device)
        features = encode_clip(model, paths, view, args.batch_size, args.workers, device)
        save_tensor(features, cache)
        encodings[(mode, view)] = features
    return encodings[("sketch", "rgb")], encodings[("photo", "rgb")], encodings.get(("photo", "contours"))
