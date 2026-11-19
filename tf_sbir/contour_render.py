"""Use NPAR XDoG photos as gallery-derived sketch proxies."""

from functools import partial

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .data import threaded_batches
from .encoding import global_cache_name, load_frozen_clip, save_tensor
from .gallery_diffusion import diffusion_rows, full_gallery_inputs, linear_fusion_rows, mutual_transition
from .global_methods import centered, prepare_methods
from .manifold import build_graph, iterative_feedback
from .mta import normalized_crops, pixel_lookup


def xdog(gray, sigma):
    """Apply the weighted difference of Gaussians and one-sided tanh threshold."""
    sharp = cv2.GaussianBlur(gray, (0, 0), sigma)
    wide = cv2.GaussianBlur(gray, (0, 0), 1.6 * sigma)
    below = np.minimum(sharp - 0.99 * wide + 0.02, 0)
    return np.rint(255 * (1 + np.tanh(50 * below))).astype(np.uint8)


class XDoGInputs(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            image = source.convert("L").resize((224, 224), Image.Resampling.BILINEAR)
        gray = np.asarray(image, dtype=np.float32) / 255
        return torch.from_numpy(np.stack([xdog(gray, sigma) for sigma in (0.8, 1.6)]))


@torch.inference_mode()
def load_xdog(args, paths):
    device = torch.device(args.device)
    precision = "fp16" if device.type == "cuda" else "fp32"
    filename = args.cache / global_cache_name(args.root, paths, "photo", "xdog-v1", precision, args.clip_checkpoint)
    if not filename.is_file():
        model = load_frozen_clip(args.clip_checkpoint, device)
        lookup = pixel_lookup(device, model.visual.conv1.weight.dtype)
        crop_batch = getattr(args, "crop_batch_size", None) or args.batch_size
        workers = getattr(args, "crop_workers", None)
        batch_size = max(1, crop_batch // 2)
        batches = threaded_batches(XDoGInputs(paths), batch_size, args.workers if workers is None else workers, device.type == "cuda")
        total = (len(paths) + batch_size - 1) // batch_size
        encoded = []
        for batch in tqdm(batches, total=total, desc="CLIP XDoG contours"):
            pixels = batch.to(device=device, non_blocking=True).flatten(0, 1).unsqueeze(1).expand(-1, 3, -1, -1)
            normalized = normalized_crops(pixels, lookup)
            descriptors = F.normalize(model.encode_image(normalized).float(), dim=-1)
            encoded.append(descriptors.unflatten(0, (-1, 2)).cpu())
        save_tensor(torch.cat(encoded), filename)
    print(f"Cache {filename.name}", flush=True)
    return torch.load(filename, map_location="cpu", mmap=True, weights_only=True)


def contour_sources(canny, rendered):
    yield "canny", canny
    yield "xdog", rendered
    yield "canny-xdog", torch.cat((canny, rendered), dim=1)


def xdog_image_settings(queries, photos, contours, rendered):
    sources = contour_sources(contours, rendered)
    geometry = centered(photos, photos.mean(dim=0))
    graph = build_graph(geometry, geometry, 640)
    bank = F.normalize(graph.diffuse(geometry, restart=0.1, steps=4), dim=-1)
    for name, proxies in sources:
        query, _ = prepare_methods(queries, photos, proxies, ("local_residual",))["local_residual"]
        yield f"{name}-residual", query, geometry
        yield f"{name}-residual-walk-k640", iterative_feedback(query, bank, 64, 2, 2.0), bank


def xdog_transport_settings(queries, photos, contours, sketch_words, photo_words, rendered, score_batch):
    for name, proxies in contour_sources(contours, rendered):
        base, region, gallery, _, _ = full_gallery_inputs(queries, photos, proxies, sketch_words, photo_words, score_batch)
        yield f"{name}-full-transport", partial(linear_fusion_rows, base, region, score_batch, 0.25)
        transition = mutual_transition(gallery, 160)
        yield f"{name}-transport-diffusion", partial(diffusion_rows, base, region, transition, score_batch, 0.9, 320, "regional")
