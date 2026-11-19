"""Adapt CVPR 2024 MeanShift Test-time Augmentation to individual sketches."""

import hashlib
import math
import random

import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .data import normal_transform, threaded_batches
from .encoding import global_cache_name, load_frozen_clip, save_tensor
from .semantic import agreement_settings


class AugmentedImages(Dataset):
    def __init__(self, paths, views):
        self.paths, self.views = paths, views
        self.transform = normal_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            image = source.convert("RGB")
        digest = hashlib.blake2b(image.tobytes(), digest_size=8).digest()
        rng = random.Random(int.from_bytes(digest, "little"))
        return torch.stack([self.transform(random_resized_view(image, rng)) for _ in range(self.views)])


def random_resized_view(image, rng):
    """Match the published RandomResizedCrop and RandomHorizontalFlip distributions."""
    width, height = image.size
    for _ in range(10):
        area = width * height * rng.uniform(0.08, 1.0)
        aspect = math.exp(rng.uniform(math.log(0.75), math.log(4 / 3)))
        crop_width, crop_height = round(math.sqrt(area * aspect)), round(math.sqrt(area / aspect))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            left, top = rng.randrange(width - crop_width + 1), rng.randrange(height - crop_height + 1)
            view = image.crop((left, top, left + crop_width, top + crop_height))
            break
    else:
        view = image
    return ImageOps.mirror(view) if rng.random() < 0.5 else view


@torch.inference_mode()
def load_augmentations(args, paths, views=7, mode="sketch"):
    device = torch.device(args.device)
    precision = "fp16" if device.type == "cuda" else "fp32"
    name = global_cache_name(args.root, paths, mode, f"mta-v1-{views}", precision, args.clip_checkpoint)
    file = args.cache / name
    if file.is_file():
        return torch.load(file, map_location="cpu", weights_only=True)
    model = load_frozen_clip(args.clip_checkpoint, device)
    batch_size = max(1, args.batch_size // views)
    ds = AugmentedImages(paths, views)
    stream = threaded_batches(ds, batch_size, args.workers, device.type == "cuda")
    output = []
    for batch in tqdm(stream, total=(len(ds) + batch_size - 1) // batch_size, desc=f"CLIP {mode} views", mininterval=1):
        pixels = batch.flatten(0, 1).to(device, dtype=model.visual.conv1.weight.dtype, non_blocking=True)
        features = F.normalize(model.encode_image(pixels).float(), dim=-1)
        output.append(features.unflatten(0, batch.shape[:2]).half().cpu())
    cache = {"features": torch.cat(output), "logit_scale": model.logit_scale.exp().item()}
    save_tensor(cache, file)
    return cache


def robust_modes(views, support, logit_scale):
    """Compute the CVPR MTA inlierness and density updates per sketch."""
    modes = []
    count = views.size(1)
    neighbors = max(1, int(0.3 * (count - 1)))
    for batch in views.split(64):
        squared_distances = torch.cdist(batch, batch).square()
        closest = squared_distances.sort(dim=-1).values[:, :, 1:neighbors + 1]
        twice_bandwidth = closest.mean(dim=-1).clamp_min(1e-6)
        posterior = (logit_scale * (batch @ support.T)).softmax(dim=-1)
        affinity = posterior @ posterior.mT
        weights = batch.new_full(batch.shape[:2], 1 / count)
        mode = batch[:, :1]
        for _ in range(5):
            density = (-(batch - mode).square().sum(dim=-1) / twice_bandwidth).exp()
            for _ in range(5):
                supported = (affinity @ weights.unsqueeze(-1)).squeeze(-1)
                weights = ((density + 4 * supported) / 0.2).softmax(dim=-1)
            for _ in range(5):
                log_density = -(batch - mode).square().sum(dim=-1) / twice_bandwidth
                combined = (log_density + weights.log()).softmax(dim=-1)
                mode = F.normalize((batch * combined.unsqueeze(-1)).sum(dim=1, keepdim=True), dim=-1)
        modes.append(mode.squeeze(1))
    return torch.cat(modes)


def mta_settings(queries, views, logit_scale, photos, contours, sketch_words, photo_words):
    features = torch.cat((queries.unsqueeze(1), F.normalize(views.float(), dim=-1)), dim=1)
    uniform = F.normalize(features.mean(dim=1), dim=-1)
    modes = robust_modes(features, sketch_words, logit_scale)
    scoring = agreement_settings(torch.cat((queries, uniform, modes)), photos, contours, sketch_words, photo_words, topology=(320, "rgb"))
    combined, gallery = next((q, p) for name, q, p in scoring if name == "fuse-k320-rgb-w0.6")
    baseline, uniform, robust = combined.split(len(queries))
    yield "original-sketch", baseline, gallery
    yield "uniform-sketch-views", uniform, gallery
    yield "robust-mta-mode", robust, gallery
    yield "original-plus-mta", F.normalize(baseline + robust, dim=-1), gallery
