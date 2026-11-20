"""Compute phase congruency and encode the same phase and color views in either image domain."""

import argparse
import math
from pathlib import Path

from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from .benchmark import merge_inputs
from .crossdomain_inputs import runtime
from .data import SKETCHY_SPLITS, threaded_batches
from .encoding import global_cache_name, load_frozen_clip, save_tensor
from .mta import normalized_crops, pixel_lookup

TAG = "pc-four-octaves-six-orientations-two-polarities-v1"
VIEWS = 12


def filters(size, device):
    fy, fx = torch.meshgrid(
        torch.fft.fftfreq(size, device=device), torch.fft.fftfreq(size, device=device), indexing="ij"
    )
    radius = (fx.square() + fy.square()).sqrt()
    theta = torch.atan2(fy, fx)
    low = 1 / (1 + (radius / 0.45) ** 24)
    centers = 1 / torch.tensor([4.0, 8.0, 16.0, 32.0], device=device)
    scales = (
        torch.exp(
            -(radius.clamp_min(1e-8)[None] / centers[:, None, None]).log().square() / (2 * math.log(0.55) ** 2)
        )
        * low
    )
    scales[:, 0, 0] = 0
    angles = torch.arange(6, device=device) * math.pi / 6
    delta = theta[None] - angles[:, None, None]
    dist = torch.atan2(delta.sin(), delta.cos()).abs()
    spread = (1 + (dist * 3).clamp(max=math.pi).cos()) * 0.5
    return scales[None] * spread[:, None], angles


def phase_strength(image, bank, angles):
    signal = image - image.mean((-1, -2), keepdim=True)
    spectrum = torch.fft.fft2(signal)
    xx = torch.zeros_like(image)
    xy = torch.zeros_like(image)
    yy = torch.zeros_like(image)
    for kernel, angle in zip(bank, angles, strict=True):
        wave = torch.fft.ifft2(spectrum[:, None] * kernel[None])
        amp = wave.abs()
        total = amp.sum(1)
        maximum = amp.amax(1)
        direction = wave.sum(1)
        direction = direction / direction.abs().clamp_min(1e-7)
        relative = wave * direction.conj()[:, None]
        aligned = (relative.real - relative.imag.abs()).sum(1)
        noise = amp[:, 0].flatten(1).median(1).values / math.sqrt(math.log(4))
        noise = noise * (1 - 0.5 ** len(kernel)) / (1 - 0.5)
        threshold = noise * (math.sqrt(math.pi / 2) + 2 * math.sqrt((4 - math.pi) / 2))
        width = (total / maximum.clamp_min(1e-7) - 1) / (len(kernel) - 1)
        response = (
            (aligned - threshold[:, None, None]).clamp_min(0)
            * torch.sigmoid(10 * (width - 0.5))
            / total.clamp_min(1e-7)
        )
        x = response * angle.cos()
        y = response * angle.sin()
        xx.add_(x.square())
        xy.add_(2 * x * y)
        yy.add_(y.square())
    diff = ((xx - yy).square() + xy.square()).sqrt()
    return ((xx + yy + diff) / len(angles)).clamp_min(0).sqrt()


def square_box(structure):
    b, h, w = structure.shape
    mass = (structure - 0.12).clamp_min(0)

    def extent(v):
        cdf = v.cumsum(1)
        total = cdf[:, -1:]
        left = (cdf < total * 0.01).sum(1).float()
        right = (cdf < total * 0.99).sum(1).float() + 1
        return (left + right) / 2, (right - left).clamp_min(4)

    x, width = extent(mass.sum(1))
    y, height = extent(mass.sum(2))
    found = mass.sum((1, 2)) > 1e-4
    cx = torch.where(found, x, w / 2)
    cy = torch.where(found, y, h / 2)
    side = torch.where(found, torch.maximum(width, height), max(h, w))
    return torch.stack((cx, cy, side), dim=1)


def crop(image, box, zoom, border):
    b, c, h, w = image.shape
    x, y, side = box.unbind(1)
    matrix = image.new_zeros((b, 2, 3))
    matrix[:, 0, 0] = side * zoom / w
    matrix[:, 1, 1] = side * zoom / h
    matrix[:, 0, 2] = 2 * x / w - 1
    matrix[:, 1, 2] = 2 * y / h - 1
    grid = F.affine_grid(matrix, (b, c, 224, 224), align_corners=False)
    return (
        F.grid_sample(image - border, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + border
    )


def make_views(rgb, bank, angles):
    rgb = rgb.float() / 255
    gray = (rgb * rgb.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1)
    pc = phase_strength(gray, bank, angles)
    box = square_box(pc)
    perimeter = torch.cat((rgb[:, :, 0], rgb[:, :, -1], rgb[:, :, :, 0], rgb[:, :, :, -1]), -1)
    neutral = perimeter.median(-1).values[:, :, None, None]
    images = []
    for zoom in (1.08, 1.55):
        v = crop(rgb, box, zoom, neutral)
        images.extend((v, v.flip(-1)))
    canon = crop(pc[:, None], box, 1.08, pc.new_zeros((len(pc), 1, 1, 1)))
    for t in ((canon / 0.6).clamp(0, 1), ((canon - 0.10) / 0.20).clamp(0, 1)):
        for intensity in (1 - t, t):
            v = intensity.expand(-1, 3, -1, -1)
            images.extend((v, v.flip(-1)))
    return (torch.stack(images, dim=1).clamp(0, 1) * 255).round().to(torch.uint8)


class Frames(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as im:
            return pil_to_tensor(im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))


@torch.inference_mode()
def phase_bank(args, paths, mode):
    file = args.cache / global_cache_name(args.root, paths, mode, TAG, "fp16", args.clip_checkpoint)
    if file.is_file():
        return torch.load(file, map_location="cpu", weights_only=True, mmap=True)
    device = torch.device("cuda")
    model = load_frozen_clip(args.clip_checkpoint, device)
    lookup = pixel_lookup(device, model.visual.conv1.weight.dtype)
    mask, angles = filters(224, device)
    items = []
    for frames in threaded_batches(Frames(paths), 24, 5, True):
        v = make_views(frames.to(device, non_blocking=True), mask, angles)
        sampled = v.flatten(0, 1)
        encoded = [
            F.normalize(model.encode_image(normalized_crops(part, lookup)).float(), dim=-1)
            for part in sampled.split(144)
        ]
        items.append(torch.cat(encoded).reshape(len(v), VIEWS, -1).half().cpu())
    saved = torch.cat(items)
    save_tensor(saved, file)
    return saved


@torch.inference_mode()
def prepare(workspace, dataset):
    args = runtime(workspace, dataset)
    for mode in ("sketch", "photo"):
        paths, _ = merge_inputs(args.root, SKETCHY_SPLITS if dataset == "sketchy" else ("tuberlin",), mode)
        x = phase_bank(args, paths, mode)
        if x.shape != (len(paths), VIEWS, 512):
            raise ValueError("Unexpected phase bank")
        print(dataset, mode, list(x.shape), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    p.add_argument("--dataset", choices=("sketchy", "tuberlin"), required=True)
    cfg = p.parse_args()
    prepare(cfg.workspace, cfg.dataset)
