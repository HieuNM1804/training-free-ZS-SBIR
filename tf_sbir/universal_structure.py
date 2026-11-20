"""Local self-similarity in a frozen CLIP patch grid, using the same descriptor in both domains."""

import math

from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from .data import threaded_batches
from .encoding import global_cache_name, load_frozen_clip, save_tensor
from .mta import normalized_crops, pixel_lookup


class StructureViews(Dataset):
    def __init__(self, paths, count):
        self.paths, self.count = paths, count

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            full = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        frames = [full]
        if self.count == 4:
            center = full.crop((22, 22, 202, 202)).resize((224, 224), Image.Resampling.BILINEAR)
            frames.extend((ImageOps.mirror(full), center, ImageOps.mirror(center)))
        return torch.stack([pil_to_tensor(x) for x in frames])


def layout(grid, device):
    coords = torch.arange(grid, device=device)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    coords = torch.stack((y.flatten(), x.flatten()), dim=1)
    offset = coords[None] - coords[:, None]
    radius = offset.float().norm(dim=-1)
    ring = torch.bucketize(radius, torch.tensor([1.5, 3.5], device=device))
    direction = (
        (torch.atan2(offset[..., 0].float(), offset[..., 1].float()) + math.pi) * (4 / math.pi)
    ).floor().long() % 8
    return (ring * 8 + direction), radius.eq(0), torch.triu_indices(grid * grid, grid * grid, 1, device=device)


def self_descriptor(patches, table):
    """Pool relative patch similarity by log-polar cell and retain the unordered pair geometry."""
    n, count, _ = patches.shape
    grid = math.isqrt(count)
    bins, self_pair, tri = table
    patches = patches.float()
    normal = F.normalize(patches - patches.mean(1, keepdim=True), dim=-1)
    similarity = normal @ normal.mT
    pairs = similarity[:, tri[0], tri[1]]
    pooled = similarity.new_full((n, count, 24), -1.0)
    pooled.scatter_reduce_(
        2,
        bins[None].expand(n, -1, -1),
        similarity.masked_fill(self_pair[None], -1.0),
        reduce="amax",
        include_self=True,
    )
    compact = F.adaptive_avg_pool2d(pooled.transpose(1, 2).reshape(n, 24, grid, grid), (3, 3)).flatten(1)
    a = F.normalize(pairs - pairs.mean(1, keepdim=True), dim=1)
    b = F.normalize(compact - compact.mean(1, keepdim=True), dim=1)
    return F.normalize(torch.cat((a, b), dim=1), dim=1)


@torch.inference_mode()
def structure_bank(args, paths, mode, count):
    name = global_cache_name(
        args.root, paths, mode, f"spatial-self-sim-v1-L4-L8-{count}views", "fp16", args.clip_checkpoint
    )
    file = args.cache / name
    if file.is_file():
        return torch.load(file, map_location="cpu", weights_only=True, mmap=True)
    device = torch.device("cuda")
    model = load_frozen_clip(args.clip_checkpoint, device)
    vision = model.visual
    lookup = pixel_lookup(device, vision.conv1.weight.dtype)
    table = layout(7, device)
    batches = threaded_batches(StructureViews(paths, count), max(1, 192 // count), 6, True)
    descriptors = []
    for frames in batches:
        inputs = normalized_crops(frames.flatten(0, 1), lookup)
        _, hidden = vision.transformer.forward_intermediates(
            vision._embeds(inputs), indices=[4, 8], stop_early=True
        )
        features = torch.stack([self_descriptor(t[:, 1:], table) for t in hidden], dim=1)
        descriptors.append(features.half().cpu().reshape(len(frames), count, 2, -1))
    saved = torch.cat(descriptors)
    save_tensor(saved, file)
    return saved
