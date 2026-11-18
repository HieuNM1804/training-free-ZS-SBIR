"""Extract spatial patch tokens from the frozen OpenAI CLIP visual encoder."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset, default_collate
from tqdm.auto import tqdm

from .data import contour_views, normal_transform
from .encoding import global_cache_name, load_frozen_clip, save_tensor

SIZE = 224
PATCH = 32
GRID = SIZE // PATCH
GROUPS = (GRID + 1) // 2
SUPPORT_COUNT = GROUPS ** 2
UNIFORM_IDS = torch.tensor([row * GRID + col for row in range(0, GRID, 2) for col in range(0, GRID, 2)])


def selected_patches(ink):
    """Retain the most ink-supported patch in each spatial group."""
    mass = torch.from_numpy(ink.astype(np.float32)).view(GRID, PATCH, GRID, PATCH).mean(dim=(1, 3))
    mass = F.pad(mass, (0, 1, 0, 1), value=-1)
    groups = mass.reshape(GROUPS, 2, GROUPS, 2).permute(0, 2, 1, 3).reshape(-1, 4)
    values, offsets = groups.max(dim=1)
    cells = torch.arange(SUPPORT_COUNT)
    row = 2 * (cells // GROUPS) + offsets // 2
    col = 2 * (cells % GROUPS) + offsets % 2
    valid = values > 0
    if not valid.any():
        return UNIFORM_IDS, torch.ones(SUPPORT_COUNT, dtype=torch.bool)
    return row * GRID + col, valid


class PatchInputs(Dataset):
    def __init__(self, paths, mode):
        self.paths = paths
        self.mode = mode
        self.transform = normal_transform(SIZE)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            rgb = source.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        if self.mode == "photo":
            fine, contour = contour_views(rgb, SIZE)
            ink = np.minimum(np.asarray(fine.convert("L")), np.asarray(contour.convert("L"))) < 224
            ids, valid = selected_patches(ink)
            return torch.stack((self.transform(rgb), self.transform(contour))), ids, valid

        gray = np.asarray(rgb.convert("L"))
        border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
        if np.median(border) < 128:
            gray = 255 - gray
            rgb = Image.fromarray(gray).convert("RGB")
        ids, valid = selected_patches(gray < 224)
        thicker = Image.fromarray(cv2.erode(gray, np.ones((3, 3), dtype=np.uint8))).convert("RGB")
        return torch.stack((self.transform(rgb), self.transform(thicker))), ids, valid


def cache_path(args, paths, mode):
    precision = "fp16" if torch.device(args.device).type == "cuda" else "fp32"
    base = global_cache_name(args.root, paths, mode, "tokens", precision, args.clip_checkpoint)
    return args.cache / f"patch-v1-{base}"


def patch_batches(dataset, batch_size, workers, pin):
    ranges = iter(range(start, min(start + batch_size, len(dataset))) for start in range(0, len(dataset), batch_size))
    if workers == 0:
        for indices in ranges:
            yield default_collate([dataset[index] for index in indices])
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = [executor.submit(dataset.__getitem__, index) for index in next(ranges, ())]
        while pending:
            following = [executor.submit(dataset.__getitem__, index) for index in next(ranges, ())]
            batch = default_collate([item.result() for item in pending])
            yield tuple(tensor.pin_memory() for tensor in batch) if pin else batch
            pending = following


@torch.inference_mode()
def encode_patches(model, paths, mode, batch_size, workers, device, staging):
    cv2.setNumThreads(1)
    batches = patch_batches(PatchInputs(paths, mode), batch_size, workers, device.type == "cuda")
    vision = model.visual
    shape = (len(paths), SUPPORT_COUNT, vision.conv1.out_channels)
    layouts = {
        "primary": (shape, torch.float16),
        "secondary": (shape, torch.float16),
        "valid": (shape[:2], torch.bool),
    }
    if mode == "sketch":
        layouts["uniform"] = (shape, torch.float16)
    payload = {
        name: torch.from_file(str(staging / name), shared=True, size=int(np.prod(dims)), dtype=dtype).view(dims)
        for name, (dims, dtype) in layouts.items()
    }
    uniform_ids = UNIFORM_IDS.to(device)
    start = 0
    for views, ids, valid in tqdm(
        batches, total=(len(paths) + batch_size - 1) // batch_size, desc=f"CLIP patches {mode}", mininterval=1,
    ):
        pixels = views.flatten(0, 1).to(device=device, dtype=vision.conv1.weight.dtype, non_blocking=True)
        hidden = vision.ln_post(vision.transformer(vision._embeds(pixels))[:, 1:])
        hidden = hidden.reshape(len(views), 2, GRID * GRID, hidden.size(-1))
        ids = ids.to(device, non_blocking=True)
        selected = hidden.gather(2, ids[:, None, :, None].expand(-1, 2, -1, hidden.size(-1)))
        selected = F.normalize(selected.float(), dim=-1).half().cpu()
        end = start + len(views)
        payload["primary"][start:end] = selected[:, 0]
        payload["secondary"][start:end] = selected[:, 1]
        payload["valid"][start:end] = valid
        if mode == "sketch":
            payload["uniform"][start:end] = F.normalize(hidden[:, 0, uniform_ids].float(), dim=-1).half().cpu()
        start = end
    return payload


def load_patch_inputs(args, query_paths, photo_paths):
    device = torch.device(args.device)
    result = []
    model = None
    for mode, paths in (("photo", photo_paths), ("sketch", query_paths)):
        cache = cache_path(args, paths, mode)
        if cache.is_file():
            print(f"Cache {cache.name}", flush=True)
            result.append(torch.load(cache, weights_only=True, mmap=True))
            continue
        if model is None:
            model = load_frozen_clip(args.clip_checkpoint, device)
        args.cache.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="clip-patch-stage-", dir=args.cache) as directory:
            payload = encode_patches(
                model, paths, mode, args.patch_batch_size, args.workers, device, Path(directory),
            )
            save_tensor(payload, cache)
            del payload
        result.append(torch.load(cache, weights_only=True, mmap=True))
    return tuple(result)
