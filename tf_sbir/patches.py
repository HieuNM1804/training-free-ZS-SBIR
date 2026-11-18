"""DINOv2 patch features aligned to a common RGB/contour image grid."""

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor
from tqdm.auto import tqdm

from .data import contour_views, dataset_fingerprint
from .encoding import save_tensor

DINO_MODEL = "facebook/dinov2-base"
DINO_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
SIZE = 224
PATCH = 14
GRID = SIZE // PATCH
SUPPORT_COUNT = 24
UNIFORM_IDS = torch.tensor([row * GRID + col for row in (1, 5, 9, 13) for col in (1, 3, 6, 9, 12, 14)])
MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def selected_patches(ink):
    """Keep the strongest patch per 2x2 group before limiting the support size."""
    mass = torch.from_numpy(ink.astype(np.float32)).view(GRID, PATCH, GRID, PATCH).mean(dim=(1, 3))
    groups = mass.reshape(GRID // 2, 2, GRID // 2, 2).permute(0, 2, 1, 3).reshape(-1, 4)
    values, offsets = groups.max(dim=1)
    order = values.argsort(descending=True, stable=True)[:SUPPORT_COUNT]
    row = 2 * (order // (GRID // 2)) + offsets[order] // 2
    col = 2 * (order % (GRID // 2)) + offsets[order] % 2
    valid = values[order] > 0
    if not valid.any():
        return UNIFORM_IDS, torch.ones(SUPPORT_COUNT, dtype=torch.bool)
    return row * GRID + col, valid


def dino_input(image):
    return (to_tensor(image.convert("RGB")) - MEAN) / STD


class PatchInputs(Dataset):
    def __init__(self, paths, mode):
        self.paths = paths
        self.mode = mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        cv2.setNumThreads(1)
        with Image.open(self.paths[index]) as source:
            rgb = source.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        if self.mode == "photo":
            fine, contour = contour_views(rgb, SIZE)
            ink = np.minimum(np.asarray(fine.convert("L")), np.asarray(contour.convert("L"))) < 224
            ids, valid = selected_patches(ink)
            return torch.stack((dino_input(rgb), dino_input(contour))), ids, valid

        gray = np.asarray(rgb.convert("L"))
        border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
        if np.median(border) < 128:
            gray = 255 - gray
            rgb = Image.fromarray(gray).convert("RGB")
        ids, valid = selected_patches(gray < 224)
        thicker = cv2.erode(gray, np.ones((3, 3), dtype=np.uint8))
        return torch.stack((dino_input(rgb), dino_input(Image.fromarray(thicker)))), ids, valid


def cache_path(args, paths, mode):
    manifest = dataset_fingerprint(args.root, paths)
    device = str(args.device).replace(":", "-")
    return args.cache / f"dino-v1-{DINO_REVISION[:8]}-{mode}-{device}-{manifest}.pt"


@torch.inference_mode()
def encode_patches(model, paths, mode, batch_size, workers, device, staging):
    batches = DataLoader(
        PatchInputs(paths, mode), batch_size=batch_size, num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    shape = (len(paths), SUPPORT_COUNT, model.config.hidden_size)
    layouts = {
        "primary": (shape, torch.float16),
        "secondary": (shape, torch.float16),
        "valid": (shape[:2], torch.bool),
    }
    if mode == "sketch":
        layouts["uniform"] = (shape, torch.float16)
    # Back the full-dataset buffer with files so concurrent inference need not
    # hold multiple gigabytes of patch tensors in physical memory.
    payload = {
        name: torch.from_file(str(staging / name), shared=True, size=int(np.prod(dims)), dtype=dtype).view(dims)
        for name, (dims, dtype) in layouts.items()
    }
    dtype = next(model.parameters()).dtype
    uniform_ids = UNIFORM_IDS.to(device)
    start = 0
    for views, ids, valid in tqdm(batches, desc=f"DINO patches {mode}"):
        pixels = views.flatten(0, 1).to(device=device, dtype=dtype, non_blocking=True)
        hidden = model(pixel_values=pixels).last_hidden_state[:, 1:]
        hidden = hidden.reshape(len(views), 2, GRID * GRID, hidden.size(-1))
        ids = ids.to(device, non_blocking=True)
        selected = hidden.gather(2, ids[:, None, :, None].expand(-1, 2, -1, hidden.size(-1)))
        selected = F.normalize(selected.float(), dim=-1).half().cpu()
        end = start + len(views)
        payload["primary"][start:end] = selected[:, 0]
        payload["secondary"][start:end] = selected[:, 1]
        payload["valid"][start:end] = valid
        if mode == "sketch":
            uniform = F.normalize(hidden[:, 0, uniform_ids].float(), dim=-1).half().cpu()
            payload["uniform"][start:end] = uniform
        start = end
    return payload


def load_patch_inputs(args, query_paths, photo_paths):
    from transformers import Dinov2Model

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
            model = Dinov2Model.from_pretrained(
                DINO_MODEL, revision=DINO_REVISION, use_safetensors=True,
                dtype=torch.float16 if device.type == "cuda" else torch.float32,
            ).to(device).eval().requires_grad_(False)
        args.cache.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="dino-stage-", dir=args.cache) as directory:
            payload = encode_patches(
                model, paths, mode, args.patch_batch_size, args.workers, device, Path(directory),
            )
            save_tensor(payload, cache)
            del payload
        result.append(torch.load(cache, weights_only=True, mmap=True))
    return tuple(result)
