"""Domain-independent foreground crops from frequency-tuned Lab saliency (Achanta et al., CVPR 2009)."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .benchmark import merge_inputs
from .crossdomain_inputs import runtime
from .data import SKETCHY_SPLITS
from .mta import load_view_features

VIEW_COUNT = 10
CACHE_TAG = "ftsal-lab224-square105-145-200-bipolar-v1"


def field(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    mean = lab.mean(axis=(0, 1))
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16
    filtered = cv2.sepFilter2D(lab, -1, kernel, kernel)
    saliency = np.linalg.norm(filtered - mean, axis=2)
    saliency -= saliency.min()
    return saliency / max(float(saliency.max()), 1e-6)


def salient_box(saliency):
    height, width = saliency.shape
    yy, xx = np.nonzero(saliency > 2 * saliency.mean())
    if not len(xx):
        return (width / 2, height / 2, max(width, height))
    # Ignore the extreme half-percent in each direction when converting the mask to a region.
    x0, x1 = np.quantile(xx, (0.005, 0.995))
    y0, y1 = np.quantile(yy, (0.005, 0.995))
    return ((x0 + x1 + 1) / 2, (y0 + y1 + 1) / 2, max(x1 - x0 + 1, y1 - y0 + 1, 4))


def patch(image, box, scale, color):
    x, y, side = box
    width = max(1, round(side * scale))
    start_x, start_y = round(x - width / 2), round(y - width / 2)
    matrix = np.array(
        [[width / 224, 0, start_x + (width / 224 - 1) / 2], [0, width / 224, start_y + (width / 224 - 1) / 2]],
        np.float32,
    )
    return cv2.warpAffine(
        image,
        matrix,
        (224, 224),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=color,
    )


def views(rgb):
    saliency = field(rgb)
    box = salient_box(saliency)
    perimeter = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
    color = tuple(int(x) for x in np.median(perimeter, axis=0))
    result = []
    for scale in (1.05, 1.45, 2.0):
        region = patch(rgb, box, scale, color)
        result.extend((region, region[:, ::-1]))
    gray = np.rint(255 * saliency).astype(np.uint8)
    small = patch(gray, box, 1.05, int(np.median(np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1])))))
    for sign in (255 - small, small):
        image = cv2.cvtColor(sign, cv2.COLOR_GRAY2RGB)
        result.extend((image, image[:, ::-1]))
    return torch.from_numpy(np.stack(result)).permute(0, 3, 1, 2)


class SalientViews(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            rgb = im.convert("RGB")
            scale = 320 / max(rgb.size)
            rgb = rgb.resize(tuple(max(1, round(v * scale)) for v in rgb.size), Image.Resampling.BILINEAR)
            return views(np.asarray(rgb))


def bank(args, paths, mode):
    return load_view_features(args, SalientViews(paths), VIEW_COUNT, mode, CACHE_TAG)["features"]


@torch.inference_mode()
def prepare(workspace, dataset):
    args = runtime(workspace, dataset, batch=64, workers=2, crop_batch=192)
    args.crop_workers = 6
    splits = SKETCHY_SPLITS if dataset == "sketchy" else ("tuberlin",)
    for mode in ("sketch", "photo"):
        paths, _ = merge_inputs(args.root, splits, mode)
        features = bank(args, paths, mode)
        if features.shape != (len(paths), VIEW_COUNT, 512):
            raise ValueError("Unexpected CLIP region features")
        print(dataset, mode, list(features.shape), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--dataset", required=True, choices=("sketchy", "tuberlin"))
    options = parser.parse_args()
    prepare(options.workspace, options.dataset)
