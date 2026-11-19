"""Unseen splits and preprocessing from the KD-SBIR evaluation protocol."""

from concurrent.futures import ThreadPoolExecutor
import glob
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, default_collate
from torchvision import transforms

SKETCHY_SPLITS = ("sketchy_1", "sketchy_2")
UNSEEN_CLASSES = {
    "sketchy_1": (
        "cup", "swan", "harp", "squirrel", "snail", "ray", "pineapple",
        "volcano", "rifle", "scissors", "parrot", "windmill", "teddy_bear",
        "tree", "wine_bottle", "deer", "chicken", "airplane", "wheelchair",
        "tank", "umbrella", "butterfly", "camel", "horse", "bell",
    ),
    "sketchy_2": (
        "bat", "cabin", "cow", "dolphin", "door", "giraffe", "helicopter",
        "mouse", "pear", "raccoon", "rhinoceros", "saw", "scissors",
        "seagull", "skyscraper", "songbird", "sword", "tree", "wheelchair",
        "windmill", "window",
    ),
    "quickdraw": (
        "alarm_clock", "axe", "airplane", "bat", "car", "bread", "bus",
        "cake", "campfire", "cruise ship", "dolphin", "eiffel tower", "eyeglasses",
        "horse", "flower", "giraffe", "hamburger", "helicopter", "hot-air_balloon",
        "kangaroo", "tree", "megaphone", "moon", "motorcycle", "palm tree",
        "parrot", "raccoon", "sailboat", "skyscraper", "windmill",
    ),
    "tuberlin": (
        "helicopter", "wrist-watch", "mermaid", "mosquito", "pear", "couch",
        "hammer", "airplane", "house", "baseball bat", "toilet", "panda",
        "butterfly", "mug", "wineglass", "motorbike", "eyeglasses",
        "hot air balloon", "cup", "skull", "truck", "palm tree", "cell phone",
        "horse", "sailboat", "suv", "church", "floor lamp", "bus", "tv",
    ),
}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CONTOURS = ((3, 55, 110), (7, 90, 180))


def normal_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def collect_paths(root, classes, mode):
    paths = []
    labels = []
    for label, category in enumerate(classes):
        members = [
            path for path in sorted(glob.glob(os.path.join(root, mode, category, "*")))
            if os.path.isfile(path)
        ]
        if not members:
            raise ValueError(f"Missing {mode} images for {category!r} in {root}")
        paths.extend(members)
        labels.extend([label] * len(members))
    return paths, torch.tensor(labels, dtype=torch.long)


def contour_views(image, size=224):
    gray = cv2.cvtColor(
        np.asarray(image.resize((size, size), Image.Resampling.BILINEAR)),
        cv2.COLOR_RGB2GRAY,
    )
    kernel = np.ones((2, 2), dtype=np.uint8)
    for blur, low, high in CONTOURS:
        smoothed = cv2.GaussianBlur(gray, (blur, blur), 0)
        edges = cv2.Canny(smoothed, low, high, L2gradient=True)
        yield Image.fromarray(255 - cv2.dilate(edges, kernel)).convert("RGB")


class Images(Dataset):
    def __init__(self, paths, transform, view="rgb"):
        self.paths = paths
        self.transform = transform
        self.view = view

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            image = source.convert("RGB")
        if self.view == "contours":
            return torch.stack([self.transform(view) for view in contour_views(image)])
        return self.transform(image)


def dataset_fingerprint(root, paths):
    digest = hashlib.sha256()
    for name in paths:
        path = Path(name)
        stat = path.stat()
        digest.update(
            f"{path.relative_to(root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()[:20]


def configure_image_worker(_worker_id):
    cv2.setNumThreads(1)


def threaded_batches(dataset, batch_size, workers, pin):
    configure_image_worker(None)
    ranges = iter(
        range(start, min(start + batch_size, len(dataset)))
        for start in range(0, len(dataset), batch_size)
    )
    if workers == 0:
        for indices in ranges:
            yield default_collate([dataset[index] for index in indices])
        return

    with ThreadPoolExecutor(max_workers=workers) as readers, ThreadPoolExecutor(max_workers=1) as prefetch:
        def read(indices):
            batch = default_collate(list(readers.map(dataset.__getitem__, indices)))
            if pin:
                batch = batch.pin_memory() if isinstance(batch, torch.Tensor) else tuple(t.pin_memory() for t in batch)
            return batch

        first = next(ranges, None)
        if first is None:
            return
        pending = prefetch.submit(read, first)
        for indices in ranges:
            batch = pending.result()
            pending = prefetch.submit(read, indices)
            yield batch
        yield pending.result()
