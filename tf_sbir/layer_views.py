"""Pool intermediate ViT-B/32 patch features for cross-domain retrieval."""

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .data import CLIP_MEAN, CLIP_STD, contour_views, normal_transform, threaded_batches
from .encoding import global_cache_name, load_frozen_clip, save_tensor

LAYERS = (6, 8, 10)


class LayerViews(Dataset):
    def __init__(self, paths, mode):
        self.paths = paths
        self.mode = mode
        self.transform = normal_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            rgb = source.convert("RGB")
        if self.mode == "photo":
            contour = next(contour_views(rgb))
            return torch.stack((self.transform(rgb), self.transform(contour)))
        return self.transform(rgb).unsqueeze(0)


def spatial_weights(vision, tokens, pixels, is_photo):
    projected = F.normalize((vision.ln_post(tokens) @ vision.proj).float(), dim=-1)
    affinity = (projected[:, 1:] * projected[:, :1]).sum(dim=-1)
    image_weights = (affinity / 0.12).softmax(dim=1)
    grayscale = pixels[:, :1].float() * CLIP_STD[0] + CLIP_MEAN[0]
    ink = F.avg_pool2d((grayscale < 0.80).float(), 32).flatten(1).sqrt()
    sketch_weights = image_weights * (0.05 + ink)
    sketch_weights /= sketch_weights.sum(dim=1, keepdim=True)
    if is_photo:
        sketch_weights[::2] = image_weights[::2]
    return sketch_weights


@torch.inference_mode()
def encode_layer_views(vision, paths, mode, batch_size, workers, device):
    inputs = LayerViews(paths, mode)
    batches = threaded_batches(inputs, batch_size, workers, device.type == "cuda")
    encoded = []
    for views in tqdm(batches, total=(len(inputs) + batch_size - 1) // batch_size, desc=f"CLIP layers {mode}", mininterval=1.0):
        pixels = views.flatten(0, 1).to(device=device, dtype=vision.conv1.weight.dtype, non_blocking=True)
        final, hidden_layers = vision.transformer.forward_intermediates(vision._embeds(pixels), indices=list(LAYERS))
        weights = spatial_weights(vision, final, pixels, mode == "photo").unsqueeze(-1)
        pooled = []
        for hidden in hidden_layers:
            tokens = F.normalize(hidden[:, 1:].float(), dim=-1)
            pooled.append(F.normalize((tokens * weights).sum(dim=1), dim=-1))
        features = torch.stack(pooled, dim=1).unflatten(0, views.shape[:2])
        encoded.append(features.cpu())
    return torch.cat(encoded)


def load_layer_inputs(args, query_paths, photo_paths):
    device = torch.device(args.device)
    precision = "fp16" if device.type == "cuda" else "fp32"
    model = None
    features = []
    for paths, mode in ((query_paths, "sketch"), (photo_paths, "photo")):
        name = global_cache_name(args.root, paths, mode, "layers-v1", precision, args.clip_checkpoint)
        path = args.cache / name
        if not path.is_file():
            if model is None:
                model = load_frozen_clip(args.clip_checkpoint, device)
            tensor = encode_layer_views(model.visual, paths, mode, args.batch_size, args.workers, device)
            save_tensor(tensor, path)
        features.append(torch.load(path, map_location="cpu", weights_only=True))
    queries, gallery = features
    return queries[:, 0], gallery[:, 0], gallery[:, 1]
