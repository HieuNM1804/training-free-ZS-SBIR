"""Use TokenCut proposals with ReCLIP's crop and background-blur views."""

from functools import partial

from PIL import Image, ImageFilter
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from tqdm.auto import tqdm

from .data import threaded_batches
from .dense_views import tokencut_mask
from .encoding import global_cache_name, load_frozen_clip, save_tensor
from .gallery_diffusion import diffusion_rows, full_gallery_inputs, linear_fusion_rows, mutual_transition, regional_cosines
from .mta import normalized_crops, pixel_lookup


def token_boxes(mask):
    """Return bounding boxes with exclusive bottom and right grid coordinates."""
    side = int(mask.size(1) ** 0.5)
    ids = torch.arange(mask.size(1), device=mask.device)
    y, x = ids.div(side, rounding_mode="floor"), ids % side
    left = torch.where(mask, x, side).amin(dim=1)
    top = torch.where(mask, y, side).amin(dim=1)
    right = torch.where(mask, x + 1, 0).amax(dim=1)
    bottom = torch.where(mask, y + 1, 0).amax(dim=1)
    return torch.stack((left, top, right, bottom), dim=1)


def isolated_views(image, box):
    crop = image.crop(box).resize((224, 224), Image.Resampling.BICUBIC)
    background = image.filter(ImageFilter.GaussianBlur(16))
    background.paste(image.crop(box), box[:2])
    return torch.stack((pil_to_tensor(crop), pil_to_tensor(background)))


class RegionInputs(Dataset):
    def __init__(self, paths, boxes):
        self.paths, self.boxes = paths, boxes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            image = source.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        return isolated_views(image, self.boxes[index])


@torch.inference_mode()
def load_isolated_regions(args, paths, patch_features):
    device = torch.device(args.device)
    precision = "fp16" if device.type == "cuda" else "fp32"
    name = global_cache_name(args.root, paths, "photo", "tokencut-reclip-v1", precision, args.clip_checkpoint)
    filename = args.cache / name
    if not filename.is_file():
        boxes = []
        for block in tqdm(patch_features.split(256), desc="TokenCut proposals"):
            features = F.normalize(block.to(device=device, dtype=torch.float32), dim=-1)
            mask = tokencut_mask(features, 0.65)
            boxes.extend((32 * token_boxes(mask)).cpu().tolist())
        model = load_frozen_clip(args.clip_checkpoint, device)
        lookup = pixel_lookup(device, model.visual.conv1.weight.dtype)
        batch_size = max(1, (args.crop_batch_size or args.batch_size) // 2)
        workers = args.workers if args.crop_workers is None else args.crop_workers
        stream = threaded_batches(RegionInputs(paths, boxes), batch_size, workers, device.type == "cuda")
        total = (len(paths) + batch_size - 1) // batch_size
        encoded = []
        for batch in tqdm(stream, total=total, desc="CLIP isolated regions"):
            pixels = normalized_crops(batch.flatten(0, 1), lookup)
            features = F.normalize(model.encode_image(pixels).float(), dim=-1)
            encoded.append(features.unflatten(0, (-1, 2)).half().cpu())
        save_tensor(torch.cat(encoded), filename)
    print(f"Cache {filename.name}", flush=True)
    return torch.load(filename, map_location="cpu", mmap=True, weights_only=True)


def isolated_region_settings(queries, photos, contours, sketch_words, photo_words, regions, score_batch):
    base, native, gallery, concepts, targets = full_gallery_inputs(queries, photos, contours, sketch_words, photo_words, score_batch)
    reference = mutual_transition(gallery, 160)
    yield "reclip-transport-control", partial(linear_fusion_rows, base, native, score_batch, 0.25)
    yield "reclip-diffusion-control", partial(diffusion_rows, base, native, reference, score_batch, 0.9, 320, "regional")
    del native
    isolated = F.normalize(regions.to(device=photos.device, dtype=torch.float32), dim=-1)
    for name, proposal in (("crop", isolated[:, 0]), ("crop-blur", isolated.mean(dim=1))):
        combined = torch.cat((targets, proposal[:, None].to(targets.dtype)), dim=1)
        scores = regional_cosines(concepts, combined, min(score_batch, 256))
        yield f"reclip-{name}-full", partial(linear_fusion_rows, base, scores, score_batch, 0.25)
        yield f"reclip-{name}-diffusion", partial(diffusion_rows, base, scores, reference, score_batch, 0.9, 320, "regional")
        if name == "crop-blur":
            proposal_geometry = F.normalize(proposal, dim=-1)
            geometry = torch.cat((gallery * (0.65 ** 0.5), proposal_geometry * (0.35 ** 0.5)), dim=1)
            graph = mutual_transition(geometry, 160)
            yield "reclip-crop-blur-graph", partial(diffusion_rows, base, scores, graph, score_batch, 0.9, 320, "regional")
