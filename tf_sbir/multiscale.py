"""S²-Wrapper patch reassembly and full-gallery retrieval at two scales."""

from functools import partial

from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from tqdm.auto import tqdm

from .data import threaded_batches
from .dense_views import clearclip_patches, tokencut_mask
from .encoding import global_cache_name, load_frozen_clip, save_tensor
from .gallery_diffusion import diffusion_rows, full_gallery_inputs, linear_fusion_rows, mutual_transition, regional_cosines
from .mta import normalized_crops, pixel_lookup
from .reranking import standardize


class TileInputs(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            image = source.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
        return pil_to_tensor(image)


def split_tiles(pixels, size=224):
    tiles = pixels.unfold(-2, size, size).unfold(3, size, size)
    return tiles.permute(0, 2, 3, 1, 4, 5).flatten(0, 2)


def merge_tile_patches(tiles):
    """Restore spatial neighbors across tile boundaries before average pooling."""
    side = int(tiles.size(1) ** 0.5)
    grid = tiles.reshape(-1, 2, 2, side, side, tiles.size(2))
    stitched = grid.permute(0, 5, 1, 3, 2, 4).reshape(-1, tiles.size(2), 2 * side, 2 * side)
    return F.avg_pool2d(stitched, 2).flatten(2).transpose(1, 2).contiguous()


def tiled_batches(model, paths, batch_size, workers, device, description=None):
    lookup = pixel_lookup(device, model.visual.conv1.weight.dtype)
    images_per_batch = max(1, batch_size // 4)
    batches = threaded_batches(TileInputs(paths), images_per_batch, workers, device.type == "cuda")
    if description is not None:
        batches = tqdm(batches, total=(len(paths) + images_per_batch - 1) // images_per_batch, desc=description)
    for uint8 in batches:
        pixels = normalized_crops(uint8, lookup)
        yield merge_tile_patches(clearclip_patches(model.visual, split_tiles(pixels)))


@torch.inference_mode()
def load_multiscale_patches(args, query_paths, photo_paths):
    device = torch.device(args.device)
    precision = "fp16" if device.type == "cuda" else "fp32"
    model = None
    result = []
    for mode, paths in (("sketch", query_paths), ("photo", photo_paths)):
        file = args.cache / global_cache_name(args.root, paths, mode, "clearclip-tiles-v1", precision, args.clip_checkpoint)
        if not file.is_file():
            if model is None:
                model = load_frozen_clip(args.clip_checkpoint, device)
            batch_size = args.crop_batch_size if args.crop_batch_size is not None else args.batch_size
            workers = args.crop_workers if args.crop_workers is not None else args.workers
            start = 0
            stored = None
            for band in tiled_batches(model, paths, batch_size, workers, device, f"S² CLIP {mode}"):
                # Sketches only need the pooled second-scale vector for the scale cosine.
                band = F.normalize(band, dim=-1).mean(dim=1) if mode == "sketch" else band
                if stored is None:
                    stored = torch.empty((len(paths), *band.shape[1:]), dtype=torch.float16)
                end = start + len(band)
                stored[start:end].copy_(band.to(dtype=torch.float16, device="cpu"))
                start = end
            save_tensor(stored, file)
        print(f"Cache {file.name}", flush=True)
        result.append(torch.load(file, map_location="cpu", mmap=True, weights_only=True))
    return tuple(result)


def foreground_regions(native, fine, device):
    """Use identical TokenCut pooling for single-scale and S² foregrounds."""
    original, multiscale, full = [], [], []
    for whole, tiles in zip(native.split(256), fine.split(256), strict=True):
        whole, tiles = (F.normalize(group.to(device=device, dtype=torch.float32), dim=-1) for group in (whole, tiles))
        single_mask = tokencut_mask(whole, 0.65)
        joint = torch.cat((whole, tiles), dim=-1) * (0.5 ** 0.5)
        joint_mask = tokencut_mask(joint, 0.65)
        original.append(F.normalize((whole * single_mask.unsqueeze(-1)).sum(dim=1), dim=-1))
        pooled = (whole + tiles) * joint_mask.unsqueeze(-1)
        multiscale.append(F.normalize(pooled.sum(dim=1), dim=-1))
        full.append(F.normalize(tiles.mean(dim=1), dim=-1))
    return torch.cat(original), torch.cat(multiscale), torch.cat(full)


def tiled_score_rows(base, region, sketch_band, photo_band, batch_size, scale_weight):
    for initial, cue, sketch in zip(base.split(batch_size), region.split(batch_size), sketch_band.split(batch_size), strict=True):
        scale = standardize(sketch @ photo_band.T)
        score = standardize(initial) + 0.25 * standardize(cue) + scale_weight * scale
        yield from (0.15 * score).tanh().cpu()


def multiscale_settings(queries, photos, contours, sketch_words, photo_words, native, bands, score_batch):
    fine_sketch, fine_photo = bands
    native, multiscale, photo_band = foreground_regions(native, fine_photo, photos.device)
    sketch_band = F.normalize(fine_sketch.to(device=queries.device, dtype=torch.float32), dim=-1)
    base, regional, gallery, concepts, targets = full_gallery_inputs(queries, photos, contours, sketch_words, photo_words, score_batch)
    yield "s2-full-no-foreground", partial(linear_fusion_rows, base, regional, score_batch, 0.25)
    reference_graph = mutual_transition(gallery, 160)
    yield "s2-diffusion-control", partial(diffusion_rows, base, regional, reference_graph, score_batch, 0.9, 320, "regional")
    del regional, reference_graph
    for name, foreground in (("native", native), ("tiled", multiscale)):
        augmented = torch.cat((targets, foreground[:, None].to(targets.dtype)), dim=1)
        region = regional_cosines(concepts, augmented, min(score_batch, 256))
        yield f"s2-{name}-foreground", partial(linear_fusion_rows, base, region, score_batch, 0.25)
        if name == "tiled":
            for weight in (0.15, 0.35):
                yield f"s2-full-scale-w{weight:g}", partial(tiled_score_rows, base, region, sketch_band, photo_band, score_batch, weight)
        for graph_name, weight in (("rgb", 0.0), ("foreground", 0.35)):
            geometry = torch.cat((gallery * ((1 - weight) ** 0.5), foreground * (weight ** 0.5)), dim=-1)
            transition = mutual_transition(geometry, 160)
            yield f"s2-{name}-{graph_name}-diffusion", partial(diffusion_rows, base, region, transition, score_batch, 0.9, 320, "regional")
