"""Transfer GEM's dense grounding features to sketch-to-gallery region matching."""

from functools import partial
import hashlib
from math import isqrt

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from .data import threaded_batches
from .encoding import global_cache_name, load_frozen_clip, save_tensor
from .gallery_diffusion import regional_cosines
from .mta import normalized_crops, pixel_lookup
from .multiscale import TileInputs
from .reranking import standardize
from .union_density import density_prefixes, density_rows, paired_density_inputs


def spatial_embeddings(vision, pixels):
    spatial = vision.conv1(pixels)
    height, width = spatial.shape[-2:]
    patches = spatial.flatten(2).transpose(1, 2)
    cls = vision.class_embedding.to(dtype=patches.dtype).expand(len(patches), 1, -1)
    position = vision.positional_embedding
    side = isqrt(len(position) - 1)
    if (height, width) != (side, side):
        grid = position[1:].float().T.reshape(1, -1, side, side)
        grid = F.interpolate(grid, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
        position = torch.cat((position[:1], grid.flatten(2)[0].T))
    tokens = torch.cat((cls, patches), dim=1) + position.to(patches.dtype)
    return vision.ln_pre(vision.patch_dropout(tokens))


def self_grounded_attention(path, values, scale):
    """One GEM clustering update, followed by assignment to the original values."""
    path = F.normalize(path, dim=-1, eps=1e-6)
    centers = F.scaled_dot_product_attention(path * scale, path, path, scale=1.)
    centers = F.normalize(centers, dim=-1, eps=1e-6)
    return F.scaled_dot_product_attention(centers * scale, centers, values, scale=1.)


def dense_branches(vision, pixels, include_grounding=True):
    """Share the original CLIP trunk and compare spatial features at the same scale."""
    blocks = vision.transformer.resblocks
    hidden = spatial_embeddings(vision, pixels)
    for block in blocks[:-3]:
        hidden = block(hidden)

    ground = hidden
    for index, block in enumerate(blocks[-3:]):
        attention = block.attn
        normalized = block.ln_1(hidden)
        qkv = F.linear(normalized, attention.in_proj_weight, attention.in_proj_bias).chunk(3, dim=-1)
        query, key, values = (
            heads.unflatten(-1, (attention.num_heads, -1)).transpose(1, 2) for heads in qkv
        )
        if include_grounding:
            inverse_temp = normalized.float().norm(dim=-1).mean(dim=1) * query.size(-1) ** -0.5
            scale = inverse_temp.to(query.dtype).view(-1, 1, 1, 1)
            update = sum(self_grounded_attention(path, values, scale) for path in (query, key, values)) / 3
            ground = ground + block.ls_1(attention.out_proj(update.transpose(1, 2).flatten(2)))

        if index == 2:
            clear = F.scaled_dot_product_attention(query, query, values)
            clear = attention.out_proj(clear.transpose(1, 2).flatten(2))
        else:
            vanilla = F.scaled_dot_product_attention(query, key, values)
            hidden = hidden + block.ls_1(attention.out_proj(vanilla.transpose(1, 2).flatten(2)))
            hidden = hidden + block.ls_2(block.mlp(block.ln_2(hidden)))

    branches = (clear, ground) if include_grounding else (clear,)
    return tuple(F.normalize((vision.ln_post(tokens[:, 1:]) @ vision.proj).float(), dim=-1) for tokens in branches)


def region_weights(patches, guides, ink_log_prior, logit_scale):
    logits = (guides @ patches.transpose(1, 2)) * logit_scale
    if ink_log_prior is not None:
        logits = logits + ink_log_prior
    return logits.softmax(dim=-1)


@torch.inference_mode()
def load_gem_regions(args, query_paths, photo_paths, queries, photos, sketch_words, photo_words, *, centers=False):
    device = torch.device(args.device)
    precision = "fp16" if device.type == "cuda" else "fp32"
    word_key = hashlib.sha256(args.vocabulary.read_bytes()).hexdigest()[:12]
    batch_size = args.batch_size if args.crop_batch_size is None else args.crop_batch_size
    workers = args.workers if args.crop_workers is None else args.crop_workers
    view = "clearclip-centers-v1" if centers else "gem-regions-v1"
    model = None
    result = []
    for mode, paths, embeddings, words in (
        ("sketch", query_paths, queries, sketch_words), ("photo", photo_paths, photos, photo_words),
    ):
        file = args.cache / global_cache_name(args.root, paths, mode, f"{view}-{word_key}", precision, args.clip_checkpoint)
        if not file.is_file():
            if model is None:
                model = load_frozen_clip(args.clip_checkpoint, device)
                lookup = pixel_lookup(device, model.visual.conv1.weight.dtype)
                logit_scale = model.logit_scale.exp().float()
                if centers:
                    axis = (torch.arange(14, device=device, dtype=torch.float32) + .5) / 14
                    grid = torch.cartesian_prod(axis, axis)
            inputs = TileInputs(paths)
            batches = threaded_batches(inputs, batch_size, workers, device.type == "cuda")
            shape = (len(paths), 5, 2) if centers else (len(paths), 2, 5, words.size(-1))
            stored = torch.empty(shape, dtype=torch.float32 if centers else torch.float16)
            start = 0
            for batch in tqdm(batches, total=(len(paths) + batch_size - 1) // batch_size, desc=f"{view} {mode}", mininterval=5):
                batch = batch.to(device=device, non_blocking=True)
                end = start + len(batch)
                features = embeddings[start:end].to(device)
                guides = words[(features @ words.T).topk(5, dim=-1).indices]
                ink_log_prior = None
                if mode == "sketch":
                    ink = (batch.float().mean(dim=1, keepdim=True) < 204).float()
                    density = F.avg_pool2d(ink, 32).flatten(1).sqrt()
                    ink_log_prior = (0.05 + density).log().unsqueeze(1)
                branches = dense_branches(model.visual, normalized_crops(batch, lookup), include_grounding=not centers)
                if centers:
                    regions = region_weights(branches[0], guides, ink_log_prior, logit_scale) @ grid
                else:
                    regions = torch.stack([
                        F.normalize(region_weights(patches, guides, ink_log_prior, logit_scale) @ patches, dim=-1)
                        for patches in branches
                    ], dim=1)
                stored[start:end].copy_(regions.to(device="cpu", dtype=stored.dtype))
                start = end
            save_tensor(stored, file)
        print(f"Cache {file.name}", flush=True)
        result.append(torch.load(file, map_location="cpu", mmap=True, weights_only=True))
    return tuple(result)


def gem_region_settings(queries, photos, contours, sketch_words, photo_words, maps, sketch_regions, photo_regions, score_batch):
    iso, context, initial, banks, graph = paired_density_inputs(queries, photos, contours, sketch_words, photo_words, maps, score_batch)
    yield "gem-region-global-control", partial(density_rows, initial, None, score_batch, None)
    prefix = density_prefixes(iso, context, (iso, context), initial, banks, graph, 1280, score_batch)
    yield "gem-region-graph-control", partial(density_rows, initial, prefix, score_batch, 0.8)
    dtype = torch.float16 if queries.is_cuda else torch.float32
    for index, method in enumerate(("clearclip448", "gem448")):
        sketch = sketch_regions[:, index].to(device=queries.device, dtype=dtype).contiguous()
        photo = photo_regions[:, index].to(device=photos.device, dtype=dtype).contiguous()
        regional = regional_cosines(sketch, photo, min(score_batch, 192))
        standardized = standardize(regional)
        for weight in (0.25, 0.75):
            fused = initial + weight * standardized
            yield f"{method}-regions-w{weight:g}", partial(density_rows, fused, None, score_batch, None)
        fused = initial + 0.25 * standardized
        prefix = density_prefixes(iso, context, (iso, context, regional), fused, banks, graph, 1280, score_batch)
        yield f"{method}-regions-graph", partial(density_rows, fused, prefix, score_batch, 0.8)
