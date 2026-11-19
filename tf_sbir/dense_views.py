"""Extract ClearCLIP patch embeddings from the same frozen ViT-B/32 weights."""

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from .data import Images, normal_transform, threaded_batches
from .encoding import global_cache_name, load_frozen_clip, save_tensor
from .semantic_transport import prepare_transport


def clearclip_patches(vision, pixels):
    """Use the last block's query–query attention without its residual or FFN."""
    stack = vision.transformer
    hidden, _ = stack.forward_intermediates(vision._embeds(pixels), indices=[len(stack.resblocks) - 2], stop_early=True)
    block = stack.resblocks[-1]
    attention = block.attn
    normalized = block.ln_1(hidden)
    query, _, values = F.linear(normalized, attention.in_proj_weight, attention.in_proj_bias).chunk(3, dim=-1)
    heads = attention.num_heads
    q_heads = query.unflatten(-1, (heads, -1)).transpose(1, 2)
    v_heads = values.unflatten(-1, (heads, -1)).transpose(1, 2)
    result = F.scaled_dot_product_attention(q_heads, q_heads, v_heads)
    result = attention.out_proj(result.transpose(1, 2).flatten(2))
    return F.normalize((vision.ln_post(result[:, 1:]) @ vision.proj).float(), dim=-1)


@torch.inference_mode()
def load_clearclip_gallery(args, paths):
    device = torch.device(args.device)
    precision = "fp16" if device.type == "cuda" else "fp32"
    file = args.cache / global_cache_name(args.root, paths, "photo", "clearclip-v1", precision, args.clip_checkpoint)
    if not file.is_file():
        model = load_frozen_clip(args.clip_checkpoint, device)
        inputs = Images(paths, normal_transform())
        batches = threaded_batches(inputs, args.batch_size, args.workers, device.type == "cuda")
        total = (len(paths) + args.batch_size - 1) // args.batch_size
        stored = None
        start = 0
        for batch in tqdm(batches, total=total, desc="ClearCLIP gallery patches", mininterval=1):
            pixels = batch.to(device, dtype=model.visual.conv1.weight.dtype, non_blocking=True)
            features = clearclip_patches(model.visual, pixels).half().cpu()
            if stored is None:
                stored = torch.empty((len(paths), *features.shape[1:]), dtype=torch.float16)
            end = start + len(features)
            stored[start:end].copy_(features)
            start = end
        save_tensor(stored, file)
    return torch.load(file, map_location="cpu", mmap=True, weights_only=True)


def dense_transport_settings(queries, photos, contours, sketch_words, photo_words, patch_features):
    query, gallery, offset, concepts, _ = prepare_transport(queries, photos, contours, sketch_words, photo_words)
    dtype = torch.float16 if queries.is_cuda else torch.float32
    concepts = concepts.to(dtype)
    patches = patch_features.to(device=queries.device, dtype=dtype)
    yield "transport-control", query, gallery, (offset, None)
    for weight in (0.25, 0.50):
        name = f"conditional-transport-clearclip-k640-w{weight:g}"
        yield name, query, gallery, (offset, concepts, patches, 640, weight)
