"""Read 13 regional CLS previews from one native CLIP trunk forward."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json, read
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .data import Images, configure_image_worker, normal_transform
from .encoding import load_frozen_clip, save_tensor
from .regional_interventions import SIDES, windows
from .retrieval_diagnosis import global_inputs

FAMILY = "preview-conditioned-region-acquisition"


def masks(device):
    grid = torch.arange(1, 50, device=device).reshape(7, 7)
    rows = [torch.ones(50, dtype=torch.bool, device=device)]
    for side in SIDES:
        for y, x in windows(side):
            keep = torch.zeros_like(rows[0])
            keep[0] = True
            keep[grid[y:y + side, x:x + side].flatten()] = True
            rows.append(keep)
    return torch.stack(rows)


def preview(visual, pixels, keep):
    hidden = visual._embeds(pixels)
    blocks = visual.transformer.resblocks
    for block in blocks[:-1]:
        hidden = block(hidden)
    block = blocks[-1]
    attn = block.attn
    q, k, v = F.linear(block.ln_1(hidden), attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
    q, k, v = [x.unflatten(-1, (attn.num_heads, -1)).transpose(1, 2) for x in (q[:, :1], k, v)]
    logits = (q.float() @ k.float().transpose(-1, -2)) * q.size(-1) ** -.5
    probability = logits.masked_fill(~keep[None, None], -torch.inf).softmax(-1)
    update = (probability @ v.float()).to(hidden.dtype).transpose(1, 2).flatten(2)
    cls = hidden[:, :1] + block.ls_1(attn.out_proj(update))
    cls = cls + block.ls_2(block.mlp(block.ln_2(cls)))
    return F.normalize((visual.ln_post(cls) @ visual.proj).float(), dim=-1)


@torch.inference_mode()
def run(w, ds):
    cfg, qpaths, _, ppaths, _, qraw, praw = global_inputs(w, ds)
    visual = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda")).visual
    keep = masks(visual.conv1.weight.device)
    folder = w / "cache" / FAMILY / ds
    for name, paths, raw in (("photo", ppaths, praw), ("sketch", qpaths, qraw)):
        file = folder / (name + ".pt")
        info = file.with_suffix(".json")
        signature = digest({"paths": [str(p) for p in paths], "checkpoint": CHECKPOINT,
                            "preview": "last block native CLS + selected keys, including CLS, 13 windows"})
        if file.exists() and info.exists() and read(info)["signature"] == signature:
            continue
        out = torch.empty((len(paths), 13, 512), dtype=torch.float16)
        dl = DataLoader(Images(paths, normal_transform(), "rgb"), batch_size=256, num_workers=5,
                        pin_memory=True, worker_init_fn=configure_image_worker, prefetch_factor=2)
        start = 0
        max_error = 0.
        for step, pixels in enumerate(dl):
            native = preview(visual, pixels.cuda(non_blocking=True).to(visual.conv1.weight.dtype), keep)
            gold = raw[start:start + len(pixels)].cuda()
            max_error = max(max_error, float((1 - (native[:, 0] * gold).sum(-1)).abs().max()))
            if max_error > .002:
                raise ValueError(f"Full mask must reproduce native CLIP, got {max_error}")
            out[start:start + len(pixels)] = native[:, 1:].half().cpu()
            start += len(pixels)
            if step % 40 == 0:
                print(ds, name, start, len(paths), max_error, flush=True)
        if start != len(paths):
            raise ValueError("Missing regional previews")
        save_tensor(out, file)
        atomic_json(info, {"signature": signature, "rows": start, "native_max_one_minus_cosine": max_error})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    p.add_argument("--dataset", choices=("sketchy", "tuberlin"), required=True)
    args = p.parse_args()
    run(args.workspace, args.dataset)
