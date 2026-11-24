"""Predict the CLIP word distribution of a withheld sketch strip from a single photo."""
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .benchmark import merge_inputs
from .concept_benchmark import atomic_json
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import revision
from .crossdomain_inputs import runtime
from .data import Images, SKETCHY_SPLITS, configure_image_worker, normal_transform
from .depiction_path_diagnostic import class_interval
from .encoding import load_frozen_clip, save_tensor
from .regional_counterfactual_substitution import assess, input_tokens
from .semantic import read_words


NAMES = ("native", "bridge", "swapped_native", "swapped_bridge", "photo_alone", "sketch_alone")


def load_photo_strips(paths, vision):
    loader = DataLoader(Images(paths, normal_transform()), batch_size=64, num_workers=2,
                        pin_memory=True, worker_init_fn=configure_image_worker,
                        generator=torch.Generator().manual_seed(42))
    rows = []
    for images in loader:
        images = images.cuda(non_blocking=True).to(vision.conv1.weight.dtype)
        versions = [images]
        for axis in (3, 2):
            first, gap, last = images.split((96, 32, 96), dim=axis)
            versions.append(torch.cat((last, gap, first), dim=axis))
        tokens = vision._embeds(torch.cat(versions, 0))
        rows.append(torch.stack(tokens.chunk(3, 0), 1).cpu())
    return torch.cat(rows)


def log_forecast(vision, tokens, words, scale, mask=None):
    cls = vision.transformer(tokens, attn_mask=mask)[:, 0]
    feature = F.normalize((vision.ln_post(cls) @ vision.proj).float(), dim=-1)
    return (scale * (feature @ words.T)).log_softmax(-1)


def paired_strips(tokens, first, second):
    # Stack both forecast directions before encoding. All images share the native CLS.
    return torch.cat((tokens[:, first], tokens[:, second]), 0)


def forecast(vision, query, photos, words, scale, strips, bridge):
    per_axis = []
    for axis, (first, second) in enumerate(strips):
        batch = len(query)
        anchor = paired_strips(query, first, second)
        query_cls = query[:, :1].repeat(2, 1, 1)
        anchor_log = log_forecast(vision, torch.cat((query_cls, anchor), 1), words, scale)
        # The hidden observation stays outside every candidate representation.
        target = anchor_log.roll(batch, 0).exp()
        def loss(log_probability):
            return -(target * log_probability).sum(-1)
        losses = []
        for variant in (0, axis + 1):
            donor = paired_strips(photos[:, variant], second, first)
            joined = torch.cat((query_cls, anchor, donor), 1)
            for path_mask in (None, bridge):
                losses.append(loss(log_forecast(vision, joined, words, scale, path_mask)))
            if variant == 0:
                photo_loss = loss(log_forecast(vision, torch.cat((query_cls, donor), 1), words, scale))
        losses.extend((photo_loss, loss(anchor_log)))
        per_axis.append(torch.stack(losses, -1).reshape(2, batch, len(NAMES)).transpose(0, 1))
    return torch.stack(per_axis, 1)


def contrasts(meta, scores):
    relevant = 1 - meta[:, 3]
    foil = meta[:, 3]
    idx = torch.arange(len(meta))
    queries, counts = meta[:, 0].unique(return_counts=True)
    both = (meta[:, 0, None] == queries[counts == 2][None]).T
    labels = meta[both.long().argmax(1), 5]
    result = {}
    for control in NAMES[1:]:
        gain = scores[:, NAMES.index(control)] - scores[:, 0]
        qgain = (both.double() @ torch.stack((gain[idx, relevant], gain[idx, foil]), 1)) / 2
        endpoints = (qgain[:, 0], qgain[:, 1], qgain[:, 0] - qgain[:, 1])
        result[control] = {name: dict(mean_per_query=v.mean().item(), per_class=class_interval(v, labels, seed=42))
                           for name, v in zip(("relevant", "foil", "discrimination"), endpoints)}
    return result


@torch.inference_mode()
def run():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    work = Path("/content/sbir")
    out = work / "results/regional-hidden-strip-forecast"
    out.mkdir(parents=True, exist_ok=True)
    cfg = runtime(work, "sketchy")
    qpaths, splits = merge_inputs(cfg.root, SKETCHY_SPLITS, "sketch")
    ppaths, galleries = merge_inputs(cfg.root, SKETCHY_SPLITS, "photo")
    refs = {s:torch.load(work / "results/regional-counterfactual-substitution" / (s + ".pt"), weights_only=True)
            for s in splits}
    qs = torch.cat([r["meta"][:, 0] for r in refs.values()]).unique()
    ps = torch.cat([r["meta"][:, 1:3].flatten() for r in refs.values()]).unique()
    vision = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda")).visual
    vocab = read_words(cfg.vocabulary)
    assert len(vocab) == 20931
    words, scale = bare_words(cfg, vocab)
    queries = input_tokens([qpaths[i] for i in qs.tolist()], vision)
    photos = load_photo_strips([ppaths[i] for i in ps.tolist()], vision)
    y, x = torch.meshgrid(torch.arange(7, device="cuda"), torch.arange(7, device="cuda"), indexing="ij")
    strips = [((pos.flatten() < 3).nonzero().flatten() + 1, (pos.flatten() > 3).nonzero().flatten() + 1)
              for pos in (x, y)]
    mask = torch.zeros((43, 43), device="cuda", dtype=vision.conv1.weight.dtype)
    mask[1:22, 22:] = mask[22:, 1:22] = -torch.inf
    assert all(len(a) == len(b) == 21 and not torch.isin(a, b).any() for a, b in strips)
    torch.testing.assert_close(queries[0, 0], photos[0, 0, 0], atol=0, rtol=0)
    for split, ref in refs.items():
        meta = ref["meta"]
        assert torch.equal(ref["pi"], galleries[split][0]) and torch.isin(meta[:, 0], ref["qi"]).all()
        rows = []
        for batch in meta.split(32):
            query = queries[torch.searchsorted(qs, batch[:, 0].contiguous())].cuda()
            candidates = []
            for slot in (1, 2):
                candidate = photos[torch.searchsorted(ps, batch[:, slot].contiguous())].cuda()
                candidates.append(forecast(vision, query, candidate, words, scale, strips, mask))
            rows.append(torch.stack(candidates, -1).cpu())
        directional = torch.cat(rows).double()
        scores = directional.mean((1, 2))
        torch.testing.assert_close(scores[:, -1, 0], scores[:, -1, 1], atol=0, rtol=0)
        assessment = assess(meta, scores, NAMES)
        gains = contrasts(meta, scores)
        save_tensor(dict(meta=meta, qi=ref["qi"], pi=ref["pi"], directional=directional, scores=scores),
                    out / (split + ".pt"))
        atomic_json(out / (split + ".json"), dict(commit=revision(), seed=42, gpu=torch.cuda.get_device_name(),
                    arms=NAMES, assessment=assessment, forecast_gain=gains))
        print(split, gains["bridge"], flush=True)


if __name__ == "__main__":
    run()
