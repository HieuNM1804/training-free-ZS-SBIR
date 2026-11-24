"""Trace cross-part paths on identical image tokens using the pretrained CLIP reader."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .benchmark import merge_inputs
from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .crossdomain_inputs import runtime
from .data import Images, SKETCHY_SPLITS, configure_image_worker, normal_transform
from .depiction_path_diagnostic import class_interval
from .encoding import load_frozen_clip, load_global_inputs, save_tensor
from .foundation import load_words
from .geometry_word_flat import KD_FUNCTION_SHA, PrefixReplay, load_kd, make_word_bank, score_rows
from .semantic import read_words
from .semantic_context_union import subsets


ARMS = ("read_only", "shared_feedback", "native_joint")


def graph_mask(arm, device, dtype):
    # Order: shared CLS, private CLS A, private CLS B, 21 A patches, 21 B patches.
    group = torch.tensor([0, 1, 2] + [1] * 21 + [2] * 21, device=device)
    allowed = group[:, None] == group[None, :]
    allowed[0, 3:] = True
    if arm == "shared_feedback":
        allowed[:, 1:3] = False
        allowed[1, 1] = allowed[2, 2] = True
        allowed[3:, 0] = True
    assert allowed.any(1).all() and not allowed[3:24, 24:].any()
    return torch.zeros_like(allowed, dtype=dtype).masked_fill(~allowed, -torch.inf)


def projected(vision, states):
    return vision.ln_post(states) @ vision.proj


def encode(paths, vision):
    loader = DataLoader(Images(paths, normal_transform()), batch_size=64,
                        num_workers=2, pin_memory=True, worker_init_fn=configure_image_worker,
                        generator=torch.Generator().manual_seed(42))
    dtype = vision.conv1.weight.dtype
    masks = {arm: graph_mask(arm, "cuda", dtype) for arm in ARMS[:2]}
    strips = subsets()[:4]
    rows = []
    private_replay = []
    for images in loader:
        tokens = vision._embeds(images.cuda(non_blocking=True).to(dtype))
        cls = tokens[:, :1]
        groups = [torch.cat((tokens[:, a[1:]], tokens[:, b[1:]]), 1)
                  for a, b in ((strips[0], strips[1]), (strips[2], strips[3]))]
        parts = torch.cat(groups, 0)
        shared = cls.repeat(2, 1, 1)
        packed = torch.cat((shared.expand(-1, 3, -1), parts), 1)
        features = []
        for arm in ARMS:
            if arm == "native_joint":
                states = vision.transformer(torch.cat((shared, parts), 1))
            else:
                states = vision.transformer(packed, attn_mask=masks[arm])
            if not rows and arm == "read_only":
                for i, part in enumerate(parts.split(21, dim=1)):
                    single = vision.transformer(torch.cat((shared[:4], part[:4]), 1))
                    reference = F.normalize(projected(vision, single[:, 0]).float())
                    observed = F.normalize(projected(vision, states[:4, i+1]).float())
                    private_replay.append((reference - observed).abs().max().item())
                assert max(private_replay) < .003, private_replay
            features.append(torch.stack(projected(vision, states[:, 0]).chunk(2), 1))
        rows.append(torch.stack(features, 1).cpu())
    return torch.cat(rows), private_replay


@torch.inference_mode()
def run():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    work = Path("/content/sbir")
    out = work / "results/semantic-context-communication"
    out.mkdir(parents=True, exist_ok=True)
    cfg = runtime(work, "sketchy")
    query_paths, queries = merge_inputs(cfg.root, SKETCHY_SPLITS, "sketch")
    photo_paths, galleries = merge_inputs(cfg.root, SKETCHY_SPLITS, "photo")
    refs = {s: torch.load(work / "results/semantic-context-union" / (s + ".pt"), weights_only=True)
            for s in queries}
    query_ids = torch.cat([r["qi"] for r in refs.values()]).unique()
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    features, private_replay = encode([query_paths[i] for i in query_ids.tolist()], model.visual)
    print("Encoded", len(features), "queries. Private lane replay:", private_replay, flush=True)
    source, target = [v.cuda() for v in load_words(cfg, read_words(cfg.vocabulary))]
    scale = model.logit_scale.exp().float()
    _, photos, _ = load_global_inputs(cfg, query_paths, photo_paths, False)
    kd, _ = load_kd(work)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() == KD_FUNCTION_SHA
    for split, ref in refs.items():
        qi, labels = ref["qi"], ref["labels"]
        pi, gallery_labels = galleries[split]
        assert torch.equal(pi, ref["pi"])
        bank = make_word_bank(F.normalize(photos[pi].cuda().float(), dim=-1), target, scale, 192)
        evaluator = PrefixReplay(kd, gallery_labels, split)
        positions = torch.searchsorted(query_ids, qi)
        rows = {arm: [] for arm in ARMS}
        with ThreadPoolExecutor(max_workers=2) as pool:
            for batch in torch.arange(len(qi)).split(32):
                visual = F.normalize(features[positions[batch]].cuda().float(), dim=-1)
                words = (scale * (visual @ source.T)).softmax(-1)
                scores = (words @ bank.T).double().square().mean(2)
                for i, arm in enumerate(ARMS):
                    rows[arm].append(score_rows(scores[:, i], labels[batch], gallery_labels.cuda(),
                                                split, evaluator, pool)[:, :2])
        rows = {arm: torch.cat(parts).double() for arm, parts in rows.items()}
        rows["additive"] = ref["rows"]["raw_additive/words"]
        drift = 100 * (rows["native_joint"] - ref["rows"]["joint_union/words"])
        assert drift.mean(0).abs().max() < .15, drift.mean(0)
        ordered = ("additive", *ARMS)
        contrasts = {}
        for previous, current in zip(ordered, ordered[1:]):
            delta = 100 * (rows[current] - rows[previous])
            contrasts[current + " minus " + previous] = dict(points=delta.mean(0).tolist(),
                class_ap=class_interval(delta[:, 0], labels, seed=42))
        save_tensor(dict(qi=qi, pi=pi, labels=labels, rows=rows, native_replay=drift), out / (split + ".pt"))
        atomic_json(out / (split + ".json"), dict(commit=revision(), gpu=torch.cuda.get_device_name(),
            seed=42, evaluator=KD_FUNCTION_SHA, queries=len(qi), gallery=len(pi), private_replay=private_replay,
            replay_points=drift.mean(0).tolist(), absolute_replay_points=drift.abs().mean(0).tolist(),
            metrics={k:(100 * x.mean(0)).tolist() for k, x in rows.items()}, contrasts=contrasts))
        print(split, {k:v["points"] for k,v in contrasts.items()}, flush=True)
        del bank


if __name__ == "__main__":
    run()
