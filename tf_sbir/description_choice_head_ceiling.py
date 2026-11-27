"""Measure achievable full-gallery AP within nominated public nouns and meanings."""

import argparse
import json

import numpy as np
import open_clip
import torch

from .concept_benchmark import atomic_json, checksum
from .description_block_choices import NativeReads
from .description_block_results import Population
from .description_choice_blocks import OUT, PILOT, inputs
from .encoding import load_frozen_clip, save_tensor
from .native_block_probe import SEED, deterministic
from .wordnet_renaming_pilot import SPLITS
from .wordnet_renaming_text import Sense, describe, encode


def ordered_heads(scores, head_id, count, locked):
    order = scores.argsort(dim=1, descending=True, stable=True)
    ranks = torch.arange(order.shape[1], device=order.device).expand_as(order)
    first = torch.full((len(order), count), order.shape[1], dtype=torch.long, device=order.device)
    first.scatter_reduce_(1, head_id[order], ranks, reduce="amin")
    if (first == order.shape[1]).any():
        raise ValueError("Public noun missing from ranking")
    nominal = first.argsort(dim=1)
    nominated_senses = order.gather(1, first.gather(1, nominal[:, :2]))
    fixed_heads = head_id[locked]
    if (fixed_heads[:, 0] == fixed_heads[:, 1]).any():
        raise ValueError("The two archived nouns must be distinct")
    keep = ((nominal != fixed_heads[:, :1]) & (nominal != fixed_heads[:, 1:2]))
    tail = nominal[keep].reshape(len(order), count - 2)
    return torch.cat((fixed_heads, tail), dim=1), (nominated_senses == locked).all(1)


def baseline(name, query_ids):
    parts = []
    for i in (0, 1):
        p = PILOT / f"{name}-{i}.pt"
        meta = json.loads(p.with_suffix(".json").read_bytes())
        if meta["revision"] != "6fa17be0cdb20611110879dec3dafc1b31f0f798" or checksum(p) != meta["tensor_sha256"]:
            raise ValueError("Original baseline hash mismatch")
        parts.append(torch.load(p, mmap=True, weights_only=True))
    order = torch.cat([r["positions"] for r in parts]).argsort()
    if not torch.equal(torch.cat([r["query_ids"] for r in parts])[order], query_ids):
        raise ValueError("Baseline query order mismatch")
    return torch.cat([r["rows"]["semantic"] for r in parts])[order].double()


def uniform_curve(by_head, labels):
    count = len(by_head)
    # For a uniform m-subset, the probability that the best value is at sorted rank j
    # is choose(H-j-1,m-1)/choose(H,m) with j indexed from zero.
    probability = np.zeros((count, count), dtype=np.float64)
    for m in range(1, count + 1):
        j = np.arange(count - m)
        probability[m-1, :count-m+1] = (m / count) * np.r_[1., np.cumprod((count-j-m)/(count-j-1))]
    if not np.allclose(probability.sum(axis=1), 1., atol=1e-12):
        raise ValueError("Invalid random-prefix distribution")
    weights = labels.bincount(minlength=by_head.shape[1]).double() / len(labels)
    values = by_head.cpu().double().sort(dim=0, descending=True).values
    mean_sorted = (values * weights[None, :, None]).sum(1)
    return torch.from_numpy(probability) @ mean_sorted


def budget_crossings(curve, base_ap):
    result = {}
    for delta in (0, 4, 5):
        at = (100 * curve[:, 0] >= base_ap + delta).nonzero().flatten()
        result[str(delta)] = int(at[0]) + 1 if len(at) else None
    return result


@torch.inference_mode()
def run(name, revision, native_revision, choice_revision):
    manifest, _, _ = inputs()
    f = OUT / f"{name}-free.pt"
    meta = json.loads(f.with_suffix(".json").read_bytes())
    if meta["revision"] != choice_revision or meta["native_revision"] != native_revision or checksum(f) != meta["sha256"]:
        raise ValueError("Unverified nominations or KD utility table")
    data = torch.load(f, mmap=True, weights_only=True)
    labels, source, qi = (data[k] for k in ("classes", "source_ids", "query_ids"))
    base = baseline(name, qi)
    pop = Population(labels, source)
    utility = data["class_utility"].cuda().double()
    catalog = json.loads((PILOT / "inventory.json").read_bytes())
    senses = [Sense(**s) for s in catalog["senses"]]
    public, head = torch.tensor([s.head for s in senses], device="cuda").unique(return_inverse=True)
    n = len(public)
    if (len(senses), n) != (1135, 512) or len(utility) != len(senses):
        raise ValueError("Wrong dictionary")
    model = load_frozen_clip(manifest["checkpoint"], torch.device("cuda"))
    texts = encode(model, open_clip.get_tokenizer("ViT-B-32"), [describe("drawing", s.anonymous) for s in senses])
    del model
    by_head = torch.stack([utility[head == i].amax(0) for i in range(n)])
    uniform = uniform_curve(by_head, labels)
    all_bank = utility.amax(0)[labels.cuda()].mean(0).cpu()
    reference = 100 * base.mean(0)
    reports, rows = {}, {}
    depths = torch.tensor([12], device="cuda")
    for mode in ("sketch", "photo"):
        reads = NativeReads(mode, native_revision)
        for role in (("sketch", "sketch_two") if mode == "sketch" else ("source",)):
            ids = qi if mode == "sketch" else source
            locked = data[role][:, -1].long().cuda()
            cumulative = torch.zeros((n, 2), dtype=torch.float64, device="cuda")
            recorded, replay = [], []
            for start in range(0, len(qi), 256):
                sl = slice(start, start + 256)
                end = reads.read(ids[sl], depths, combined=role=="sketch_two")[2]
                ordered, agrees = ordered_heads(end @ texts.T, head, n, locked[sl])
                truth = labels[sl].cuda()[:, None]
                available = by_head[ordered, truth].cummax(1).values
                nominated = utility[locked[sl], truth]
                recorded.append(torch.stack((nominated[:, 0], nominated.amax(1), available[:, 0], available[:, 1]), 1).cpu())
                cumulative += available.sum(0)
                replay.append(agrees.cpu())
            curve = cumulative.cpu() / len(qi)
            if (curve.diff(dim=0) < -1e-12).any() or not torch.allclose(curve[-1], all_bank, atol=1e-10, rtol=0):
                raise ValueError("Incomplete or non-monotone oracle curve")
            if not torch.allclose(uniform[-1], all_bank, atol=1e-10, rtol=0):
                raise ValueError("Random and observed curves must reach the same ceiling")
            values = torch.cat(recorded)
            if (values[:, 3] + 1e-12 < values[:, 1]).any() or not torch.allclose(values[:, 3].mean(0), curve[1], atol=1e-10):
                raise ValueError("Relaxing sense selection reduced its attainable score")
            rows[role] = dict(metrics=values, locked_senses=locked.cpu(), replay=torch.cat(replay))
            reports[role] = {
                "means": (100 * values.mean(0)).tolist(), "two_head_minus_two_sense": pop.effect(values[:, 3]-values[:, 1]),
                "two_head_minus_baseline": pop.effect(values[:, 3]-base), "curve": (100 * curve).tolist(),
                "first_budgets": budget_crossings(curve, reference[0].item()),
                "replay_percent": 100 * torch.cat(replay).double().mean().item(),
            }
        del reads
    dest = OUT / f"head-ceiling-{name}.pt"
    save_tensor({"query_ids": qi, "gallery_ids": data["gallery_ids"], "source_ids": source, "classes": labels,
                 "baseline": base, "rows": rows, "public_ids": public.cpu(), "best_within_head": by_head.cpu()}, dest)
    atomic_json(dest.with_suffix(".json"), {
        "revision": revision, "native_revision": native_revision, "choice_revision": choice_revision, "seed": SEED,
        "split": name, "queries": len(qi), "gallery": len(data["gallery_ids"]), "senses": len(senses), "nouns": n,
        "columns": ["actual_first", "best_actual_two", "oracle_first_head", "oracle_first_two_heads"],
        "baseline": reference.tolist(), "all_bank": (100 * all_bank).tolist(), "reports": reports,
        "expected_uniform_curve": (100 * uniform).tolist(),
        "expected_uniform_first_budgets": budget_crossings(uniform, reference[0].item()),
        "raw_sha256": checksum(dest), "device": torch.cuda.get_device_name(),
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=SPLITS, required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--native-revision", required=True)
    p.add_argument("--choice-revision", required=True)
    args = p.parse_args()
    deterministic()
    run(args.split, args.revision, args.native_revision, args.choice_revision)
