"""Hide reference identities and evaluate how well each source predicts its paired target."""

import json

import torch
from torch.nn import functional as F

from .benchmark import merge_inputs
from .encoding import load_global_inputs
from .foundation import load_words
from .global_methods import centered
from .isoclip import isoclip_maps
from .semantic import read_words
from .semantic_benchmark import arguments
from .transfer_reconstruction import anchor_errors, nearest_anchors, transfer_coordinates


def pair_measures(predictions, truth):
    estimates = F.normalize(predictions, dim=-1)
    truth = F.normalize(truth, dim=-1)
    ranks = []
    for start in range(0, len(truth), 512):
        block = estimates[start:start + 512]
        ids = (block @ truth.T).topk(min(5, len(truth)), dim=1).indices
        paired = torch.arange(start, start + len(block), device=truth.device)[:, None]
        ranks.append(ids.eq(paired))
    match = torch.cat(ranks)
    return torch.stack(((estimates * truth).sum(dim=-1), (predictions - truth).square().sum(dim=-1),
                        match[:, 0].float(), match.any(dim=1).float()), dim=1).sum(dim=0)


def audit_bank(source, target, *, seed, center):
    generator = torch.Generator(device=source.device).manual_seed(seed)
    folds = torch.randperm(len(source), device=source.device, generator=generator).tensor_split(5)
    keys = ("cosine", "squared_distance", "recall_at_1", "recall_at_5")
    totals = {}
    fold_results = []
    for index, heldout in enumerate(folds):
        fit = torch.cat([indices for other, indices in enumerate(folds) if other != index])
        keys_fit, values_fit = source[fit], target[fit]
        probes, truth = source[heldout], target[heldout]
        if center:
            source_mean, target_mean = keys_fit.mean(dim=0), values_fit.mean(dim=0)
            keys_fit, probes = centered(keys_fit, source_mean), centered(probes, source_mean)
            values_fit, truth = centered(values_fit, target_mean), centered(truth, target_mean)
        ids = nearest_anchors(probes, keys_fit, 64)
        errors = anchor_errors(keys_fit, values_fit)
        shuffled = errors[torch.randperm(len(fit), device=fit.device, generator=generator)]
        observed = {}
        def record(name, predicted):
            sums = pair_measures(predicted, truth)
            observed[name] = dict(zip(keys, (sums / len(heldout)).tolist(), strict=True))
            totals[name] = sums if name not in totals else totals[name] + sums

        record("identity", probes)
        for count in (16, 64):
            for suffix, penalty, bank in (("plain", 0., errors), ("g0.1", .1, errors),
                                          ("g1", 1., errors), ("g1-shuffled", 1., shuffled)):
                pred = transfer_coordinates(probes, keys_fit, values_fit, ids[:, :count], .1, errors=bank,
                                            strength=penalty)
                record(f"k{count}-{suffix}", pred)
        fold_results.append({"fold": index, "held_out": len(heldout), "fit": len(fit), "methods": observed})
    return {"pairs": len(source), "seed": seed, "center_on_fit": center, "folds": fold_results,
            "pooled": {name: dict(zip(keys, (stats / len(source)).tolist(), strict=True)) for name, stats in totals.items()}}


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    query_paths, _ = merge_inputs(args.root, args.splits, "sketch")
    photo_paths, photos = merge_inputs(args.root, args.splits, "photo")
    _, native, contours = load_global_inputs(args, query_paths, photo_paths, True)
    transform = isoclip_maps(args, device)[(150, 50)]
    results = {}
    for i, split in enumerate(args.splits):
        ids, _ = photos[split]
        target = F.normalize(native[ids].to(device) @ transform, dim=-1)
        source = F.normalize(F.normalize(contours[ids].to(device) @ transform, dim=-1).mean(dim=1), dim=-1)
        results[split] = audit_bank(source, target, seed=823 + i, center=True)
    source, target = [v.to(device) for v in load_words(args, read_words(args.vocabulary))]
    results["public_prompts"] = audit_bank(source, target, seed=825, center=False)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "held-out-pairs.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run(arguments())
