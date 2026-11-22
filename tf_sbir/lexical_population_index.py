"""Build certified gallery populations for the public vocabulary without using any query."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .cluster_mine import cluster_assignments
from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import digest
from .cross_view_region_cache import FAMILY as REGIONAL
from .foundation import load_words
from .lexical_mass_packing import pack
from .open_name_diagnosis import FAMILY
from .regional_intervention_study import index as compact_index
from .retrieval_diagnosis import global_inputs
from .semantic import read_words
from .word_conditional_verifier import partition

TEMPS = (.5, 1.)


def groups_for(a):
    g = cluster_assignments(F.normalize(a - a.mean(0)), 64)
    sizes = g.bincount(minlength=64)
    keep = (sizes >= 8).nonzero().flatten()
    small = sizes[g] < 8
    if small.any():
        centers = a.new_zeros((64, a.size(1))).index_add_(0, g, a)
        g[small] = (F.normalize(a[small] - a.mean(0)) @ F.normalize(centers[keep] - sizes[keep, None] * a.mean(0)).T).argmax(1)
        # Restore original group ids after comparing the compacted eligible list.
        g[small] = keep[g[small]]
    return g.unique(sorted=True, return_inverse=True)[1]


def group_probabilities(a, text, scale, groups, folds, temp):
    n = int(groups.max()) + 1
    out = a.new_zeros((2 * n, len(text)), dtype=torch.float64)
    for start in range(0, len(a), 192):
        selection = slice(start, start + 192)
        evidence = ((a[selection] @ text.T) * (scale * temp)).softmax(1)
        out.index_add_(0, groups[selection] + folds[selection] * n, evidence.double())
    out = out.view(2, n, -1)
    out /= out.sum(1, keepdim=True)
    for fold in (0, 1):
        in_fold = folds == fold
        prior = groups[in_fold].bincount(minlength=n).double() / in_fold.sum()
        # One empirical background observation per panel stabilizes rare tails.
        size = in_fold.sum()
        out[fold] = (size * out[fold] + prior[:, None]) / (size + 1)
    return out


@torch.inference_mode()
def run(w, ds, workers):
    cfg, _, _, paths, gs, _, raw = global_inputs(w, ds)
    words = read_words(cfg.vocabulary)
    if len(words) != 20931:
        raise ValueError("Public vocabulary mismatch")
    _, target = load_words(cfg, words)
    target = target.cuda()
    _, scale = bare_words(cfg, words)
    folder = w / "cache" / FAMILY
    for split, (pi, _) in gs.items():
        gallery = raw[pi].cuda()
        strong = read(w / "results" / REGIONAL / (split + ".json"))
        photo_digest = digest([str(paths[i]) for i in pi.tolist()])
        signature = digest({"algorithm": "maximum-common-mass-two-panels-v1", "gallery": photo_digest, "vocabulary": checksum(cfg.vocabulary)})
        ident = digest({"signature": strong["signature"], "gallery": photo_digest, "vocabulary": checksum(cfg.vocabulary)})[:16]
        heads, _ = compact_index(w, split, gallery, target, scale, ident)
        groups = groups_for(gallery)
        folds = partition(groups, seed=91337) % 2
        for temp in TEMPS:
            dest = folder / f"{split}-gallery-{temp:g}.pt"
            info = dest.with_suffix(".json")
            if info.exists() and dest.exists() and read(info)["signature"] == signature:
                continue
            print("packing", split, temp, "words", len(words), flush=True)
            p = group_probabilities(gallery, target, scale, groups, folds, temp)
            data = pack(p, heads, workers)
            from .encoding import save_tensor
            save_tensor({k:v for k,v in data.items() if k != "checks"} | {"groups": groups.cpu()}, dest)
            atomic_json(info, {"signature": signature, "temperature": temp, "gallery": len(gallery), "checks": data["checks"]})
            print("certified", split, temp, data["checks"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    p.add_argument("--dataset", choices=("sketchy","tuberlin"), required=True)
    p.add_argument("--workers", type=int, default=9)
    args = p.parse_args()
    run(args.workspace, args.dataset, args.workers)
