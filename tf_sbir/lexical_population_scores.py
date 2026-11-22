"""Redistribute only the gallery mass a sketch nominates outside its compact dictionary."""

import argparse
import math
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import digest
from .cross_view_region_cache import FAMILY as VIEWS
from .encoding import save_tensor
from .foundation import load_words
from .open_name_diagnosis import FAMILY
from .retrieval_diagnosis import global_inputs
from .semantic import read_words


@torch.inference_mode()
def proposals(w, dataset):
    cfg, paths, qs, _, _, native, _ = global_inputs(w, dataset)
    words = read_words(cfg.vocabulary)
    if len(words) != 20931:
        raise ValueError("Different public vocabulary")
    src = load_words(cfg, words)[0].cuda()
    _, scale = bare_words(cfg, words)
    crops = torch.load(w / "cache" / VIEWS / dataset / "views.pt", weights_only=True, mmap=True)["original_regions"]
    for s, (ids, _) in qs.items():
        file = w / "cache" / FAMILY / (s + "-proposals.pt")
        signature = digest({"paths": digest([str(paths[j]) for j in ids.tolist()]), "max_proposals": 64,
                            "vocabulary": checksum(cfg.vocabulary), "views": ["native","13 regions","equal mixture"]})
        if file.exists() and file.with_suffix(".json").exists() and read(file.with_suffix(".json"))["signature"] == signature:
            continue
        observed = {name: ([],[]) for name in ("whole","regions","both")}
        for ix in ids.split(72):
            full = (native[ix].cuda() @ src.T) * scale
            z = (crops[ix].cuda().float() @ src.T) * scale
            regions = torch.logaddexp(z[:,:9].logsumexp(1) - math.log(9), z[:,9:].logsumexp(1) - math.log(4)) - math.log(2)
            for name, x in (("whole", full), ("regions", regions), ("both", torch.logaddexp(full, regions) - math.log(2))):
                logs, indices = x.topk(64)
                observed[name][0].append(logs.cpu())
                observed[name][1].append(indices.cpu())
        save_tensor({name: (torch.cat(a),torch.cat(b)) for name,(a,b) in observed.items()},file)
        atomic_json(file.with_suffix(".json"), {"signature":signature,"queries":len(ids)})
        print("proposed",s,len(ids),flush=True)


def populations(book):
    full = book["total"].cuda()
    missing = book["missing"].cuda()
    single = book["single"].cuda()
    amount = missing.sum(0)
    rng = torch.Generator().manual_seed(61531)
    shuffled = torch.randperm(len(amount), generator=rng).to(amount.device)
    normalized = torch.where(amount[None] > 1e-7, missing / amount[None].clamp_min(1e-8), full)
    return {"full": full.T.contiguous(), "missing": missing.T.contiguous(), "single": single.T.contiguous(),
            "sham": (normalized[:,shuffled] * amount[None]).T.contiguous(), "groups":book["groups"].cuda()}


def corrections(incumbent, z, indices, bank):
    def mix(name):
        return (bank[name][indices] * z[:,:,None]).sum(1)
    proposed = mix("full")
    missing = mix("missing")
    single = mix("single")
    mass = missing.sum(1, keepdim=True)
    covered = (proposed - missing).clamp_min(0)
    return {"joint": missing - mass * incumbent, "single": single - single.sum(1,keepdim=True) * incumbent,
            "matched": mass * (proposed - incumbent), "raw": proposed - incumbent,
            "reverse": mass * (covered / covered.sum(1,keepdim=True).clamp_min(1e-12) - incumbent),
            "sham": mix("sham") - mass * incumbent, "mass":mass[:,0]}


def group_prob(base, groups):
    prob = torch.softmax(base, 1)
    grouped = prob.new_zeros((len(base),int(groups.max())+1))
    return grouped.index_add_(1, groups, prob)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),required=True)
    args=p.parse_args()
    proposals(args.workspace,args.dataset)
