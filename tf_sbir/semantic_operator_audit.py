"""Decompose an existing gallery neighbor update into scores available by editing only a query and the residual."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .conditional_completion_study import evaluate
from .encoding import save_tensor
from .gallery_diffusion import nearest_gallery
from .kaggle_unseen_features import DEFAULT_CACHE, load_split
from .native_block_probe import SEED, deterministic
from .replicated_simplex_study import evaluator
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import mean_percent
from .semantic_disputed_word_risk import query_posteriors, word_bank


DST = DEFAULT_CACHE.parent / "results" / "semantic-operator-identifiability"
SPLITS = ("sketchy_1", "sketchy_2", "tuberlin")
K = 64


def design(probability):
    # The omitted last word is one minus the sum of the other word probabilities.
    return torch.cat((torch.ones_like(probability[:, :1]), probability[:, :-1]), dim=1)


def closest_index(p, b):
    x = design(p)
    q, r = torch.linalg.qr(x, mode="reduced")
    u, s, vh = torch.linalg.svd(r, full_matrices=False, driver="gesvd")
    keep = s > max(x.shape) * torch.finfo(s.dtype).eps * s[0]
    basis = q @ u[:, keep]
    coefficients = basis.T @ b
    approx = basis @ coefficients
    affine = vh[keep].T @ (coefficients / s[keep, None])
    residual = b - approx
    err = (basis.T @ residual).norm().item() / residual.norm().item()
    recon = (x @ affine - approx).abs().max().item()
    if err > 5e-8 or recon > 5e-8:
        raise ValueError(f"Projection error: orthogonality {err}, reconstruction {recon}")
    return approx, affine, {"rank":int(keep.sum()), "basis_columns":int(x.shape[1]),
                            "singular_max":float(s[0]), "singular_min_kept":float(s[keep][-1]),
                            "relative_orthogonality_error":err, "affine_replay_max_abs":recon}


def word_fingerprint(words):
    return hashlib.sha256(("\n".join(words)+"\n").encode()).hexdigest()


@torch.inference_mode()
def prepare(args):
    started = time.monotonic()
    split, text, words = load_split(DEFAULT_CACHE, args.split)
    rgb = F.normalize(split.photo.float(), dim=1)
    p = word_bank(rgb, text[DEFAULT_PROMPTS[1]], split.scale).T.contiguous().double()
    if p.shape[1] != 601:
        raise ValueError("The diagnostic requires the existing 601-word index")
    p = p / p.sum(1, keepdim=True)
    historical, _ = nearest_gallery(rgb, 256)
    indices = historical[:, :K]
    if (indices == torch.arange(len(rgb),device=rgb.device)[:, None]).any():
        raise ValueError("A photo appeared among its own donors")
    b = torch.cat([p[part].mean(1) for part in indices.split(128)])
    approx, affine, info = closest_index(p, b)
    path = args.output / f"index-{args.split}.pt"
    signature = word_fingerprint(words)
    save_tensor({"p":p.cpu(), "b":b.cpu(), "f":approx.cpu(), "a":affine.cpu(),
                 "gallery_ids":split.gallery_ids, "word_fingerprint":signature, "revision":args.revision}, path)
    atomic_json(path.with_suffix(".json"), {"split":args.split, "revision":args.revision, "seed":SEED,
                "gallery":len(p), "k":K, "word_fingerprint":signature, "geometry":info,
                "seconds":time.monotonic()-started, "device":torch.cuda.get_device_name(), "sha256":checksum(path)})
    print(args.split, len(p), info, flush=True)


def energy_per_query(query, p, b, f):
    change, resid = b-p, b-f
    change -= change.mean(0,keepdim=True)
    resid -= resid.mean(0,keepdim=True)
    grams = [m.T @ m for m in (change, resid, change-resid)]
    e = torch.stack([((query @ gram) * query).sum(1) for gram in grams],dim=1)
    if (e[:, 0] <= 0).any():
        raise ValueError("Zero centered neighbor change")
    pythagoras = ((e[:, 0] - e[:, 1] - e[:, 2]).abs()/e[:, 0]).max().item()
    if pythagoras > 1e-6:
        raise ValueError(f"Per-query decomposition error {pythagoras}")
    return e.cpu(), pythagoras


def edited_scores(query, p, affine):
    coordinates = affine @ query.T
    last = torch.zeros_like(coordinates[:1])
    coefficients = torch.cat((coordinates[1:], last),dim=0) + coordinates[:1]
    return p @ coefficients


@torch.inference_mode()
def score(args):
    if args.part is None:
        raise ValueError("A score shard is required")
    started = time.monotonic()
    path = args.output / f"index-{args.split}.pt"
    rec = torch.load(path,mmap=True,map_location="cpu",weights_only=True)
    split,text,words = load_split(DEFAULT_CACHE,args.split)
    if not torch.equal(split.gallery_ids,rec["gallery_ids"]) or word_fingerprint(words) != rec["word_fingerprint"]:
        raise ValueError("Unmatched word coordinates or gallery order")
    ids = torch.arange(args.part, len(split.query_ids), 2)
    q = torch.cat([query_posteriors(split.sketch[sl], split.mirror[sl], text[DEFAULT_PROMPTS[0]], split.scale)
                   for sl in ids.split(384)])
    sample = replace(split, sketch_labels=split.sketch_labels[ids])
    p,b,f = [rec[key].cuda() for key in ("p","b","f")]
    delta_energy, err = energy_per_query(q.double(),p,b,f)
    check_q = q[:16].double()
    pair_error = (edited_scores(check_q,p,rec["a"].cuda()) - f @ check_q.T).abs().max().item()
    if pair_error > 1e-7:
        raise ValueError(f"Edited-query score replay error {pair_error}")
    banks = {"dot":p,"donor64":b,"query_projection":f,"dot_plus_residual":p+(b-f)}
    if args.split != "sketchy_1":
        other = torch.load(args.output/"index-sketchy_1.pt", mmap=True,map_location="cpu",weights_only=True)
        if other["word_fingerprint"] != rec["word_fingerprint"] or other["revision"] != rec["revision"]:
            raise ValueError("Invalid fixed Ext-1 calibration")
        banks["fixed_ext1_query_edit"] = design(p) @ other["a"].cuda()
    kd = evaluator(DEFAULT_CACHE.parent/"real-sketchy-smoke"/"evaluator.py")
    replay = PrefixReplay(kd,split.photo_labels,split.name)
    rows = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name,bank in banks.items():
            rows[name] = evaluate(sample,q,bank.float().contiguous(),replay,pool)
            print(args.split,args.part,name,mean_percent(rows[name]),flush=True)
    result = args.output / f"{args.split}-{args.part}.pt"
    save_tensor({"query_ids":split.query_ids[ids],"positions":ids,"labels":sample.sketch_labels,
                 "energy":delta_energy,"metrics":rows},result)
    atomic_json(result.with_suffix(".json"), {"split":args.split,"part":args.part,"revision":args.revision,
                "prepare_revision":rec["revision"],"seed":SEED,"queries":len(q),"gallery":len(p),
                "energy_check":err,"edited_query_replay_max_abs":pair_error,"sha256":checksum(result),
                "device":torch.cuda.get_device_name(),"seconds":time.monotonic()-started})


def report(args):
    result = {}
    for split in SPLITS:
        frames, metas = [], []
        for part in (0,1):
            path = args.output / f"{split}-{part}.pt"
            meta = json.loads(path.with_suffix(".json").read_bytes())
            if checksum(path) != meta["sha256"]:
                raise ValueError("Metric shard checksum differs")
            metas.append(meta)
            frames.append(torch.load(path,weights_only=True))
        order = torch.cat([a["positions"] for a in frames]).sort()
        n = {"sketchy_1":14179,"sketchy_2":12694,"tuberlin":2130}[split]
        if not torch.equal(order.values,torch.arange(n)):
            raise ValueError("Incomplete unseen query set")
        rows = {k:torch.cat([a["metrics"][k] for a in frames])[order.indices] for k in frames[0]["metrics"]}
        labels = torch.cat([a["labels"] for a in frames])[order.indices]
        energy = torch.cat([a["energy"] for a in frames])[order.indices]
        names = [("donor64","query_projection"),("query_projection","dot"),("dot_plus_residual","dot")]
        if "fixed_ext1_query_edit" in rows:
            names.append(("fixed_ext1_query_edit","dot"))
        paired = {}
        classes = labels.unique()
        draws = np.random.default_rng(SEED).integers(0,len(classes),(10_000,len(classes)))
        for x,y in names:
            change = 100 * (rows[x][:,0].double() - rows[y][:,0].double())
            means = np.array([change[labels==cls].mean().item() for cls in classes])
            paired[x+"_minus_"+y] = {"mean_AP_points":change.mean().item(),"macro_AP_points":float(means.mean()),
                                      "class_bootstrap95_macro":np.quantile(means[draws].mean(1),[.025,.975]).tolist()}
        result[split] = {"queries":n,"classes":len(classes),"metrics":{k:mean_percent(v) for k,v in rows.items()},
                         "residual_fraction_centered_squared_change":float(energy[:,1].sum()/energy[:,0].sum()),
                         "median_query_residual_fraction":float((energy[:,1]/energy[:,0]).quantile(.5)),
                         "paired":paired,"index":json.loads((args.output/f"index-{split}.json").read_bytes()),
                         "shards":metas}
    atomic_json(args.output/"operator-summary.json",{"revision":args.revision,"seed":SEED,"splits":result})
    print(json.dumps({s:{k:r[k] for k in ("metrics","residual_fraction_centered_squared_change","median_query_residual_fraction")}
                      for s,r in result.items()}),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare","score","report"))
    parser.add_argument("--split",choices=SPLITS)
    parser.add_argument("--part",type=int,choices=(0,1))
    parser.add_argument("--revision",required=True)
    parser.add_argument("--output",type=Path,default=DST)
    args = parser.parse_args()
    deterministic()
    if args.stage != "report" and args.split is None:
        parser.error("Specify a split for this stage")
    {"prepare":prepare,"score":score,"report":report}[args.stage](args)
