"""Aggregate the preregistered natural difference, rendering effect and resampling effect."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .concept_benchmark import atomic_json, checksum
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .drawing_size_probe import DST, SEED, SIZE
from .semantic_boxable_prompt_search import mean_percent


def assemble(root, name):
    shards, logs = [], []
    for part in (0, 1):
        path = root / f"{name}-{part}.pt"
        meta = json.loads(path.with_suffix(".json").read_bytes())
        if checksum(path) != meta["sha256"]:
            raise ValueError("Invalid per-query file")
        logs.append(meta)
        shards.append(torch.load(path, weights_only=True))
    row_order = torch.cat([a["rows"] for a in shards])
    rows = row_order.argsort()
    expected = {"sketchy_1":14179, "sketchy_2":12694}[name]
    if len(rows) != expected or not torch.equal(row_order[rows], torch.arange(expected)):
        raise ValueError("Incomplete unseen split")
    data = {key:torch.cat([a[key] for a in shards])[rows] for key in ("sources","labels","geometry","display","sides")}
    for group in ("metrics", "rhino_margins"):
        data[group] = {key:torch.cat([a[group][key] for a in shards])[rows] for key in shards[0][group]}
    return logs, data


def intervention(data, name):
    source, labels = data["sources"].numpy(), data["labels"].numpy()
    radii = data["geometry"][:,2].numpy()
    small, large = [], []
    for sid in np.unique(source):
        ids = np.flatnonzero((source == sid) & (radii > 0))
        if len(ids) > 1 and radii[ids].min() < radii[ids].max():
            small.append(ids[radii[ids].argmin()])
            large.append(ids[radii[ids].argmax()])
    if not np.array_equal(labels[small], labels[large]):
        raise ValueError("Mismatched source-level pairs")
    shared = set.intersection(*(set(UNSEEN_CLASSES[n]) for n in SKETCHY_SPLITS))
    classes = sorted(np.unique(labels[large]))
    effects = {}
    for reader in ("native", "semantic"):
        raw, reduced, restored = [data["metrics"][reader+"_"+k].double().numpy() for k in ("original","small","restored")]
        delta = np.stack((raw[large] - raw[small], raw[large] - restored[large],
                          restored[large] - reduced[large], reduced[large] - raw[small]), axis=1)
        if np.max(np.abs(delta[:,0]-delta[:,1:].sum(1))) > 1e-9:
            raise ValueError("Natural-size decomposition failed")
        by_class = 100*np.stack([delta[labels[large]==c].mean(0) for c in classes])
        effect_names = ("natural_gap", "resampling", "display", "remaining_drawing_gap")
        measures = {}
        for group, include in (("all", np.ones(len(classes),dtype=bool)),
                               ("nonoverlap", np.array([UNSEEN_CLASSES[name][c] not in shared for c in classes]))):
            subset = by_class[include]
            ids = np.random.default_rng(SEED).integers(0,len(subset),size=(10000,len(subset)))
            ci = np.quantile(subset[ids].mean(1), [.025,.975],axis=0)
            measures[group] = {key:{"mean_mAP_P":subset[:,j].mean(0).tolist(),
                                     "mAP_ci95":ci[:,j,0].tolist(),"P_ci95":ci[:,j,1].tolist(),
                                     "classes_AP_positive":int((subset[:,j,0]>0).sum())}
                               for j,key in enumerate(effect_names)}
        effects[reader] = measures
    def quantile(values):
        return np.quantile(values, [.05,.5,.95]).tolist()
    return {"subjects":len(large), "classes":len(classes), "effects":effects,
            "radius_ratio_large_over_small_q05_median_q95":quantile(radii[large]/radii[small]),
            "achieved_relative_radius_error_q05_median_q95":quantile(data["display"][large,2].numpy()/radii[small]-1),
            "centroid_shift_pixels_q05_median_q95":quantile(np.linalg.norm(
                (data["display"][large,:2]-data["geometry"][large,:2]).numpy(), axis=1))}


def report(args):
    payload = {"revision":args.revision, "seed":SEED, "splits":{}}
    for name in SKETCHY_SPLITS:
        logs, data = assemble(args.output, name)
        section = {"full":{k:mean_percent(v) for k,v in data["metrics"].items()},
                   "paired":intervention(data, name), "manifest":logs}
        if name == "sketchy_2":
            ids = [6602,6599,6600]
            section["rhino"] = {"rows":ids, "radius":data["geometry"][ids,2].tolist(),
                                "reduced_side":data["sides"][ids].tolist(),
                                "fixed_pair_margin":{k:v[ids].tolist() for k,v in data["rhino_margins"].items()}}
        payload["splits"][name] = section
    atomic_json(args.output/"summary.json",payload)
    print(json.dumps({n:{"full":d["full"],"source_pairs":d["paired"]["subjects"],
                         "geometry":{k:v for k,v in d["paired"].items() if "q05" in k},
                         "effects":{k:v["all"] for k,v in d["paired"]["effects"].items()},
                         "rhino":d.get("rhino")} for n,d in payload["splits"].items()}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision",required=True)
    parser.add_argument("--output",type=Path,default=DST)
    torch.set_num_threads(1)
    report(parser.parse_args())
