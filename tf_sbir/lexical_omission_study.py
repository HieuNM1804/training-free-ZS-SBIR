"""Evaluate conservative open-vocabulary expansion with matched mass and coverage controls."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .encoding import save_tensor
from .lexical_population_index import TEMPS
from .lexical_population_scores import corrections, group_prob, populations
from .map_diagnosis import map_values
from .open_name_diagnosis import FAMILY
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import stratified_sample
from .retrieval_evidence import load_kd, paired_delta
from .verification_inputs import get_inputs

SPLITS = ("sketchy_1", "sketchy_2")
BASE = ("base", .5, "whole", 16, 1., 0.)
WEIGHTS = (.04, .15, .45, .85)


def key(x):
    return json.dumps(x, separators=(",", ":"))


def grid():
    return [BASE, *((m, temp, source, n, t, a) for temp in TEMPS for source in ("whole","regions","both")
                    for n in (4,16,64) for t in (.3,1.) for m in ("joint","raw","matched","single") for a in WEIGHTS)]


def pick(means, kinds, n):
    candidates = sorted((k for k in means if json.loads(k)[0] in kinds), key=means.get, reverse=True)
    return [tuple(json.loads(k)) for k in candidates[:n]]


def load_banks(w,s):
    folder = w / "cache" / FAMILY
    banks, checks = {}, {}
    for temp in TEMPS:
        f = folder / f"{s}-gallery-{temp:g}.pt"
        x = torch.load(f, mmap=True, weights_only=True)
        banks[temp] = populations(x)
        checks[str(temp)] = read(f.with_suffix(".json"))
    if not torch.equal(banks[TEMPS[0]]["groups"],banks[TEMPS[1]]["groups"]):
        raise ValueError("Changing response temperature altered the visual partition")
    prop = torch.load(folder / (s + "-proposals.pt"), weights_only=True, mmap=True)
    return banks, prop, checks


def measure(base_bank, y, g, ids, banks, proposed, cells, split, kd, primary, workers):
    groups = banks[TEMPS[0]]["groups"]
    fits = sorted({c[1:5] for c in cells})
    rows, details, err = defaultdict(list), defaultdict(list), {}
    replay = Replay(kd, g, split) if kd else None
    photo_labels = g.cuda()
    max_mass_error = 0.
    with (ThreadPoolExecutor(max_workers=workers) if kd else nullcontext()) as pool:
        for ix in ids.split(64):
            base = base_bank[ix].cuda()
            incumbent = group_prob(base, groups)
            labels = y[ix]
            for spec in fits:
                temp, pose, n, t = spec
                logs, words = (x[ix, :n].cuda() for x in proposed[pose])
                weights = (logs*t).softmax(1)
                changes = corrections(incumbent, weights, words, banks[temp])
                if kd and list(spec) == json.loads(primary)[1:5]:
                    details["estimated_missing_mass"].append(changes["mass"].cpu())
                for c in (c for c in cells if c[1:5] == spec):
                    mode, eta = c[0], c[5]
                    if mode == "base":
                        final = base
                    else:
                        new_mass = incumbent + eta * changes[mode]
                        max_mass_error = max(max_mass_error, float((new_mass.sum(1) - 1).abs().max()))
                        if float(new_mass.amin()) < -1e-6 or max_mass_error > 1e-4:
                            raise ValueError("The updated group measure violates its constraints")
                        delta = new_mass.clamp_min(1e-30).log() - incumbent.clamp_min(1e-30).log()
                        final = base + delta[:, groups]
                    quick, order, rel, _ = map_values(final, labels.cuda(), photo_labels, split)
                    name = key(c)
                    if kd:
                        actual = replay.score(order, labels, pool)
                        error = float((quick.cpu() - actual).abs().max())
                        if error > 2e-6:
                            raise ValueError("Original KD score is inconsistent")
                        err[name] = max(err.get(name, 0.), error)
                        if name == primary:
                            details["top25"].append(order[:, :25].int().cpu())
                            details["primary_top1"].append(rel[:,0].cpu())
                        if c == BASE:
                            details["base_top1"].append(rel[:,0].cpu())
                    else:
                        actual = quick.cpu()
                    rows[name].append(actual)
    return {k:torch.cat(v) for k,v in rows.items()}, {k:torch.cat(v) for k,v in details.items()}, err, max_mass_error


@torch.inference_mode()
def run(w, phase, workers):
    root = w / "results" / FAMILY
    rev = revision()
    spec = {"revision": rev, "grid": grid(), "pilot_seed": 63241, "per_class":36, "primary":"joint",
            "selection":"equal-weight full Sketchy mAP then mean precision", "incumbent": "regional self, 13 dual-view crops, 0.35"}
    signature = digest(spec)
    protocol = root / "protocol.json"
    if protocol.exists() and read(protocol)["sha256"] != signature:
        raise ValueError("Inconsistent experiment revision")
    atomic_json(protocol, spec | {"sha256":signature})
    kd, _ = load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("Different evaluator")
    for stage in {"pilot":("pilot",), "lock":("shortlist",), "audit":("final",)}[phase]:
        primary = None
        if stage == "pilot":
            cells = grid()
        elif stage == "shortlist":
            means = read(root / "pilot.json")["means"]
            cells = sorted({BASE, *pick(means, {"joint"}, 6), *pick(means, {"matched"}, 3),
                            *pick(means, {"raw"}, 4), *pick(means, {"single"}, 3)})
        else:
            lock = read(root / "freeze.json")
            if lock["revision"] != rev or lock["shortlist_sha256"] != checksum(root / "shortlist.json"):
                raise ValueError("The full Sketchy freeze does not match")
            chosen, raw, match, single = [tuple(lock[k]) for k in ("primary","raw","matched","single")]
            primary = key(chosen)
            cells = sorted({BASE, chosen, raw, match, single,
                            *[(kind, *chosen[1:]) for kind in ("raw","matched","single","sham","reverse")]})
        table = {}
        for ds in (("sketchy","tuberlin") if stage == "final" else ("sketchy",)):
            for s, (data, y, g) in get_inputs(w, ds).items():
                dest = root / ((s if stage == "final" else stage + "-" + s) + ".json")
                if dest.exists():
                    old = read(dest)
                    if old["signature"] != signature or (stage in ("pilot","final") and old["per_query_sha256"] != checksum(dest.with_suffix(".pt"))):
                        raise ValueError("Invalid checkpoint")
                    table[s] = old["values"]
                    continue
                ids = stratified_sample(y,36,63241) if stage == "pilot" else torch.arange(len(y))
                bank, prop, checks = load_banks(w,s)
                rows, extras, errors, maxerr = measure(data["regional_score"], y, g, ids, bank, prop, cells, s,
                                                       kd if stage == "final" else None, primary, workers)
                means = {name:v.mean(0).tolist() for name,v in rows.items()}
                info = {"signature":signature, "values":means, "queries": len(ids), "gallery":len(g),
                        "gallery_certificates": checks, "maximum_mass_error":maxerr}
                if stage in ("pilot","final"):
                    path = dest.with_suffix(".pt")
                    save_tensor({"metrics": rows, "query_labels":y[ids], "query_ids":ids} | extras, path)
                    info["per_query_sha256"] = checksum(path)
                if stage == "final":
                    base = rows[key(BASE)]
                    actual = rows[primary]
                    mass = extras["estimated_missing_mass"]
                    bins = torch.quantile(mass, torch.tensor([.25, .75]))
                    diag = {}
                    for nm, use in (("low_missing",mass <= bins[0]), ("high_missing",mass >= bins[1]),
                                    ("base_rank1_error",~extras["base_top1"]), ("base_rank1_hit",extras["base_top1"])):
                        diag[nm] = {"queries":int(use.sum()), "mean_missing_mass":float(mass[use].mean()),
                                    "change_pp":(100*(actual-base)[use].mean(0)).tolist()}
                    info.update({"protocol":{"split":s,"queries":len(y),"gallery":len(g),"classes":len(g.unique()),"checkpoint":CHECKPOINT,
                                             "vocabulary":20931, "gpu":torch.cuda.get_device_name(),"revision":rev,"kd_sha256":KD_FUNCTION_SHA,
                                             "map_k":200 if s=="sketchy_2" else 0,"p_k":200 if s=="sketchy_2" else 100},
                                 "primary":primary, "baseline":key(BASE), "best_contrast":key(raw), "matched_base":key(match),
                                 "one_anchor":key(single), "scores":{name:dict(zip(("mAP","precision"),v)) for name,v in means.items()},
                                 "vs_controls":{name:paired_delta(actual,v,y) for name,v in rows.items()}, "omission_diagnostic":diag,
                                 "kd_max_abs_error": errors, "original_kd_queries_per_condition":len(y),
                                 "top1_changes":{"fixed":int((~extras["base_top1"] & extras["primary_top1"]).sum()),
                                                 "broken":int((extras["base_top1"] & ~extras["primary_top1"]).sum())}})
                table[s] = means
                atomic_json(dest,info)
                win = pick(means,{"joint"},1)[0]
                print(stage,s,win,means[key(win)],"best",max(means,key=means.get),means[max(means,key=means.get)],flush=True)
        if stage != "final":
            means = {k:[sum(table[s][k][i] for s in SPLITS)/2 for i in (0,1)] for k in table[SPLITS[0]]}
            atomic_json(root/(stage+".json"),{"means":means,"splits":table})
            if stage == "shortlist":
                lock = {label:pick(means,{name},1)[0] for label,name in (("primary","joint"),("raw","raw"),("matched","matched"),("single","single"))}
                lock.update({"revision":rev,"shortlist_sha256":checksum(root/"shortlist.json")})
                atomic_json(root/"freeze.json",lock|{"sha256":digest(lock)})
                print("locked",lock,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--phase",required=True,choices=("pilot","lock","audit"))
    p.add_argument("--workers",type=int,default=10)
    args=p.parse_args()
    run(args.workspace,args.phase,args.workers)
