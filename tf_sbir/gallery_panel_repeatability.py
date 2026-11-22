"""Repeat the frozen cross-panel correction with independently drawn partitions of the same photos."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .encoding import save_tensor
from .gallery_depiction_study import PRIOR_RUN
from .gallery_region_panel_full import load_split_inputs
from .gallery_region_panel_study import parse
from .gallery_region_panels import FAMILY, FOLDS, compile_all, read_regions
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import stratified_sample
from .retrieval_evidence import load_kd, paired_delta


SEEDS = (72937, 197903, 991927)


def find_indices(w, split, size):
    folder = w/"cache"/FAMILY
    paths = {72937: folder/(split+"-donor-indices.pt")}
    for p in folder.glob(split+"-donor-*.json"):
        spec = read(p).get("seed_settings",{})
        if spec.get("seed") in SEEDS and spec["capacities"]==[size] and spec["types"]==["iid"]:
            if spec["seed"] in paths:
                raise ValueError("Ambiguous donor index")
            paths[spec["seed"]] = p.with_suffix(".pt")
    if set(paths) != set(SEEDS):
        raise ValueError("A prescribed seed is missing")
    return paths


def centered_cos(x, y):
    x = x-x.mean(1,keepdim=True)
    y = y-y.mean(1,keepdim=True)
    return (x*y).mean(1)/(x.square().mean(1)*y.square().mean(1)).clamp_min(1e-25).sqrt()


@torch.inference_mode()
def run(w, workers):
    root=w/"results"/FAMILY
    lock=read(root/"freeze.json")
    kind,size,mode,temp,weight=parse(lock["nominee"])
    if (kind,mode)!=("iid","cross_diff"):
        raise ValueError("This repeatability test requires the frozen iid score-ratio intervention")
    for seed in SEEDS[1:]:
        compile_all(w,"sketchy",seed=seed,capacities=(size,),kinds=("iid",))
    qs,gs,src,scale,view=load_split_inputs(w,"sketchy")
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Changed original evaluator")
    from .conditional_agreement_study import Replay
    captures,summary={},{}
    for s,(qi,labels) in qs.items():
        seedpaths=find_indices(w,s,size)
        banks=[]
        for seed in SEEDS:
            cpu=torch.load(seedpaths[seed],mmap=True,weights_only=True)["banks"]
            banks.append([tuple(v.cuda() for v in cpu[kind,size,j]) for j in range(FOLDS)])
        locked=torch.load(w/"cache"/PRIOR_RUN/"full"/(s+".pt"),mmap=True,weights_only=True)["regional_score"]
        idx=stratified_sample(labels,36,63241)
        reference=torch.load(root/("pilot-"+s+".pt"),mmap=True,weights_only=True)
        if not torch.equal(idx,reference["query_ids"]):
            raise ValueError("Pilot queries changed")
        truth=gs[s][1]
        gallery=truth.cuda()
        evaluator=Replay(kd,truth,s)
        metrics,diagnostics=defaultdict(list),defaultdict(list)
        worst=0.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for ix in idx.split(32):
                crop=view(qi[ix]).float().contiguous()
                base=locked[ix].cuda()
                r=base.argsort(dim=1,descending=True,stable=True)[:,:200]
                effects=[]
                for bank in banks:
                    scored,_=read_regions(crop,src,bank,scale,(temp,))
                    effects.append(scored[temp]["cross"]-scored[temp]["same"])
                scores={"regional":base, **{str(k):base+weight*e for k,e in zip(SEEDS,effects,strict=True)},
                        "integrated":base+weight*torch.stack(effects).mean(0)}
                orders={}
                for k,v in scores.items():
                    fast,order,_,_=map_values(v,labels[ix].cuda(),gallery,s)
                    real=evaluator.score(order,labels[ix],pool)
                    worst=max(worst,float((fast.cpu()-real).abs().max()))
                    metrics[k].append(real)
                    orders[k]=order[:,:100]
                for j,e in enumerate(effects):
                    for i,d in enumerate(effects[:j]):
                        left,right=map(str,(SEEDS[i],SEEDS[j]))
                        # Membership of the comparison pool is fixed by the unchanged baseline.
                        comparison=torch.zeros_like(base,dtype=torch.bool)
                        comparison.scatter_(1,orders[left],True)
                        overlap=comparison.gather(1,orders[right]).float().mean(1)
                        diagnostics[left+"/"+right].append(torch.stack((centered_cos(e,d),
                            centered_cos(e.gather(1,r),d.gather(1,r)),overlap),1).cpu())
        met={k:torch.cat(v) for k,v in metrics.items()}
        diag={k:torch.cat(v) for k,v in diagnostics.items()}
        reproduction=float((met[str(SEEDS[0])]-reference["metrics"][lock["nominee"]]).abs().max())
        if max(worst,reproduction)>2e-6:
            raise ValueError(f"Original ranking or pilot changed {worst}, {reproduction}")
        captures[s]={"metrics":met,"query_ids":idx,"query_labels":labels[idx],"paired_seed_diagnostics":diag}
        summary[s]={"queries":len(idx),"gallery":len(truth),"scores":{k:v.mean(0).tolist() for k,v in met.items()},
                    "delta_to_regional":{k:paired_delta(v,met["regional"],labels[idx]) for k,v in met.items()},
                    "agreement":{k:v.mean(0).tolist() for k,v in diag.items()},
                    "integrated_minus_first":paired_delta(met["integrated"],met[str(SEEDS[0])],labels[idx]),
                    "original_KD_error":worst,"original_pilot_error":reproduction}
        print("independent seeds",s,summary[s]["scores"],summary[s]["agreement"],flush=True)
    out=root/"seed-repeatability.pt"
    save_tensor(captures,out)
    atomic_json(out.with_suffix(".json"),{"freeze":lock["sha256"],"nominee":lock["nominee"],"seeds":SEEDS,
                "design":"two fresh partitions, same settings, average all three with equal weights; no winner selection",
                "agreement_fields":["centered correction correlation, all gallery","centered correction correlation, native top 200",
                                    "edited top100 photo overlap"],"splits":summary,"per_query_sha256":checksum(out)})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args()
    run(args.workspace,args.workers)
