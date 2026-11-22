"""Check whether real whole-image/region associations and real category labels agree on a common state."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .crossdomain_benchmark import revision
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .integrated_observable_operators import build
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .response_geometry_views import FAMILY as VIEW_CACHE, read_photos, study_base
from .retrieval_diagnosis import stratified_sample
from .retrieval_evidence import load_kd, paired_delta
from .shared_semantic_operators import FAMILY
from .visual_observable_study import parse


SEEDS=(59731,91995,137677)
PAIRING=("actual","within_visual_group","global")


def pair_order(scores,labels):
    all_rows=[]
    for c in range(len(scores)):
        pos=scores[c,labels==c].contiguous()
        neg=scores[c,labels!=c].sort().values
        count=(torch.searchsorted(neg,pos,right=True)+torch.searchsorted(neg,pos,right=False)).double()/2
        all_rows.append(count.mean()/len(neg))
    return torch.stack(all_rows).cpu()


def read_folds(z,folds,t):
    result={k:torch.empty_like(z) for k in ("shared","raw","matched","rotated","linear")}
    energy=[]
    for fold in folds:
        read=z[:,fold.read_ids]
        center=read-read.mean(1,keepdim=True)
        B=fold.common.basis.unpack(center@fold.train_moments/read.shape[1])
        energy.append(fold.common.energy(B))
        pred=fold.read(z,(t,))
        for k in result:
            result[k][:,fold.target_ids]=pred[k if k in ("raw","linear") else (k,t)]
    return result,torch.stack(energy,dim=1)


def exact(kd,s,z,y,g,pool):
    replay=Replay(kd,g,s)
    parts=[]
    bits=[]
    errors=[]
    for j in range(0,len(y),64):
        sl=slice(j,j+64)
        val,r,rel,_=map_values(z[sl],y[sl].cuda(),g.cuda(),s)
        actual=replay.score(r,y[sl],pool)
        errors.append(float((actual-val.cpu()).abs().max()))
        parts.append(actual)
        bits.append(rel[:,0].cpu())
    if max(errors)>2e-6:raise ValueError("Changed KD-SBIR evaluator")
    return torch.cat(parts),torch.cat(bits),max(errors)


@torch.inference_mode()
def run(w,workers,source):
    output=w/"results"/FAMILY
    root=output/"integrated"
    if source=="public":
        file=output/"pilot.json"
        data=read(file)
        p,mode,st,a=data["nominee"].split("/")
        d,ref,m,t,amount=int(p),"public",mode.replace("shared","real"),float(st),float(a)
        nominal=data["nominee"]
    else:
        file=root/("freeze.json" if source=="full" else "pilot.json")
        data=read(file)
        nominal=data["nominee"]
        d,ref,m,t,amount=parse(nominal)
    inputs=semantic_inputs(w,"sketchy")
    _,_,qs,_,gals,_,raw,_,target,scale=inputs
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Original evaluator hash changed")
    records={}
    tables={}
    for s,(_,labels) in qs.items():
        ids=stratified_sample(labels,36,63241)
        base=study_base(w,s,inputs,ids).cuda()
        pi,g=gals[s]
        folder=w/"cache"/VIEW_CACHE/"sketchy"
        plan=torch.load(folder/"plan.pt",mmap=True,weights_only=True)
        views=read_photos(folder,plan,pi).cuda().float().mean(2)
        y=labels[ids]
        categories=torch.arange(len(g.unique()))
        oracle=(g.cuda()[None,:]==categories.cuda()[:,None]).float()
        official,oracle_met,ranks,items,corrections={}, {}, {}, {}, defaultdict(list)
        errors={}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            base_metrics,base_top,base_error=exact(kd,s,base,y,g,pool)
            for seed in SEEDS:
                for pairing in PAIRING:
                    key=f"{seed}/{pairing}"
                    pack=build(views,raw[pi].cuda(),target,{(d,ref)},seed,scale,pairing)[d,ref]
                    read,energy=read_folds(base,pack,t)
                    diff=(read["shared"]-read["raw"] if m.endswith("_gain") else read["shared"]-base)
                    # Equal averaging across predetermined seeds can be assessed without searching for a seed.
                    corrections[pairing].append((amount*diff).cpu())
                    score=base+amount*diff
                    val,top,err=exact(kd,s,score,y,g,pool)
                    official[key]=val
                    errors[key]=err
                    obs,oe=read_folds(oracle,pack,t)
                    oc={}
                    auc={}
                    for kind,rank in obs.items():
                        v,_,er=exact(kd,s,rank,categories,g,pool)
                        oc[kind]=v
                        auc[kind]=pair_order(rank,g.cuda())
                        errors[key]=max(errors[key],er)
                    oracle_met[key],ranks[key]=oc,auc
                    items[key]={"class_signal_energy":oe.cpu(),"query_energy":energy.cpu(),"top1":top,
                                "ref_stats":[part.diagnostics for part in pack]}
                    del pack
                    print("locked assessment",s,key,"real",val.mean(0).tolist(),"oracle",oc["shared"].mean(0).tolist(),flush=True)
            for pairing,delta in corrections.items():
                z=base+torch.stack(delta).mean(0).cuda()
                val,top,err=exact(kd,s,z,y,g,pool)
                key="mean/"+pairing
                official[key]=val
                items[key]={"top1":top}
                errors[key]=err
        baseref=official[str(SEEDS[0])+"/actual"]
        records[s]={"query_ids":ids,"query_labels":y,"classes":categories,"baseline":base_metrics,"metrics":official,
                    "original_top":base_top,"oracle_metrics":oracle_met,"pair_order":ranks,"diagnosis":items}
        tables[s]={"baseline":base_metrics.mean(0).tolist(),"scores":{k:v.mean(0).tolist() for k,v in official.items()},
                   "delta_actual_minus_each":{k:paired_delta(baseref,v,y) for k,v in official.items()},
                   "delta_each_minus_base":{k:paired_delta(v,base_metrics,y) for k,v in official.items()},
                   "oracle_average_class_metrics":{k:{m:v.mean(0).tolist() for m,v in by.items()} for k,by in oracle_met.items()},
                   "oracle_class_pair_accuracy":{k:{m:float(v.mean()) for m,v in by.items()} for k,by in ranks.items()},
                   "mean_energy":{k:{"class":float(v["class_signal_energy"].mean()),"real_queries":float(v["query_energy"].mean())}
                                  for k,v in items.items() if "class_signal_energy" in v},
                   "official_queries_per_condition":len(y),"oracle_categories":len(categories),"max_error":max(max(errors.values()),base_error)}
    f=root/("associations-"+source+".pt")
    save_tensor(records,f)
    atomic_json(f.with_suffix(".json"),{"revision":revision(),"setting":nominal,"source_record":checksum(file),
              "seeds":SEEDS,"reference_designs":PAIRING,"splits":tables,"per_query_sha256":checksum(f),
              "scope":"oracle gallery labels only for retrospective readout on non-overlapping candidate partitions"})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--source",choices=("public","integrated","full"),required=True)
    a=p.parse_args()
    run(a.workspace,a.workers,a.source)
