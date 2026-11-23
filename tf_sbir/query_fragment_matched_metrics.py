"""Replay identical frozen graph and coefficient settings for all controls in the paired top-rank audit."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import os
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,checksum,read
from .conditional_agreement_study import Replay
from .crossdomain_benchmark import digest,revision
from .encoding import save_tensor
from .map_diagnosis import map_values
from .query_fragment_likelihood import GalleryWalk
from .query_fragment_rank_audit import GAMMA,TAU,views
from .query_fragment_reference_inputs import FAMILY,dataset,load
from .query_fragment_reference_study import split_context
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd,paired_delta


@torch.inference_mode()
def run(w,ds):
    root=w/"results"/FAMILY
    frozen=read(root/"freeze.json")
    if digest(frozen["config"])!=frozen["sha256"] or frozen["config"]["pure"]["best"]!="geometric/64/0.8/walk_after_ratio":
        raise ValueError("Unexpected locked scorer")
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Wrong evaluator")
    inputs=dataset(w,ds);bank=load(w,ds,"full")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    for s in inputs[2]:
        dest=root/("matched-"+s+".json")
        if dest.exists():
            report=read(dest)
            if report["freeze"]!=frozen["sha256"] or checksum(dest.with_suffix(".pt"))!=report["per_query_sha256"]:raise ValueError("Changed matched metrics")
            continue
        audit=read(root/("rank-audit-"+s+".json"))
        if audit["freeze"]!=frozen["sha256"]:raise ValueError("Different diagnostic protocol")
        previous=torch.load(root/("rank-audit-"+s+".pt"),mmap=True,weights_only=True)
        p=split_context(w,s,ds,inputs,bank,"official",False)
        if not torch.equal(p["ids"],previous["ids"]):raise ValueError("Different query order")
        g=p["gallery"];pl=p["gl"];gt=pl.cuda()
        graph=GalleryWalk(g);replay=Replay(kd,pl,s)
        m=defaultdict(list);maxerr=0.
        with ThreadPoolExecutor(max_workers=min(8,max(1,len(os.sched_getaffinity(0))-1))) as pool:
            for off in range(0,len(p["ids"]),64):
                sl=slice(off,off+64);q=p["q"][sl];labels=p["labels"][sl]
                rs=views(bank,p["pos"][sl])
                for name in ("appearance","mean_colour","empty_token","permuted","area_shuffle"):
                    score=graph.log(TAU*((q-GAMMA*rs[name])@g.T))
                    val,order,_,_=map_values(score,labels.cuda(),gt,s)
                    key="native" if name=="appearance" else name
                    expected=previous["pairs"][key+"/walk"]["top"][sl,1 if name=="appearance" else 0]
                    # GPU batching can change near-ties. Preserve the realized rankings, and record tied differences.
                    top=order[:,0].cpu()
                    m["top/"+name].append(top.int())
                    m["audit_top_match/"+name].append(top==expected)
                    exact=replay.score(order,labels,pool);err=float((exact-val.cpu()).abs().max())
                    if err>2e-6:raise ValueError("Different KD metric")
                    maxerr=max(maxerr,err);m[name].append(exact)
                if off%3072==0:print(s,off,len(p["ids"]),flush=True)
        allrows={k:torch.cat(rows) for k,rows in m.items()}
        values={k:v for k,v in allrows.items() if "/" not in k}
        labels=p["labels"].unique(return_inverse=True)[1]
        previous_best=read(root/(s+".json"))["values"][frozen["config"]["pure"]["best"]]
        observed=values["appearance"].double().mean(0).tolist()
        mean_error=max(abs(a-b) for a,b in zip(previous_best,observed,strict=True))
        if mean_error>1e-4:raise ValueError("Standalone score differs from the equivalent frozen likelihood")
        save_tensor({"ids":p["ids"],"labels":p["labels"],**allrows},dest.with_suffix(".pt"))
        atomic_json(dest,{"code":revision(),"freeze":frozen["sha256"],"post_lock_matched_diagnostic":True,"coefficient":GAMMA,"temperature":TAU,
            "query_count":len(labels),"gallery":len(pl),"classes":len(labels.unique()),"KD_max_error":maxerr,"original_mean_error":mean_error,
            "values":{k:v.double().mean(0).tolist() for k,v in values.items()},
            "paired":{k:paired_delta(values["appearance"],v,labels) for k,v in values.items() if k!="appearance"},
            "top1_replay_matches":{k:int(allrows["audit_top_match/"+k].sum()) for k in values},"per_query_sha256":checksum(dest.with_suffix(".pt"))})
        print("verified matched",s,flush=True)
        del p,graph


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--dataset",choices=("sketchy","tuberlin"),required=True)
    args=parser.parse_args();run(args.workspace,args.dataset)
