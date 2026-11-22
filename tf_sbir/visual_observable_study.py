"""Compare the whole visual measurement space to named probes on an identical retrieval base."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
from pathlib import Path
import time

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .crossdomain_benchmark import digest, revision
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .integrated_observable_operators import BLENDS, CONDITIONS, GAINS, REFS, STRENGTHS, build, population_checks
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .response_geometry_views import FAMILY as SHARED_PHOTOS, read_photos, study_base
from .retrieval_diagnosis import stratified_sample
from .retrieval_evidence import load_kd, paired_delta
from .shared_semantic_operators import FAMILY


DIMS=(16,32,48,64)
DIRECT=("visual","white")
SEED=59731


def key(d,r,m,s,a):
    return f"{d}/{r}/{m}/{s:g}/{a:g}"


def parse(k):
    d,r,m,s,a=k.split("/")
    return int(d),r,m,float(s),float(a)


def grid():
    cells=["regional"]
    for d in DIMS:
        cells.extend(key(d,"visual",m,0,a) for m in ("raw","linear") for a in BLENDS)
        for r in REFS:
            modes=CONDITIONS if r in DIRECT else ("real",)
            for s in STRENGTHS:
                cells.extend(key(d,r,m,s,a) for m in modes for a in BLENDS)
                cells.extend(key(d,r,m+"_gain",s,a) for m in modes for a in GAINS)
    return cells


def choose(means):
    def best(check):
        return max((k for k in means if k!="regional" and check(*parse(k))),key=means.get)
    main=best(lambda d,r,m,s,a:r in DIRECT and m in ("real","real_gain"))
    d,r,m,st,a=parse(main)
    suffix="_gain" if m.endswith("_gain") else ""
    matches=[key(d,r,v+suffix,st,a) for v in CONDITIONS]
    matches.extend(key(d,v,"real"+suffix,st,a) for v in REFS if v!=r)
    if not suffix:matches.extend(key(d,"visual",v,0,a) for v in ("raw","linear"))
    return {"nominee":main,"matched":matches,"best_base_pool":best(lambda d,r,m,s,a: m in ("raw","linear")),
            "best_public":best(lambda d,r,m,s,a:r=="public"),"best_one":best(lambda d,r,m,s,a:r=="one"),
            "best_visual16":best(lambda d,r,m,s,a:r=="visual16"),
            "best_norm_matched":best(lambda d,r,m,s,a:m.startswith("matched")),
            "best_rotated":best(lambda d,r,m,s,a:m.startswith("rotated")),
            "best_delta":best(lambda d,r,m,s,a:r in DIRECT and m=="real_gain")}


def include_fixed(rows,nominees):
    out=set(rows)
    for k in nominees:
        d,r,m,s,a=parse(k)
        suffix="_gain" if m.endswith("_gain") else ""
        out.update(key(d,ref,c+suffix,s,a) for ref in DIRECT for c in CONDITIONS)
        out.update(key(d,v,"real"+suffix,s,a) for v in REFS)
        if not suffix:out.update(key(d,"visual",c,0,a) for c in ("raw","linear"))
    return sorted(out)


def subset(choice):
    return sorted({"regional",*choice["matched"],*(v for k,v in choice.items() if k!="matched")})


def shortlist(means):
    out=set(subset(choose(means)))
    for ref in REFS:
        for group in (("real",),("real_gain",),("matched","matched_gain"),("rotated","rotated_gain"),("raw","linear")):
            candidates=sorted((k for k in means if k!="regional" and parse(k)[1]==ref and parse(k)[2] in group),
                              key=means.get,reverse=True)
            out.update(candidates[:2])
    possible=[k for k in out if k!="regional" and parse(k)[1] in DIRECT and parse(k)[2] in ("real","real_gain")]
    return include_fixed(out,possible)


def calculate(s,base,y,g,folds,rows,kd,workers):
    grouped=defaultdict(list)
    for k in rows:
        if k!="regional":
            d,r,m,t,a=parse(k)
            grouped[d,r].append((k,m,t,a))
    truth=g.cuda()
    evaluator=Replay(kd,g,s) if kd else None
    met,top,errors=defaultdict(list),defaultdict(list),{}
    with (ThreadPoolExecutor(max_workers=workers) if kd else nullcontext()) as pool:
        for off in range(0,len(y),80):
            sl=slice(off,off+80)
            z=base[sl].cuda()
            labels=y[sl]
            def record(name,scores):
                val,ranks,hit,_=map_values(scores,labels.cuda(),truth,s)
                val=val.cpu()
                if evaluator:
                    actual=evaluator.score(ranks,labels,pool)
                    err=float((actual-val).abs().max())
                    if err>2e-6:raise ValueError("KD-SBIR disagrees with fast calculation")
                    errors[name]=max(err,errors.get(name,0.))
                    val=actual
                met[name].append(val)
                top[name].append(hit[:,0].cpu())
            record("regional",z)
            for conf,variants in grouped.items():
                strengths=sorted({t for _,m,t,_ in variants if m not in ("raw","linear")})
                idx={}
                for fold in folds[conf]:
                    pred=fold.read(z,strengths)
                    for k,v in pred.items():
                        if k not in idx:idx[k]=torch.empty_like(z)
                        idx[k][:,fold.target_ids]=v
                for name,m,t,a in variants:
                    mode=m.removesuffix("_gain")
                    lookup=("shared" if mode=="real" else mode,t)
                    if mode in ("raw","linear"):
                        score=z+a*(idx[mode]-z)
                    elif m.endswith("_gain"):
                        score=z+a*(idx[lookup]-idx["raw"])
                    else:
                        score=z+a*(idx[lookup]-z)
                    record(name,score)
            if off%1200==0:print("KD" if kd else "scan",s,off,len(y),flush=True)
    return {k:torch.cat(v) for k,v in met.items()},{k:torch.cat(v) for k,v in top.items()},errors


def file_state(path,sig):
    if not path.is_file():return None
    record=read(path)
    if record["signature"]!=sig or checksum(path.with_suffix(".pt"))!=record["per_query_sha256"]:
        raise ValueError("Invalid split result "+str(path))
    return record


def locked(path,data):
    rec=data|{"sha256":digest(data)}
    if path.is_file() and rec!=read(path):raise ValueError("Conflicting protocol "+str(path))
    atomic_json(path,rec)
    return rec["sha256"]


def one(w,s,inputs,rows,ids,dest,sig,kd,workers,main=None,reference=None,seed=SEED):
    done=file_state(dest,sig)
    if done is not None:return done
    _,_,qs,_,gs,_,photo,_,tgt,scale=inputs
    pi,g=gs[s]
    labels=qs[s][1][ids]
    base=study_base(w,s,inputs,ids)
    folder=(w/"cache"/FAMILY/"gallery-views/tuberlin" if s=="tuberlin" else w/"cache"/SHARED_PHOTOS/"sketchy")
    plan=torch.load(folder/"plan.pt",weights_only=True,mmap=True)
    views=read_photos(folder,plan,pi).cuda().float().mean(2)
    settings={parse(k)[:2] for k in rows if k!="regional"}
    folds=build(views,photo[pi].cuda(),tgt,settings,seed,scale)
    del views
    vals,top,errors=calculate(s,base,labels,g,folds,rows,kd,workers)
    diag={f"{d}/{r}":[o.diagnostics for o in v] for (d,r),v in folds.items()}
    del folds
    worst=None
    if reference is not None:
        old=torch.load(reference,mmap=True,weights_only=True)["metrics"]
        worst=max(float((v-old[k]).abs().max()) for k,v in vals.items())
        if worst>2e-6:raise ValueError(f"Scan and independent original replay differ {worst}")
    save_tensor({"query_ids":ids,"query_labels":labels,"metrics":vals,"top1":top},dest.with_suffix(".pt"))
    result={"signature":sig,"queries":len(ids),"classes":int(labels.unique().numel()),"gallery":len(g),
            "values":{k:v.mean(0).tolist() for k,v in vals.items()},"reference_diagnosis":diag,
            "GPU":torch.cuda.get_device_name(),"per_query_sha256":checksum(dest.with_suffix(".pt"))}
    if kd:
        result.update(nominee=main,KD={"original_queries_per_condition":len(ids),"max_error":max(errors.values()),
                      "scan_error":worst},paired={k:paired_delta(vals[main],v,labels) for k,v in vals.items()},
                      top1={"fixed":int((~top["regional"]&top[main]).sum()),"broken":int((top["regional"]&~top[main]).sum())})
    atomic_json(dest,result)
    print("saved",s,dest.stem,result["values"].get(main),flush=True)
    return result


@torch.inference_mode()
def run(w,workers,phase):
    root=w/"results"/FAMILY/"integrated"
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Wrong KD implementation")
    rows=grid()
    # The same frozen source revision runs the screen, full development and transfer.
    spec={"revision":revision(),"grid":rows,"KD":KD_FUNCTION_SHA,"pilot_seed":63241,"gallery_seed":SEED,
          "pilot_queries_per_class":36,"selection":"equal weights for Sketchy mAP followed by precision",
          "all_visual_probes_rotation_check":population_checks(torch.device("cuda"))}
    sig=locked(root/"protocol.json",spec)
    if phase=="pilot":
        inputs=semantic_inputs(w,"sketchy")
        results={}
        for s,(_,labels) in inputs[2].items():
            ids=stratified_sample(labels,36,63241)
            results[s]=one(w,s,inputs,rows,ids,root/("pilot-"+s+".json"),sig,None,workers)
        means=averages(results)
        choice=choose(means)
        checks={}
        for s,(_,labels) in inputs[2].items():
            ids=stratified_sample(labels,36,63241)
            checks[s]=one(w,s,inputs,subset(choice),ids,root/("pilot-official-"+s+".json"),sig,kd,workers,
                          choice["nominee"],root/("pilot-"+s+".pt"))
        atomic_json(root/"pilot.json",choice|{"means":means,"official":{s:v["values"] for s,v in checks.items()},
                    "checked":{s:v["KD"] for s,v in checks.items()}})
        print("chosen pilot",choice["nominee"],means[choice["nominee"]],means["regional"],flush=True)
        return
    if phase=="full-sketchy":
        pilot=read(root/"pilot.json")
        selected=shortlist(pilot["means"])
        shortlist_sig=locked(root/"shortlist.json",{"protocol":sig,"rows":selected,"pilot":checksum(root/"pilot.json")})
        inputs=semantic_inputs(w,"sketchy")
        records={}
        for s,(_,y) in inputs[2].items():
            records[s]=one(w,s,inputs,selected,torch.arange(len(y)),root/("scan-"+s+".json"),shortlist_sig,None,workers)
        means=averages(records)
        choice=choose(means)
        frozen=locked(root/"freeze.json",choice|{"means":means,"official_rows":subset(choice),"shortlist":shortlist_sig,
                      "full_Sketchy":{s:checksum(root/("scan-"+s+".json")) for s in records}})
        print("full freeze",choice["nominee"],means[choice["nominee"]],means["regional"],flush=True)
        for s,(_,y) in inputs[2].items():
            one(w,s,inputs,subset(choice),torch.arange(len(y)),root/(s+".json"),frozen,kd,workers,choice["nominee"],
                root/("scan-"+s+".pt"))
        return
    freeze=read(root/"freeze.json")
    if digest({k:v for k,v in freeze.items() if k!="sha256"})!=freeze["sha256"]:raise ValueError("Invalid selection")
    inputs=semantic_inputs(w,"tuberlin")
    y=inputs[2]["tuberlin"][1]
    one(w,"tuberlin",inputs,freeze["official_rows"],torch.arange(len(y)),root/"tuberlin.json",freeze["sha256"],
        kd,workers,freeze["nominee"])


def averages(results):
    return {k:[sum(v["values"][k][j] for v in results.values())/len(results) for j in (0,1)]
            for k in next(iter(results.values()))["values"]}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    cli.add_argument("--workers",type=int,default=2)
    cli.add_argument("--phase",choices=("pilot","full-sketchy","transfer"),required=True)
    a=cli.parse_args()
    run(a.workspace,a.workers,a.phase)
