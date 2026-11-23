"""Test gallery-assisted identity borrowing and its exact scalar counterfactual on complete benchmark ranks."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
import os
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,checksum,read
from .conditional_agreement_study import Replay
from .crossdomain_benchmark import revision
from .crossdomain_inputs import CHECKPOINT
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .partition_rank_replay import PendingReplay
from .query_partition_study import locked,save_lock,rank_metrics,mean_tables
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import stratified_sample
from .retrieval_evidence import load_kd,paired_delta
from .shared_private_names import FAMILY,path
from .shared_private_score import BASES,Row,ConditionalIndex,grid,read_scores


def choices(tables,screen):
    a,b=tables
    if a.keys()!=b.keys():raise ValueError("Unequal choices")
    def best(keys):return max(sorted(keys),key=lambda k:tuple((a[k][i]+b[k][i])/2 for i in (0,1)))
    rs={k:Row.parse(k) for k in a if not k.startswith("base/")}
    fit={};out=set(BASES)
    for mode in sorted({r.mode for r in rs.values()}):
        sub=[k for k,r in rs.items() if r.mode==mode]
        fit[mode]=best(sub);out.add(fit[mode])
        for family,prefix in (("arithmetic","arithmetic"),("prototype","prototype"),("tree","tree-"),("fixed","fixed-")):
            eligible=[k for k in sub if rs[k].model.startswith(prefix)]
            if not eligible:continue
            fit[family+"/"+mode]=best(eligible)
            if mode in ("joint","mass","peer","reg-cov","reg-joint"):out.add(fit[family+"/"+mode])
            if screen and mode in ("joint","reg-cov"):
                for level in sorted({rs[k].depth for k in eligible}):
                    out.add(best([k for k in eligible if rs[k].depth==level]))
    for k in list(out):
        if k in BASES:continue
        r=rs[k]
        if r.mode not in ("joint","reg-cov","reg-joint"):continue
        for v in ("joint","mass","fixed","peer","wrong","opposite","reg-cov","reg-wrong","reg-mass","reg-joint"):
            if v=="peer":c=Row(r.model,r.depth,1.,.6,v,1.).label()
            else:c=Row(r.model,r.depth,r.power,r.alpha,v,r.gain if v.startswith("reg-") else 1.).label()
            if c in a:out.add(c)
    return fit,sorted(out)


def audit(p,available):
    leads=[p["tree/joint"],p["joint"],p["tree/reg-cov"],p["reg-cov"],p["reg-joint"]]
    out={*BASES,*leads,p["mass"],p["peer"],p["fixed"],p["arithmetic/joint"],p["prototype/joint"],p["reg-mass"]}
    for k in leads:
        r=Row.parse(k)
        for v in ("mass","fixed","wrong","opposite","reg-wrong","reg-mass"):
            n=Row(r.model,r.depth,r.power,r.alpha,v,r.gain if v.startswith("reg-") else 1.).label()
            if n in available:out.add(n)
    return sorted(out),list(dict.fromkeys(leads))


@torch.inference_mode()
def run(w,ds,stage,only):
    root=w/"results"/FAMILY;root.mkdir(parents=True,exist_ok=True)
    if stage=="lock":
        prev=locked(root/"shortlist.json");p,_=choices(mean_tables(root,"scan-"),False);rows,leaders=audit(p,set(prev["config"]["scan"]))
        conf={"rows":rows,"leaders":leaders,"picked":p,"primary":p["tree/joint"],"prior":prev["sha256"],"code":revision(),
              "scans":{s:checksum(root/("scan-"+s+".json")) for s in ("sketchy_1","sketchy_2")},"selection":"mean two full Sketchy mAP then mean precision"}
        print("freeze",save_lock(root/"freeze.json",conf),len(rows),flush=True);return
    pilot=stage.startswith("pilot");official=stage in ("official","pilot-official")
    if ds=="tuberlin" and stage!="official":raise ValueError("No transfer development")
    if stage=="pilot":
        rows=grid();sig=save_lock(root/"protocol.json",{"code":revision(),"model":CHECKPOINT,"KD":KD_FUNCTION_SHA,"grid":rows,
             "pilot":"36 unseen queries per class, full gallery; primary tree-based shared/private name model; exact mass-matched controls"})
    else:
        lock=locked(root/("freeze.json" if stage=="official" else "shortlist.json"))
        sig=lock["sha256"];rows=lock["config"][{"pilot-official":"audit","scan":"scan","official":"rows"}[stage]]
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Original evaluator mismatch")
    cfg,qp,qs,pp,gs,query,_,src,_,scale=semantic_inputs(w,ds)
    torch.set_float32_matmul_precision("high")
    load=lambda f:torch.load(f,weights_only=True,mmap=True)
    for s,(qi,y) in qs.items():
        if only and s!=only:continue
        dest=root/(("" if stage=="official" else stage+"-")+s+".json")
        if dest.is_file():
            x=read(dest)
            if x["signature"]!=sig or checksum(dest.with_suffix(".pt"))!=x["per_query_sha256"]:raise ValueError("Changed checkpoint")
            continue
        cache=path(w,s);info=read(cache/"done.json");bank=load(cache/"photo-words.pt").cuda()
        rawgeo=load(cache/"geometry.pt")
        geo={k:[z.cuda() for z in rawgeo[k]] for k in ("folds","groups","priors")};pi,gl=gs[s];gt=gl.cuda()
        ix=stratified_sample(y,36,63241) if pilot else torch.arange(len(y))
        labels=y[ix];q=query[qi[ix]].cuda()
        likelihood=(scale*(q@src.T)).softmax(1)
        raw=likelihood@bank.T
        if not torch.isfinite(raw).all():raise ValueError("Bad photo-name score")
        reg=load(w/"cache/gallery-stable-region-evidence"/(s+"-regional-reference.pt"))[ix].cuda()
        by=defaultdict(list);stats={};m=defaultdict(list);top=defaultdict(list);lead=defaultdict(list)
        for k in rows:
            if k not in BASES:
                r=Row.parse(k);by[(r.model,r.depth,r.power)].append(r)
        tail=torch.randperm(len(gl),generator=torch.Generator().manual_seed(37001)).to(gt.device)
        # Use the original direct score to break remaining numerical ties in lexical models.
        # For large galleries keep tie orders in host memory between candidate banks.
        def ties(v):
            z=[]
            for batch in v.split(192):z.append(tail[batch[:,tail].argsort(dim=1,descending=True,stable=True)].int().cpu())
            return torch.cat(z)
        stable={"direct":ties(raw),"regional":ties(reg)}
        workers=min(8,len(os.sched_getaffinity(0)))
        with (ThreadPoolExecutor(max_workers=workers) if official else nullcontext()) as pool:
            pending=PendingReplay(Replay(kd,gl,s),pool,workers) if official else None
            def record(key,v,sl):
                sec="regional" if (key=="base/regional" or "/reg-" in key) else "direct"
                val,ids,good=rank_metrics(v,stable[sec][sl].long().cuda(),labels[sl].cuda(),gt,s)
                if pending:
                    pending.submit(ids,labels[sl],val,m[key]);lead[key].append(ids[:,:5].int().cpu())
                else:m[key].append(val.cpu())
                top[key].append(good.cpu())
            for b in range(0,len(ix),96):
                sl=slice(b,b+96)
                for key,mat in (("base/direct",raw),("base/regional",reg)):
                    if key in rows:record(key,mat[sl],sl)
            for no,((model,lev,power),recipes) in enumerate(sorted(by.items())):
                peers=[load(cache/f"words-{h}-{model}-{lev}.pt").cuda() for h in (0,1)]
                index=ConditionalIndex(bank,geo,peers,lev,power)
                for b in range(0,len(ix),96):
                    sl=slice(b,b+96);group,joint=index.values(likelihood[sl])
                    for key,score in read_scores(raw[sl],reg[sl],group,joint,index,recipes):record(key,score,sl)
                stats[f"{model}/{lev}/{power:g}"]={"posterior_error":index.audit,"null_error":index.frozen_chance_error,
                      "median_bayes_factor":float(index.ratio.median()),"peer_candidate_log_agreement":float(index.ratio.log().mean())}
                del index,peers
                if no%5==0:print(s,stage,no+1,len(by),flush=True)
            err=pending.finish() if pending else None
        metrics={k:torch.cat(v) for k,v in m.items()};tops={k:torch.cat(v) for k,v in top.items()}
        if set(metrics)!=set(rows):raise ValueError("Incomplete name posterior scan")
        save_tensor({"ids":ix,"labels":labels,"metrics":metrics,"top1":tops,"first_five":{k:torch.cat(v) for k,v in lead.items()}},dest.with_suffix(".pt"))
        result={"signature":sig,"code":revision(),"queries":len(ix),"gallery":len(gl),"classes":len(labels.unique()),"words":info["vocabulary"],
                "index_checks":stats,"values":{k:v.double().mean(0).tolist() for k,v in metrics.items()},"original_evaluator":official,
                "KD_max_error":err,"per_query_sha256":checksum(dest.with_suffix(".pt"))}
        if official:
            cl=labels.unique(return_inverse=True)[1];primary=lock["config"]["leaders"]
            result["paired"]={k:{comp:paired_delta(metrics[k],metrics[comp],cl) for comp in rows} for k in primary}
            result["top1"]={k:{comp:[int((tops[k]&~tops[comp]).sum()),int((~tops[k]&tops[comp]).sum())] for comp in rows} for k in primary}
        atomic_json(dest,result);print("saved",s,stage,flush=True)
        del bank,raw,reg,likelihood
    if stage=="pilot" and not only:
        p,scan=choices(mean_tables(root,"pilot-"),True);ar,leads=audit(p,set(scan))
        conf={"picked":p,"scan":scan,"audit":ar,"leaders":leads,"protocol":sig}
        print("shortlist",save_lock(root/"shortlist.json",conf),len(scan),len(ar),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace",type=Path,default=Path("/content/sbir"));ap.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    ap.add_argument("--stage",choices=("pilot","pilot-official","scan","lock","official"),required=True)
    ap.add_argument("--split",choices=("sketchy_1","sketchy_2","tuberlin"))
    a=ap.parse_args();run(a.workspace,a.dataset,a.stage,a.split)
