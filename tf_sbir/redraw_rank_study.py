"""Run the single-sketch shared-margin retrieval study on Colab."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,checksum,read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES,bare_words
from .crossdomain_benchmark import SPLIT_HASHES,digest,revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .reciprocal_reassembly import adjust
from .redraw_margins import group_inputs,read_group
from .redraw_quantiles import field,check
from .regional_intervention_study import mixscore
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import load_kd,paired_delta
from .semantic import read_words

FAMILY="redraw-rank-stability"
SPLITS=("sketchy_1","sketchy_2")


def key(c):return json.dumps(c,separators=(",",":"))


def grid():
    return [(n,t,p,m,b) for n in (6,16,64) for t in (.5,1.5,4.) for p in (.5,1.)
            for m in ("paired","independent","fixed_gallery") for b in (.4,1.)]


def best(values,kind,n):
    return [tuple(json.loads(k)) for k in sorted((k for k in values if json.loads(k)[3]==kind),key=values.get,reverse=True)[:n]]


def collect(w,ds,s,qi,pi,qp,pp,q,p,src,tgt,scale,bank):
    ident=digest({"source":revision(),"q":digest([str(qp[i]) for i in qi.tolist()]),"p":digest([str(pp[i]) for i in pi.tolist()])})
    file=w/"cache"/FAMILY/(s+"-readouts-"+ident[:14]+".pt")
    if file.exists():return torch.load(file,weights_only=True,mmap=True)
    qview=torch.load(w/"cache/regional-evidence-interventions"/ds/"regions.pt",mmap=True,weights_only=True)
    pview=torch.load(w/"cache/nested-scale-word-evidence"/ds/"photo.pt",mmap=True,weights_only=True)
    g=torch.Generator().manual_seed(95531)
    refer=torch.randperm(len(pi),generator=g)[:1024]
    gallery=tgt.new_zeros((bank["full"].size(0),bank["full"].size(0)),dtype=torch.float64)
    for part in refer.split(52):
        j=pi[part]
        _,v,_=read_group(p[j].cuda(),pview[j].cuda().float(),tgt,scale,bank)
        gallery+=v.double().sum(0)
    gallery=(gallery/len(refer)).float().cpu()
    n=[];cv=[];zooms=[]
    for part in qi.split(52):
        ix=torch.searchsorted(qview["ids"],part)
        if not torch.equal(qview["ids"][ix],part):raise ValueError("Wrong query ordering")
        v=qview["features"][ix,:,2].cuda().float()
        c,dev,z=read_group(q[part].cuda(),v,src,scale,bank)
        n.append(c.cpu());cv.append(dev.cpu());zooms.append(z.cpu())
    out={"nominal":torch.cat(n),"cov":torch.cat(cv),"zoom":torch.cat(zooms),"gallery":gallery}
    save_tensor(out,file)
    return out


def measure(readings,historical,bank,ids,y,g,split,cells,reference,workers,main):
    buffers={key(c):[] for c in cells}
    errs={};tops=[];tests={}
    gallery=readings["gallery"].cuda()
    ctx=ThreadPoolExecutor(max_workers=workers) if reference else nullcontext()
    with ctx as pool:
        for j,part in enumerate(ids.split(60)):
            s=readings["nominal"][part].cuda();cov=readings["cov"][part].cuda()
            if j==0:tests=check(s,cov,gallery)
            labels=y[part]
            base=mixscore(readings["zoom"][part].cuda(),bank["compact"],historical[part].cuda().clamp_min(1e-30).log(),.4)
            for cfg in sorted(set(c[:4] for c in cells)):
                n,t,p,mode=cfg
                groups,offset=field(s,cov,n,t,p,mode,gallery)
                for c in [c for c in cells if c[:4]==cfg]:
                    pred=base if c[4]==0 else adjust(base,bank["groups"],groups,offset,c[4])
                    score,order,_,_=map_values(pred,labels.cuda(),g.cuda(),split)
                    k=key(c)
                    if reference:
                        exact=reference.score(order,labels,pool)
                        e=float((score.cpu()-exact).abs().max())
                        if e>2e-6:raise ValueError("KD replay differs")
                        errs[k]=max(errs.get(k,0.),e)
                        if k==main:tops.append(order[:,:25].int().cpu())
                    else:exact=score.cpu()
                    buffers[k].append(exact)
    return {k:torch.cat(v) for k,v in buffers.items()},torch.cat(tops) if tops else None,errs,tests


@torch.inference_mode()
def run(w,phase,workers):
    root=w/"results"/FAMILY
    rev=revision()
    spec={"revision":rev,"grid":grid(),"main":"paired","grid_seed":63241,"pilot_per_class":36,
          "selection":"equal mean over both full Sketchy mAP values, then mean precision"}
    sig=digest(spec)
    if (root/"protocol.json").exists():
        if read(root/"protocol.json")["sha256"]!=sig:raise ValueError("Previous rank protocol differs")
    else:atomic_json(root/"protocol.json",spec|{"sha256":sig})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Changed original KD evaluator")
    stages={"pilot":("pilot",),"lock":("shortlist",),"audit":("final",),"all":("pilot","shortlist","final")}[phase]
    for stage in stages:
        if stage=="pilot":cells=grid()+[(6,1.5,1.,"homogeneous",0.)]
        elif stage=="shortlist":
            m=read(root/"pilot.json")["means"]
            cells=best(m,"paired",3)+best(m,"independent",2)+best(m,"fixed_gallery",2)+best(m,"homogeneous",1)
        else:
            lock=read(root/"freeze.json")
            if lock["full_sketchy_sha256"]!=checksum(root/"shortlist.json") or lock["revision"]!=rev:raise ValueError("Wrong freeze")
            chosen,dia,global_=(tuple(lock[k]) for k in ("primary","independent","fixed_global"))
            n,t,p,_,b=chosen
            cells=sorted({chosen,dia,global_,(n,t,p,"homogeneous",0.),
                          *[(n,t,p,mode,b) for mode in ("independent","reverse_pairs","rank_only","linear","fixed_gallery","homogeneous")]})
        table={}
        for ds in (("sketchy","tuberlin") if stage=="final" else ("sketchy",)):
            cfg,qp,qs,pp,gs,qraw,praw=global_inputs(w,ds)
            words=read_words(cfg.vocabulary)
            if len(words)!=20931:raise ValueError("Wrong vocabulary")
            src,tgt=(v.cuda() for v in load_words(cfg,words));_,scale=bare_words(cfg,words)
            for s,(qi,y) in qs.items():
                pi,g=gs[s]
                if (len(qi),len(pi),len(g.unique()))!=SIZES[s] or digest(UNSEEN_CLASSES[s])!=SPLIT_HASHES[s]:raise ValueError("Changed evaluation split")
                dest=root/(s+".json" if stage=="final" else stage+"-"+s+".json")
                if dest.is_file():
                    old=read(dest)
                    if old["signature"]!=sig or (stage=="final" and old["per_query_sha256"]!=checksum(root/(s+".pt"))):raise ValueError("Stale split")
                    table[s]=old["values"];continue
                bank=group_inputs(w,s,praw[pi].cuda(),pi,pp,tgt,scale,checksum(cfg.vocabulary))
                readings=collect(w,ds,s,qi,pi,qp,pp,qraw,praw,src,tgt,scale,bank)
                base=torch.load(w/"cache/pragmatic-gallery-competition"/(s+"-named-rgb16.pt"),weights_only=True,mmap=True)
                if base.shape!=(len(qi),len(pi)):raise ValueError("Wrong historical base")
                ids=stratified_sample(y,36,63241) if stage=="pilot" else torch.arange(len(qi))
                rows,top,err,tests=measure(readings,base,bank,ids,y,g,s,cells,Replay(kd,g,s) if stage=="final" else None,
                                           workers,key(chosen) if stage=="final" else None)
                vals={k:v.mean(0).tolist() for k,v in rows.items()};table[s]=vals
                obj={"signature":sig,"values":vals,"queries":len(ids),"gallery":len(pi),"identities":tests}
                if stage=="final":
                    path=root/(s+".pt");save_tensor({"metrics":rows,"query_labels":y,"top25":top},path)
                    obj.update({"protocol":{"split":s,"queries":len(qi),"gallery":len(pi),"classes":len(g.unique()),
                           "checkpoint":CHECKPOINT,"vocabulary_sha256":checksum(cfg.vocabulary),"kd_sha256":KD_FUNCTION_SHA,"gpu":torch.cuda.get_device_name(),
                           "revision":rev,"map_k":200 if s=="sketchy_2" else 0,"p_k":200 if s=="sketchy_2" else 100},
                           "primary":key(chosen),"best_contrast":key(dia),"fixed_global":key(global_),"baseline":key((n,t,p,"homogeneous",0.)),
                           "scores":{k:dict(zip(("mAP","precision"),v)) for k,v in vals.items()},
                           "vs_controls":{k:paired_delta(rows[key(chosen)],v,y) for k,v in rows.items()},
                           "per_query_sha256":checksum(path),"original_kd_queries_per_condition":len(qi),"kd_max_abs_error":err})
                atomic_json(dest,obj)
                win=max(vals,key=vals.get);print(stage,s,win,vals[win],flush=True)
        if stage!="final":
            m={k:[sum(table[s][k][i] for s in SPLITS)/2 for i in (0,1)] for k in table[SPLITS[0]]}
            atomic_json(root/(stage+".json"),{"means":m,"splits":table})
            if stage=="shortlist":
                lock={"primary":best(m,"paired",1)[0],"independent":best(m,"independent",1)[0],"fixed_global":best(m,"fixed_gallery",1)[0],
                      "revision":rev,"full_sketchy_sha256":checksum(root/"shortlist.json")}
                atomic_json(root/"freeze.json",lock|{"sha256":digest(lock)})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--phase",choices=("pilot","lock","audit","all"),default="all")
    p.add_argument("--workers",type=int,default=8)
    a=p.parse_args();run(a.workspace,a.phase,a.workers)
