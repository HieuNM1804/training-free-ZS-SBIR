"""Test a missing-parts likelihood on full photo galleries using one sketch at a time."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .cluster_mine import cluster_assignments
from .concept_benchmark import atomic_json,checksum,read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES,bare_words
from .crossdomain_benchmark import digest,revision,SPLIT_HASHES
from .crossdomain_inputs import CHECKPOINT
from .cross_view_region_cache import FAMILY as SOURCE
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .partial_object_parts import designs,vocabulary,photo_words,region_ratios,average,moments,evidence,check
from .quickdraw_locked import KD_FUNCTION_SHA
from .regional_intervention_study import index as compact_index,mixscore
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import load_kd,paired_delta
from .semantic import read_words

FAMILY="partial-object-gallery-evidence"
SPLITS=("sketchy_1","sketchy_2")


def key(c):return json.dumps(c,separators=(",",":"))


def grid():
    candidates = [(n,t,kind,eta,a,"local","disjoint") for n in (128,384) for t in (.5,1.)
            for kind in ("one_parent","independent","one_region","photo")
            for eta in ((.2,.55,.85) if kind!="one_region" else (.2,)) for a in (.2,.55)]
    candidates.append((128, .5, "base", .2, 0., "local", "disjoint"))
    return candidates


def pick(m,kind,n):
    return [tuple(json.loads(k)) for k in sorted((k for k in m if json.loads(k)[2]==kind),key=m.get,reverse=True)[:n]]


def tables_for(r,groups,design):
    per=moments(r,design)
    return {"one_parent":average(per,groups),"independent":moments(average(r,groups),design),
            "photo":per,"one_region":average(per[:,1],groups).clamp_min(1e-30).log()}


def measure(pq,src,scale,heads,compact,bank,extra,groups,hist,y,g,ids,cells,s,kd,workers,primary):
    rows=defaultdict(list);err={};tops=[];checks={}
    variants=designs()
    replay=Replay(kd,g,s) if kd else None
    context=ThreadPoolExecutor(max_workers=workers) if kd else nullcontext()
    with context as pool:
        for j,ix in enumerate(ids.split(24)):
            qv=pq[j*24:j*24+len(ix)].cuda().float()
            orig=(qv@src[heads].T)*scale
            z=torch.logaddexp(orig[:,:,:9].logsumexp((1,2))-math.log(18),
                              orig[:,:,9:].logsumexp((1,2))-math.log(8))-math.log(2)
            base=mixscore(z,compact,hist[ix].cuda().clamp_min(1e-30).log(),.35)
            labels=y[ix]
            for n,t,target,geometric in sorted(set((c[0],c[1],c[5],c[6]) for c in cells)):
                match=[c for c in cells if (c[0],c[1],c[5],c[6])==(n,t,target,geometric)]
                r=region_ratios(qv,src[extra[n]],bank[n][target],scale,t)
                d=tables_for(r,groups,variants[geometric])
                if j==0:checks[key((n,t,target,geometric))]=check(r,variants[geometric])
                for method,eta in sorted(set((c[2],c[3]) for c in match)):
                    if method=="one_region":raw=d[method][:,groups]
                    elif method=="photo":raw=evidence(d[method],eta)
                    elif method=="excess":
                        raw=(evidence(d["one_parent"],eta)-evidence(d["independent"],eta))[:,groups]
                    elif method=="base":raw=base
                    else:raw=evidence(d[method],eta)[:,groups]
                    for c in (c for c in match if c[2:4]==(method,eta)):
                        a=c[4]
                        score=base+a*raw if method=="excess" else (1-a)*base+a*raw
                        quick,order,_,_=map_values(score,labels.cuda(),g.cuda(),s)
                        nm=key(c)
                        if replay:
                            measured=replay.score(order,labels,pool)
                            e=float((quick.cpu()-measured).abs().max())
                            if e>2e-6:raise ValueError("Original KD rank replay differs")
                            err[nm]=max(err.get(nm,0.),e)
                            if nm==primary:tops.append(order[:,:25].int().cpu())
                        else:measured=quick.cpu()
                        rows[nm].append(measured)
                del d,r
    return {k:torch.cat(v) for k,v in rows.items()},err,torch.cat(tops) if tops else None,checks


@torch.inference_mode()
def run(w,phase,workers):
    root=w/"results"/FAMILY
    rev=revision(); spec={"revision":rev,"grid":grid(),"pilot_seed":63241,"pilot_per_class":36,
                          "primary":"one_parent","selection":"full equal Sketchy mAP then mean precision"}
    sig=digest(spec)
    if (root/"protocol.json").exists():
        if read(root/"protocol.json")["sha256"]!=sig:raise ValueError("Different experimental revision")
    else:atomic_json(root/"protocol.json",spec|{"sha256":sig})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Changed evaluator")
    stages={"pilot":("pilot",),"lock":("shortlist",),"audit":("final",),"all":("pilot","shortlist","final")}[phase]
    for stage in stages:
        if stage=="pilot":cells=grid()
        elif stage=="shortlist":
            m=read(root/"pilot.json")["means"]
            cells=pick(m,"one_parent",3)+pick(m,"independent",2)+pick(m,"one_region",2)+pick(m,"photo",2)
        else:
            lock=read(root/"freeze.json")
            if lock["revision"]!=rev or lock["full_sketchy_sha256"]!=checksum(root/"shortlist.json"):raise ValueError("Changed freeze")
            chosen,alternative,one,photo=(tuple(lock[k]) for k in ("primary","independent","one_region","photo"))
            n,t,_,eta,a,_,_=chosen
            cells=sorted({chosen,alternative,one,photo,(n,t,"base",eta,0.,"local","disjoint"),
                          *[(n,t,m,eta,a,"local","disjoint") for m in ("one_region","independent","photo","excess")],
                          (n,t,"one_parent",eta,a,"whole","disjoint"),(n,t,"one_parent",eta,a,"local","matched")})
        allstats={}
        for ds in (("sketchy","tuberlin") if stage=="final" else ("sketchy",)):
            cfg,qp,qs,pp,gs,qraw,praw=global_inputs(w,ds)
            words=read_words(cfg.vocabulary)
            if len(words)!=20931:raise ValueError("Changed public vocabulary")
            src,tgt=(v.cuda() for v in load_words(cfg,words));_,scale=bare_words(cfg,words)
            pviews=torch.load(w/"cache/nested-scale-word-evidence"/ds/"photo.pt",weights_only=True,mmap=True)
            qviews=torch.load(w/"cache"/SOURCE/ds/"views.pt",weights_only=True,mmap=True)
            for s,(qi,y) in qs.items():
                pi,g=gs[s]
                if (len(qi),len(pi),len(g.unique()))!=SIZES[s] or digest(UNSEEN_CLASSES[s])!=SPLIT_HASHES[s]:raise ValueError("Wrong dataset")
                dest=root/(s+".json" if stage=="final" else stage+"-"+s+".json")
                if dest.exists():
                    old=read(dest)
                    if old["signature"] != sig:
                        raise ValueError("Invalid saved checkpoint signature")
                    if stage in ("pilot", "final") and old["per_query_sha256"] != checksum(dest.with_suffix(".pt")):
                        raise ValueError("Invalid saved per-query checkpoint")
                    allstats[s]=old["values"];continue
                p=praw[pi].cuda();pv=pviews[pi]
                grp=cluster_assignments(F.normalize(p-p.mean(0)),64)
                _,grp=grp.unique(sorted=True,return_inverse=True)
                ident=digest({"run":sig,"gallery":digest([str(pp[i]) for i in pi.tolist()]),"vocabulary":checksum(cfg.vocabulary)})[:16]
                heads,compact=compact_index(w,s,p,tgt,scale,ident)
                ns=sorted(set(c[0] for c in cells))
                order=vocabulary(pv,tgt,scale,max(ns))
                terms={n:order[:n] for n in ns}
                bank={n:photo_words(p,pv,tgt[terms[n]],scale) for n in ns}
                hist=torch.load(w/"cache/pragmatic-gallery-competition"/(s+"-named-rgb16.pt"),weights_only=True,mmap=True)
                if hist.shape!=(len(qi),len(pi)):raise ValueError("Wrong historical base")
                ids=stratified_sample(y,36,63241) if stage=="pilot" else torch.arange(len(qi))
                pq=torch.stack((qviews["original_regions"][qi[ids]],qviews["regions"][qi[ids]]),1)
                rows,err,top,diagnostics=measure(pq,src,scale,heads,compact,bank,terms,grp,hist,y,g,ids,cells,s,
                                                 kd if stage=="final" else None,workers,key(chosen) if stage=="final" else None)
                vals={k:v.mean(0).tolist() for k,v in rows.items()};allstats[s]=vals
                obj={"signature":sig,"values":vals,"queries":len(ids),"gallery":len(pi),"null_checks":diagnostics,
                     "legal_sets_by_cardinality":[len(x) for x in designs()["disjoint"]]}
                if stage == "pilot":
                    path = dest.with_suffix(".pt")
                    save_tensor({"metrics": rows, "query_labels": y[ids], "split_query_ids": ids}, path)
                    obj["per_query_sha256"] = checksum(path)
                if stage=="final":
                    path=root/(s+".pt");save_tensor({"metrics":rows,"query_labels":y,"top25":top},path)
                    obj.update({"protocol":{"split":s,"queries":len(qi),"gallery":len(pi),"classes":len(g.unique()),"checkpoint":CHECKPOINT,
                                     "vocabulary_sha256":checksum(cfg.vocabulary),"kd_sha256":KD_FUNCTION_SHA,"revision":rev,"gpu":torch.cuda.get_device_name(),
                                     "map_k":200 if s=="sketchy_2" else 0,"p_k":200 if s=="sketchy_2" else 100},
                                "scores":{k:dict(zip(("mAP","precision"),v)) for k,v in vals.items()},
                                "primary":key(chosen),"best_contrast":key(alternative),"matched_base":key(one),"best_photo":key(photo),
                                "baseline":key((chosen[0],chosen[1],"base",chosen[3],0.,"local","disjoint")),
                                "vs_controls":{k:paired_delta(rows[key(chosen)],v,y) for k,v in rows.items()},
                                "kd_max_abs_error":err,"original_kd_queries_per_condition":len(qi),"per_query_sha256":checksum(path)})
                atomic_json(dest,obj);leader=max(vals,key=vals.get);print(stage,s,leader,vals[leader],flush=True)
        if stage!="final":
            m={k:[sum(allstats[s][k][i] for s in SPLITS)/2 for i in (0,1)] for k in allstats[SPLITS[0]]}
            atomic_json(root/(stage+".json"),{"splits":allstats,"means":m})
            if stage=="shortlist":
                lock={"primary":pick(m,"one_parent",1)[0],"independent":pick(m,"independent",1)[0],"one_region":pick(m,"one_region",1)[0],
                      "photo":pick(m,"photo",1)[0],"revision":rev,"full_sketchy_sha256":checksum(root/"shortlist.json")}
                atomic_json(root/"freeze.json",lock|{"sha256":digest(lock)})


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    a.add_argument("--phase",choices=("pilot","lock","audit","all"),default="all")
    a.add_argument("--workers",type=int,default=8)
    args=a.parse_args();run(args.workspace,args.phase,args.workers)
