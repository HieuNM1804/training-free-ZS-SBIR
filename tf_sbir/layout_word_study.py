"""Full-gallery selection and falsification of word-conditioned visual layout evidence."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,checksum,read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES,bare_words
from .crossdomain_benchmark import digest,revision,SPLIT_HASHES
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .layout_word_cache import FAMILY
from .layout_word_routes import layout_evidence,RouteIndex
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import load_kd,paired_delta
from .semantic import read_words


def key(c):return json.dumps(list(c),separators=(",",":"))


def choices():
    variants=[]
    for prior in ("uniform","mle"):
        variants.extend((prior,"route",v,t,"all") for v in (.25,.7,1.) for t in (.4,1.,2.))
        variants.extend((prior,c,v,1.,"all") for c in ("contrast_both","contrast_query","contrast_target")
                        for v in (.25,.7,1.3))
    return variants


def choose(table,route,k):
    return [tuple(json.loads(x)) for x in sorted((k for k in table if (json.loads(k)[1]=="route")==route),
                                                 key=table.get,reverse=True)[:k]]


def measure(w,dataset,split,q,y,qi,p,g,pi,banks,scale,ids,variants,reference,workers):
    base=w/"cache"/FAMILY/dataset
    qperm=torch.load(base/"query.pt",mmap=True,weights_only=True)[qi[ids]].cuda()
    pperm=torch.load(base/"gallery.pt",mmap=True,weights_only=True)[pi].cuda()
    per,err,top,checks={}, {},{},{}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for prior,part in sorted(set((c[0],c[-1]) for c in variants)):
            where={"all":slice(None),"first2":slice(0,2),"last2":slice(2,4)}[part]
            model=RouteIndex(p,pperm[:,where],banks[1],scale,prior)
            chunks=[]
            for i,group in enumerate(ids.split(128)):
                l,a=layout_evidence(q[group],qperm[i*128:i*128+len(group),where],banks[0],scale)
                chunks.append((l.cpu(),a.cpu()))
            for c in [c for c in variants if c[0]==prior and c[-1]==part]:
                _,kind,value,temperature,_=c
                model.make(kind,value,temperature)
                nm=key(c)
                for i,(l,a) in enumerate(chunks):
                    l,a=l.cuda(),a.cuda()
                    labels=y[ids[128*i:128*i+len(l)]]
                    raw=model.score(l,a)
                    fast,order,_,_=map_values(raw,labels.cuda(),g.cuda(),split)
                    if i==0:checks[nm]=model.check_routes(l[:4],a[:4])
                    if reference:
                        official=reference.score(order,labels,pool)
                        e=float((fast.cpu()-official).abs().max())
                        if e>2e-6:raise ValueError("KD metric mismatch")
                        err[nm]=max(err.get(nm,0.),e)
                        top.setdefault(nm,[]).append(order[:,:25].int().cpu())
                    else: official=fast.cpu()
                    per.setdefault(nm,[]).append(official)
            print(split,"complete",prior,part,len(variants),flush=True)
            del chunks,model
    return {k:torch.cat(v) for k,v in per.items()},err,checks,{k:torch.cat(v) for k,v in top.items()}


@torch.inference_mode()
def run(w,workers,resume_tuberlin=False):
    folder=w/"results"/FAMILY
    protocol={"revision":revision(),"grid":choices(),"primary":"matched visual evidence routes","challenger":"best one- or two-side contrast",
              "pilot":"24 sketches per class seed 48619","selection":"equal-weight mean Sketchy mAP then mean precision"}
    if not resume_tuberlin:
        atomic_json(folder/"protocol.json",protocol|{"sha256":digest(protocol)})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Changed KD evaluator")
    for stage in (("final",) if resume_tuberlin else ("pilot","shortlist","final")):
        if stage=="pilot":variants=choices()
        elif stage=="shortlist":
            prev=read(folder/"pilot.json")["means"]
            variants=choose(prev,True,3)+choose(prev,False,3)
        else:
            prev=read(folder/"shortlist.json")["means"]
            best=choose(prev,True,1)[0]
            contrast=choose(prev,False,1)[0]
            lock={"primary":best,"best_contrast":contrast,"revision":revision(),
                  "full_sketchy_sha256":checksum(folder/"shortlist.json")}
            if resume_tuberlin:
                frozen=read(folder/"freeze.json")
                if (tuple(frozen["primary"])!=best or tuple(frozen["best_contrast"])!=contrast
                        or frozen["full_sketchy_sha256"]!=lock["full_sketchy_sha256"]):
                    raise ValueError("The original Sketchy selection is no longer intact")
            else:
                atomic_json(folder/"freeze.json",lock|{"sha256":digest(lock)})
            variants={best,contrast,(best[0],"original",0.,1.,"all"),("uniform","original",0.,1.,"all")}
            for m in ("route_opposite","route_shuffle","flat_query","contrast_both","contrast_target","contrast_query"):
                variants.add((best[0],m,*best[2:]))
            variants.update((*best[:4],s) for s in ("first2","last2"))
            variants=sorted(variants)
        tables={}
        datasets=("tuberlin",) if resume_tuberlin else (("sketchy","tuberlin") if stage=="final" else ("sketchy",))
        for dataset in datasets:
            cfg,_,qs,_,gs,qraw,praw=global_inputs(w,dataset)
            terms=read_words(cfg.vocabulary)
            if len(terms)!=20931:raise ValueError("Changed word list")
            _,scale=bare_words(cfg,terms)
            banks=tuple(v.cuda() for v in load_words(cfg,terms))
            for split,(qi,y) in qs.items():
                pi,g=gs[split]
                if (len(qi),len(pi),len(g.unique()))!=SIZES[split] or digest(UNSEEN_CLASSES[split])!=SPLIT_HASHES[split]:
                    raise ValueError("Incomplete split")
                indices=stratified_sample(y,24,48619) if stage=="pilot" else torch.arange(len(qi))
                q,p=qraw[qi].cuda(),praw[pi].cuda()
                per,err,checks,top=measure(w,dataset,split,q,y,qi,p,g,pi,banks,scale,indices,variants,
                                           Replay(kd,g,split) if stage=="final" else None,workers)
                tables[split]={k:v.mean(0).tolist() for k,v in per.items()}
                k=max(tables[split],key=tables[split].get)
                print(stage,split,k,tables[split][k],flush=True)
                if stage=="final":
                    path=folder/(split+".pt")
                    save_tensor({"metrics":per,"query_labels":y,"top25":top[key(best)]},path)
                    rec={"protocol":{"split":split,"queries":len(q),"gallery":len(p),"classes":len(g.unique()),
                                     "checkpoint":CHECKPOINT,"vocabulary_sha256":checksum(cfg.vocabulary),"kd_sha256":KD_FUNCTION_SHA,
                                     "revision":revision(),"gpu":torch.cuda.get_device_name(),"map_k":200 if split=="sketchy_2" else 0,
                                     "p_k":200 if split=="sketchy_2" else 100},
                         "scores":{k:dict(zip(("mAP","precision"),v)) for k,v in tables[split].items()},
                         "primary":key(best),"best_contrast":key(contrast),"baseline":key(("uniform","original",0.,1.,"all")),
                         "matched_base":key((best[0],"original",0.,1.,"all")),
                         "vs_controls":{k:paired_delta(per[key(best)],v,y) for k,v in per.items()},
                         "best_contrast_vs_base":paired_delta(per[key(contrast)],per[key(("uniform","original",0.,1.,"all"))],y),
                         "layout_checks":checks,"encoder_checks":{role:read(w/"cache"/FAMILY/dataset/(role+".json")) for role in ("query","gallery")},
                         "original_kd_queries_per_condition":len(q),"kd_max_abs_error":err,"per_query_sha256":checksum(path)}
                    atomic_json(folder/(split+".json"),rec)
        if stage!="final":
            means={k:[sum(tables[s][k][i] for s in ("sketchy_1","sketchy_2"))/2 for i in range(2)] for k in tables["sketchy_1"]}
            atomic_json(folder/(stage+".json"),{"splits":tables,"means":means})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--workers",type=int,default=min(10,len(os.sched_getaffinity(0))))
    parser.add_argument("--resume-tuberlin",action="store_true")
    args=parser.parse_args()
    run(args.workspace,args.workers,args.resume_tuberlin)
