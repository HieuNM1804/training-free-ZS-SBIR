"""Test query-conditioned omission of unexpressed concepts against calibrated linear readout."""

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
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import load_kd,paired_delta
from .selective_concept_listener import compile_index,scores,validate
from .semantic import read_words
from .shared_concept_likelihood import fit_prior


FAMILY="selective-concept-listener"


def key(c):return json.dumps(list(c),separators=(",",":"))


def grid():
    return [(k,index,temp,n,"query",mix) for k in (64,192) for index in ("word","word_rgb","visual_rgb")
            for temp in (.4,1.) for n in (1,2,5,12) for mix in (.4,.85)]


def choose(table,nonlinear,k):
    return [tuple(json.loads(item)) for item in sorted((a for a in table if (json.loads(a)[3]>1)==nonlinear),
                                                        key=table.get,reverse=True)[:k]]


def measure(w,q,y,p,g,source,target,scale,prior,base,ids,split,variants,replay,workers):
    per,tops,err,checks={},{},{},{}
    def record(config,s,labels,pool):
        if not torch.isfinite(s).all(): raise ValueError("Nonfinite rank score")
        score,rank,_,_=map_values(s,labels.cuda(),g.cuda(),split)
        nm=key(config)
        if replay:
            kd=replay.score(rank,labels,pool)
            delta=float((score.cpu()-kd).abs().max())
            if delta>2e-6: raise ValueError("KD mismatch")
            err[nm]=max(err.get(nm,0.),delta)
            tops.setdefault(nm,[]).append(rank[:,:25].int().cpu())
        else: kd=score.cpu()
        per.setdefault(nm,[]).append(kd)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if replay:
            for group in ids.split(96):record(("historical",),base[group].cuda(),y[group],pool)
        for k,index in sorted(set(c[:2] for c in variants)):
            word,b,m=compile_index(p,target,scale,prior,k,index)
            checks[key((k,index))]=validate(q,source[word],b,m,scale)
            these=[c for c in variants if c[:2]==(k,index)]
            for temp,order in sorted(set((c[2],c[4]) for c in these)):
                cells=[c for c in these if c[2]==temp and c[4]==order]
                ns=sorted(set(c[3] for c in cells))
                for group in ids.split(48):
                    logbase=base[group].cuda().clamp_min(1e-30).log()
                    qq=q[group]
                    if order=="flat_likelihood":
                        # Keep each per-photo distribution intact but collapse all
                        # non-peak query utility to compare with single-word matching.
                        top=(qq@source[word].T).argmax(1)
                        raw={n:-torch.expm1(n*torch.log1p(-b[:,top].T.clamp(0,1))) for n in ns}
                    elif order=="flatten_photo":
                        bflat=b.sqrt()
                        bflat=bflat/bflat.sum(1,keepdim=True)
                        raw=scores(qq,source[word],bflat,m,scale,temp,ns,"query")
                    else:
                        raw=scores(qq,source[word],b,m,scale,temp,ns,order)
                    for c in cells:
                        v=raw[c[3]].clamp_min(1e-30).log()
                        record(c,(1-c[5])*logbase+c[5]*v,y[group],pool)
            print(split,k,index,"complete",flush=True)
    return {c:torch.cat(v) for c,v in per.items()},err,checks,{k:torch.cat(v) for k,v in tops.items()}


@torch.inference_mode()
def run(w,workers):
    folder=w/"results"/FAMILY
    protocol={"revision":revision(),"candidate_grid":grid(),"primary":"best among multiple expression opportunities",
              "control":"optimally tuned single draw","pilot":"24 unseen sketches per class seed 226139"}
    atomic_json(folder/"protocol.json",protocol|{"sha256":digest(protocol)})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("KD changed")
    for stage in ("pilot","shortlist","final"):
        if stage=="pilot": variants=grid()
        elif stage=="shortlist":
            x=read(folder/"pilot.json")["means"]
            variants=choose(x,True,3)+choose(x,False,2)
        else:
            x=read(folder/"shortlist.json")["means"]
            best=choose(x,True,1)[0]
            other=choose(x,False,1)[0]
            lock={"revision":revision(),"primary":best,"optimized_linear":other,
                  "sketchy_full_sha256":checksum(folder/"shortlist.json")}
            atomic_json(folder/"freeze.json",lock|{"sha256":digest(lock)})
            variants=sorted({best,other,(*best[:3],1,"query",best[5]),(*best[:5],1.),
                             (*best[:4],"prior",best[5]),(*best[:4],"shuffled",best[5]),
                             (*best[:3],1,"flatten_photo",best[5]),(*best[:4],"flat_likelihood",best[5])})
        tables={}
        for dataset in (("sketchy","tuberlin") if stage=="final" else ("sketchy",)):
            cfg,_,qs,_,gs,qraw,praw=global_inputs(w,dataset)
            terms=read_words(cfg.vocabulary)
            if len(terms)!=20931: raise ValueError("Gallery dictionary changed")
            _,scale=bare_words(cfg,terms)
            source,target=(v.cuda() for v in load_words(cfg,terms))
            for split,(qi,y) in qs.items():
                pi,g=gs[split]
                if (len(qi),len(pi),len(g.unique()))!=SIZES[split] or digest(UNSEEN_CLASSES[split])!=SPLIT_HASHES[split]:
                    raise ValueError("Unexpected split")
                p,q=praw[pi].cuda(),qraw[qi].cuda()
                prob=(scale*(p@target.T)).softmax(1)
                pset,_=fit_prior(prob)
                del prob
                base=torch.load(w/"cache/pragmatic-gallery-competition"/(split+"-named-rgb16.pt"),weights_only=True,mmap=True)
                if base.shape!=(len(q),len(p)):raise ValueError("Historical score geometry changed")
                ix=stratified_sample(y,24,226139) if stage=="pilot" else torch.arange(len(y))
                per,err,checks,tops=measure(w,q,y,p,g,source,target,scale,pset["mle"],base,ix,split,variants,
                                           Replay(kd,g,split) if stage=="final" else None,workers)
                tables[split]={k:v.mean(0).tolist() for k,v in per.items()}
                z=max(tables[split],key=tables[split].get)
                print(stage,split,z,tables[split][z],flush=True)
                if stage=="final":
                    path=folder/(split+".pt")
                    save_tensor({"metrics":per,"query_labels":y,"top25":tops[key(best)]},path)
                    info={"protocol":{"split":split,"queries":len(q),"gallery":len(p),"classes":len(g.unique()),
                                      "checkpoint":CHECKPOINT,"vocabulary_sha256":checksum(cfg.vocabulary),"kd_sha256":KD_FUNCTION_SHA,
                                      "revision":revision(),"gpu":torch.cuda.get_device_name(),"map_k":200 if split=="sketchy_2" else 0,
                                      "p_k":200 if split=="sketchy_2" else 100},
                          "scores":{k:dict(zip(("mAP","precision"),v)) for k,v in tables[split].items()},
                          "primary":key(best),"optimized_linear":key(other),"baseline":key(("historical",)),
                          "vs_controls":{k:paired_delta(per[key(best)],v,y) for k,v in per.items()},
                          "exact_identities":checks,"original_kd_queries_per_condition":len(q),"kd_max_abs_error":err,"per_query_sha256":checksum(path)}
                    atomic_json(folder/(split+".json"),info)
        if stage!="final":
            means={k:[sum(tables[s][k][i] for s in ("sketchy_1","sketchy_2"))/2 for i in range(2)] for k in tables["sketchy_1"]}
            atomic_json(folder/(stage+".json"),{"splits":tables,"means":means})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--workers",type=int,default=min(10,len(os.sched_getaffinity(0))))
    args=parser.parse_args()
    run(args.workspace,args.workers)
