"""Measure whether FreeDom's first retrieval pass pulls image descriptions off target."""

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,read
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import revision
from .foundation import load_words
from .map_diagnosis import map_values
from .map_study import Data
from .map_transfer_audit import import_measured
from .retrieval_diagnosis import global_inputs,stratified_sample
from .semantic import read_words


def measure(index,config,ql,pl,split):
    results={}
    for key,cfg in (("original_rgb",config),("semantic_proxy",replace(config,search="semantic"))):
        photos=index.get_neighbors(cfg.search)[:,:cfg.proxies-1]
        relevance=pl[photos]==ql[:,None]
        details,_,correct,_=map_values(index.score(cfg),ql,pl,split)
        results[key]=(relevance.float().mean(1),details[:,0],correct[:,0])
    (rgb_p,rgb_ap,rgb_correct),(sem_p,sem_ap,sem_correct)=results.values()
    improve=sem_p>rgb_p
    worse=sem_p<rgb_p
    no_change=sem_p==rgb_p
    def conditional(mask):
        delta=(sem_ap-rgb_ap)[mask]
        return {"count":len(delta),"mean_AP_gain_pp":100*float(delta.mean()) if len(delta) else None}
    return {"mean_proxy_fraction_relevant":{"rgb":float(rgb_p.mean()),"semantic":float(sem_p.mean())},
            "mean_ap":{"rgb":float(rgb_ap.mean()),"semantic":float(sem_ap.mean())},
            "rgb_zero_relevant_proxies":int((rgb_p==0).sum()),"semantic_zero_relevant_proxies":int((sem_p==0).sum()),
            "more_relevant_proxies":conditional(improve),"fewer_relevant_proxies":conditional(worse),
            "same_relevant_proxies":conditional(no_change),
            "top1_rescued":int((~rgb_correct&sem_correct).sum()),
            "top1_lost":int((rgb_correct&~sem_correct).sum())}


@torch.inference_mode()
def audit(w):
    root=w/"results"
    name="freedom-gallery-transfer"
    lock=read(root/name/"freeze.json")
    method=import_measured(w,name,lock)
    args,_,queries,_,gallery,raw,rgb=global_inputs(w,"sketchy")
    words=read_words(args.vocabulary)
    source,target=(v.cuda() for v in load_words(args,words))
    _,scale=bare_words(args,words)
    report={"revision":revision(),"family_revision":lock["revision"],"seed":55429,
            "proxies":20,"split":{}}
    for split,(qi,ql) in queries.items():
        selected=stratified_sample(ql,48,55429)
        pi,pl=gallery[split]
        data=Data(raw[qi[selected]].cuda(),rgb[pi].cuda(),source,target,scale)
        method.prepare_data(data,args,words)
        results=measure(method.Index(data),method.canonical(),ql[selected].cuda(),pl.cuda(),split)
        report["split"][split]=results
        print(split,results,flush=True)
        del data
    atomic_json(root/"map-followup/freedom-proxy-audit.json",report)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    audit(parser.parse_args().workspace)
