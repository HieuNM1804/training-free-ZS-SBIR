"""Hold proxies and the '<word> photo' scoring rule fixed and change only vocabulary votes."""

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from . import freedom_gallery_transfer as freedom
from .concept_benchmark import atomic_json, read
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import revision
from .foundation import load_words
from .map_diagnosis import map_values
from .map_study import Data
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import paired_delta
from .semantic import read_words
from .shared_concept_likelihood import target_inputs


@torch.inference_mode()
def run(w):
    args,_,queries,_,galleries,raw,rgb=global_inputs(w,"sketchy")
    words=read_words(args.vocabulary)
    source,target=(v.cuda() for v in load_words(args,words))
    _,scale=bare_words(args,words)
    old=read(w/"results/map-followup/freedom-proxy-audit.json")
    report={"revision":revision(),"seed":55429,"proxies":20,"new_hyperparameter_selection":False,"splits":{}}
    for split,(qi,ql) in queries.items():
        ids=stratified_sample(ql,48,55429)
        pi,pl=galleries[split]
        labels=ql[ids]
        d=Data(raw[qi[ids]].cuda(),rgb[pi].cuda(),source,target,scale)
        freedom.prepare_data(d,args,words)
        index=freedom.Index(d)
        cfg=replace(freedom.canonical(),search="semantic")
        same=index.get_neighbors("semantic")[:,:cfg.proxies-1]
        def stats(scores,first=0):
            return map_values(scores,labels[first:first+len(scores)].cuda(),pl.cuda(),split)[0].cpu()
        reference=stats(index.score(cfg))
        if abs(float(reference[:,0].mean())-old["split"][split]["mean_ap"]["semantic"])>1e-5:
            raise ValueError("The fixed first pass did not reproduce the proxy audit")
        photo_soft=target_inputs(d.p,d.bare,scale)
        photo_soft/=photo_soft.sum(1,keepdim=True)
        query_soft=(scale*(d.q@d.bare.T)).softmax(1)
        modes={key:[] for key in ("hard_all","soft_top7","soft_all")}
        for first in range(0,len(ids),64):
            near=same[first:first+64]
            count=len(near)
            hard_ids=torch.cat((index.query_words[first:first+count,None,:],index.photo_words[near]),dim=1).flatten(1)
            hard=d.p.new_zeros((count,len(words)))
            hard.scatter_add_(1,hard_ids,torch.ones_like(hard_ids,dtype=torch.float32))
            soft=(photo_soft[near].sum(1)+query_soft[first:first+count])/cfg.proxies
            mass,top=soft.topk(7,dim=1)
            reduced=torch.zeros_like(soft).scatter_(1,top,mass)
            for name,dist in (("hard_all",hard),("soft_top7",reduced),("soft_all",soft)):
                # Normalize by the largest weight as in FreeDom. The scale
                # is common to all photos for a given sketch.
                score=((dist/dist.amax(1,keepdim=True))@d.photo_composed)@d.p.T
                modes[name].append(stats(score,first))
        details={k:torch.cat(v) for k,v in modes.items()}
        details["hard_top7"]=reference
        contrasts=(("hard_all","hard_top7"),("soft_top7","hard_top7"),("soft_all","hard_all"),("soft_all","soft_top7"))
        report["splits"][split]={"queries":len(ids),"gallery":len(pl),
            "scores":{k:{"mAP":float(v[:,0].mean()),"precision":float(v[:,1].mean())} for k,v in details.items()},
            "paired":{a+"_minus_"+b:paired_delta(details[a],details[b],labels) for a,b in contrasts}}
        print(split,report["splits"][split]["scores"],flush=True)
        del d,index,photo_soft
    atomic_json(w/"results/map-followup/freedom-vote-ablation.json",report)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
