"""Measure the positions and neighborhood support of missed relevant photos on Sketchy."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json,read
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import revision
from .foundation import load_words
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import tensor
from .semantic import read_words
from .shared_concept_likelihood import FAMILY,PRIMARY,emission,fit_prior,posterior,target_inputs


def map_values(scores,labels,photo_labels,split):
    order=scores.argsort(dim=1,descending=True,stable=True)
    target=labels[:,None]==photo_labels[order]
    k=200 if split=="sketchy_2" else 100
    scope=target[:,:200] if split=="sketchy_2" else target
    pos=torch.arange(1,scope.size(1)+1,device=scores.device)
    precision=scope.cumsum(1)/pos
    terms=scope*precision/scope.sum(1,keepdim=True).clamp_min(1)
    return torch.stack((terms.sum(1),target[:,:k].float().mean(1)),1),order,target,terms


def graph_scores(scores,p,temperature):
    # A single leave-one-photo-out conditional expectation, used for diagnosis.
    blocks=[]; density=[]
    for start in range(0,len(p),384):
        stop=min(start+384,len(p)); rows=torch.arange(start,stop,device=p.device)
        similarity=p[start:stop]@p.T
        similarity[torch.arange(len(rows),device=p.device),rows]=-torch.inf
        weights=(temperature*similarity).softmax(1)
        blocks.append(scores@weights.T)
        density.append((-(weights*weights.clamp_min(1e-35).log()).sum(1)).exp())
    return torch.cat(blocks,1),torch.cat(density)


def by_rank(base,labels,pl,split,neighbor=None):
    details,order,target,terms=map_values(base,labels,pl,split)
    out={"mAP":float(details[:,0].mean()),"precision":float(details[:,1].mean()),
         "top1":float(target[:,0].float().mean()),
         "top1_correct_but_ap_below_half":int((target[:,0]&(details[:,0]<.5)).sum()),"bins":[]}
    sizes=target.sum(1).clamp_min(1)
    for left,right in ((0,10),(10,100),(100,200),(200,1000),(1000,target.size(1))):
        take=target[:,left:right]
        out["bins"].append({"from_rank":left+1,"to_rank":right,
            "fraction_of_all_relevant":float((take.sum(1)/sizes).mean()),
            "contribution_to_official_AP":float(terms[:,left:right].sum(1).mean()) if left<terms.size(1) else 0.})
    if neighbor is not None:
        # How many false items preceding the median relevant photo are repaired?
        positive=order.gather(1,(target.cumsum(1)>=(sizes[:,None]+1)//2).int().argmax(1)[:,None]).squeeze(1)
        first_bad=order.gather(1,(~target).int().argmax(1)[:,None]).squeeze(1)
        idx=torch.arange(len(labels),device=base.device)
        was=base[idx,positive]<base[idx,first_bad]
        out["median_positive_vs_first_wrong"]={"originally_below":int(was.sum()),
            "neighbor_puts_positive_above":int((was&(neighbor[idx,positive]>neighbor[idx,first_bad])).sum())}
    return out


@torch.inference_mode()
def run(w):
    report={"revision":revision(),"seed":37121,"splits":{}}
    args,qp,queries,pp,galleries,raw,rgb=global_inputs(w,"sketchy")
    words=read_words(args.vocabulary);_,scale=bare_words(args,words)
    src,tgt=(x.cuda() for x in load_words(args,words))
    for split,(qi,ql) in queries.items():
        pi,pl=galleries[split]; selected=stratified_sample(ql,40,37121)
        q,p=raw[qi[selected]].cuda(),rgb[pi].cuda()
        labels,glabels=ql[selected].cuda(),pl.cuda()
        em=target_inputs(p,tgt,scale); priors,_=fit_prior(em)
        observations=emission(q,src,scale)
        scores=observations@posterior(em,priors["mle"]).T
        uniform=observations@posterior(em,priors["uniform"]).T
        both=torch.cat((scores,uniform),0)
        pred,neff=graph_scores(both,p,scale)
        neigh,flat=pred.split(len(q))
        null=neigh[:,torch.randperm(len(p),generator=torch.Generator().manual_seed(869)).cuda()]
        result={"sketches":len(q),"gallery":len(p),"native":by_rank(q@p.T,labels,glabels,split),
                "semantic":by_rank(scores,labels,glabels,split,neigh),"uniform":by_rank(uniform,labels,glabels,split),
                "neighbor_semantic":by_rank(neigh,labels,glabels,split),"neighbor_uniform":by_rank(flat,labels,glabels,split),
                "shuffled_photo_assignment":by_rank(null,labels,glabels,split),
                "median_effective_neighbors_at_clip_scale":float(neff.median())}
        saved=tensor(w/"results"/FAMILY/"sketchy"/"per-query"/f"{split}-{PRIMARY}.pt")
        measured=map_values(scores,labels,glabels,split)[0].cpu()
        result["vector_vs_saved_max_AP_P"]=(measured-saved[selected]).abs().amax(0).tolist()
        fulltop=tensor(w/"results"/FAMILY/"sketchy"/f"{split}-top10.pt")[PRIMARY].long()
        correct=pl[fulltop[:,0]]==ql
        result["full_queries"]={"count":len(ql),"top1_correct":int(correct.sum()),
            "top1_correct_AP_below_half":int((correct&(saved[:,0]<.5)).sum()),
            "top1_correct_AP_below_075":int((correct&(saved[:,0]<.75)).sum())}
        report["splits"][split]=result
        print(split,{k:(v["mAP"],v["precision"]) for k,v in result.items() if isinstance(v,dict) and "mAP" in v},
              result["full_queries"],flush=True)
    atomic_json(w/"results/map-study/diagnosis.json",report)


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__);a.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(a.parse_args().workspace)
