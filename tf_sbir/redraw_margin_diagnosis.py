"""Post-hoc check of rival-margin stability using repeated independent drawings."""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
import torch

from .concept_benchmark import atomic_json,checksum
from .conditional_semantic_index import bare_words
from .foundation import load_words
from .redraw_margins import group_inputs,read_group
from .retrieval_diagnosis import global_inputs
from .semantic import read_words


def metrics(confidence,success,order):
    c=confidence.double().cpu().numpy().ravel()
    s=success.double().cpu().numpy().ravel()
    def rank_corr(a,b):
        if len(np.unique(a))<2 or len(np.unique(b))<2:return np.nan
        return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
    ix=np.argsort(c,kind="stable")
    # Every independent query-rival comparison has weight one. Fractional
    # positives are the observed fraction of its other drawings that agree.
    neg=1-s[ix];pos=s[ix]
    grouped=np.split(np.arange(len(ix)),np.flatnonzero(np.diff(c[ix]))+1)
    prior=0.;num=0.
    for sub in grouped:
        n=neg[sub].sum()
        num+=pos[sub].sum()*(prior+n/2)
        prior+=n
    return {"weighted_auc":float(num/(pos.sum()*neg.sum())),
            "rank_correlation":rank_corr(c,s),
            "mean_within_margin_decile_rank_correlation":float(np.nanmean([rank_corr(c[p],s[p]) for p in np.array_split(order,10)]))}


def compare(s,score,cv,qi,qp,pi,pp):
    top=score.topk(5,dim=1).indices
    r=torch.arange(len(qi),device=score.device)[:,None]
    first=top[:,:1].expand(-1,4)
    rival=top[:,1:5]
    d=score[r,first]-score[r,rival]
    own=cv.diagonal(dim1=-2,dim2=-1)
    paired=(own[r,first]+own[r,rival]-2*cv[r,first,rival]).clamp_min(1e-12).sqrt()
    independent=(own[r,first]+own[r,rival]).clamp_min(1e-12).sqrt()
    perm=torch.randperm(len(qi),generator=torch.Generator().manual_seed(72311)).to(score.device)
    other=(own[perm[:,None],first]+own[perm[:,None],rival]-2*cv[perm[:,None],first,rival]).clamp_min(1e-12).sqrt()
    objects={}
    gallery={(Path(pp[i]).parent.name,Path(pp[i]).stem) for i in pi.tolist()}
    for j,i in enumerate(qi.tolist()):
        path=Path(qp[i]);source=(path.parent.name,path.stem.rpartition("-")[0])
        if source not in gallery:raise ValueError("Unconfirmed photo identity")
        objects.setdefault(source,[]).append(j)
    score_cpu=score.cpu()
    votes=torch.zeros((len(qi),4))
    used=torch.zeros(len(qi),dtype=torch.bool)
    first=first.cpu();rival=rival.cpu()
    for members in objects.values():
        if len(members)<2:continue
        ids=torch.tensor(members)
        peers=score_cpu[ids]
        positive=peers[:,first[ids]]>peers[:,rival[ids]]
        # The current sketch's preference is itself strictly positive.
        selfvote=positive[torch.arange(len(ids)),torch.arange(len(ids))].float()
        votes[ids]=(positive.float().sum(0)-selfvote)/(len(ids)-1)
        used[ids]=True
    y=votes[used]
    q=d.cpu()[used]
    order=np.argsort(q.numpy().ravel(),kind="stable")
    conditions={"margin":d,"paired":d/paired,"independent":d/independent,"unrelated_variance":d/other}
    return {"split":s,"queries":len(qi),"sibling_queries":int(used.sum()),"comparisons":y.numel(),
            "mean_sibling_agreement":float(y.mean()),"rival_positions":[2,3,4,5],"groups":score.size(1),
            "median_paired_to_independent_sd":float((paired/independent).median()),
            "results":{k:metrics(v.cpu()[used],y,order) for k,v in conditions.items()}}


@torch.inference_mode()
def run(w):
    cfg,qp,qs,pp,gs,qraw,praw=global_inputs(w,"sketchy")
    words=read_words(cfg.vocabulary);src,tgt=(v.cuda() for v in load_words(cfg,words))
    _,scale=bare_words(cfg,words)
    views=torch.load(w/"cache/regional-evidence-interventions/sketchy/regions.pt",mmap=True,weights_only=True)
    root=w/"results/redraw-rank-stability"
    data={}
    for s,(qi,y) in qs.items():
        pi,g=gs[s]
        bank=group_inputs(w,s,praw[pi].cuda(),pi,pp,tgt,scale,checksum(cfg.vocabulary))
        probs=[];cov=[]
        for part in qi.split(56):
            ix=torch.searchsorted(views["ids"],part)
            if not torch.equal(views["ids"][ix],part):raise ValueError("Wrong query window order")
            q=qraw[part].cuda();z=views["features"][ix,:,2].cuda().float()
            c,v,_=read_group(q,z,src,scale,bank);probs.append(c);cov.append(v)
        score=torch.cat(probs); cov=torch.cat(cov)
        data[s]=compare(s,score,cov,qi,qp,pi,pp)
        atomic_json(root/"sibling-margin-diagnosis.json",data)
        print(s,data[s],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
