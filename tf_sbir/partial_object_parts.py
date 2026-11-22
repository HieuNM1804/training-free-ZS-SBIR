"""Marginalize a latent set of visible regions, with fixed-marginal spatial controls."""

from collections import Counter
from itertools import combinations
from math import comb

import torch

from .regional_interventions import SIDES,windows


def designs():
    positions=[(side,y,x) for side in SIDES for y,x in windows(side)]
    occupied=[{(i,j) for i in range(y,y+side) for j in range(x,x+side)} for side,y,x in positions]
    orbit=[(side,*sorted((abs(2*y+side-7),abs(2*x+side-7)))) for side,y,x in positions]
    bound=[];control=[]
    for m in range(1,5):
        all_sets=list(combinations(range(len(positions)),m))
        legal=[s for s in all_sets if len(set.union(*(occupied[r] for r in s)))==sum(len(occupied[r]) for r in s)]
        def pattern(s):return tuple(sorted(orbit[r] for r in s))
        allowed=Counter(pattern(s) for s in legal)
        total=Counter(pattern(s) for s in all_sets)
        bound.append([(s,1/len(legal)) for s in legal])
        balanced=[(s,allowed[pattern(s)]/(len(legal)*total[pattern(s)])) for s in all_sets if pattern(s) in allowed]
        control.append(balanced)
        for r in range(len(positions)):
            real=sum(p for s,p in bound[-1] if r in s)
            other=sum(p for s,p in balanced if r in s)
            if abs(real-other)>1e-12:raise ValueError("Spatial control changes a region's inclusion probability")
    return {"disjoint":bound,"matched":control}


def vocabulary(crops,target,scale,k):
    rng=torch.Generator().manual_seed(31039)
    rows=torch.randperm(len(crops),generator=rng)[:2048]
    total=target.new_zeros(len(target),dtype=torch.float64)
    entropy=torch.zeros_like(total)
    for idx in rows.split(32):
        v=crops[idx].cuda().float()
        p=(v@target.T*scale).softmax(-1)
        m=(p[:,:9].mean(1)+p[:,9:].mean(1))*.5
        total+=m.sum(0,dtype=torch.float64)
        entropy+=(m*m.clamp_min(1e-30).log()).sum(0,dtype=torch.float64)
    mean=total/len(rows)
    info=entropy/len(rows)-mean*mean.clamp_min(1e-30).log()
    return info.topk(k).indices


def photo_words(raw,zooms,target,scale):
    regional=[]
    for part in zooms.split(160):
        probs=(part.cuda().float()@target.T*scale).softmax(-1)
        regional.append(.5*(probs[:,:9].mean(1)+probs[:,9:].mean(1)))
    return {"local":torch.cat(regional),"whole":(raw@target.T*scale).softmax(-1)}


def region_ratios(q,words,target,scale,temp):
    v=(q@words.T)*scale*temp
    e=(v-v.amax((1,3),keepdim=True)).exp().mean(1)
    l=e@target.T
    return l/l.mean(2,keepdim=True).clamp_min(1e-30)


def average(x,g):
    flat=x.reshape(-1,len(g))
    out=x.new_zeros((flat.size(0),int(g.max())+1))
    out.index_add_(1,g,flat)
    out/=g.bincount()
    return out.view(*x.shape[:-1],out.size(1))


def moments(x,sets):
    out=[torch.ones_like(x[:,0])]
    for groups in sets:
        value=torch.zeros_like(x[:,0])
        for positions,weight in groups:
            part=x[:,positions[0]]
            for r in positions[1:]:part=part*x[:,r]
            value.add_(part,alpha=weight)
        out.append(value)
    return torch.stack(out,1)


def evidence(moments,visible):
    weights=moments.new_tensor([comb(4,m)*visible**m*(1-visible)**(4-m) for m in range(5)])
    return (moments*weights[None,:,None]).sum(1).clamp_min(1e-30).log()/(4*visible)


def check(x,sets):
    y=x[:2,:,:12].double()
    e=moments(torch.ones_like(y),sets)
    neutral=float((e-1).abs().max())
    group=torch.zeros(y.size(2),dtype=torch.long,device=y.device)
    m=average(moments(y,sets),group);d=moments(average(y,group),sets)
    common=float((m[:,:2]-d[:,:2]).abs().max())
    if max(neutral,common)>3e-10:raise ValueError("Partial evidence does not preserve the null or singleton")
    return {"neutral_error":neutral,"singleton_same_parent_error":common}
