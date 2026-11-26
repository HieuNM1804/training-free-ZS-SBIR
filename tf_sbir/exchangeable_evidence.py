"""Integrate latent co-reference of soft (unlabeled) region observations exactly."""

from functools import lru_cache
from math import factorial

import torch


@lru_cache(None)
def partitions(n):
    if n==0:
        return ((),)
    bit=1<<(n-1)
    previous=partitions(n-1)
    result=[]
    for groups in previous:
        result.append((*groups,bit))
        for slot in range(len(groups)):
            result.append(tuple(g|bit if i==slot else g for i,g in enumerate(groups)))
    return tuple(result)


def coefficients(likelihood,theta):
    """Return the joint predictive likelihood conditional on 1,...,m distinct latent draws."""
    rows,views,terms=likelihood.shape
    if not 1<=views<=6 or terms!=theta.size(1):raise ValueError("Unsupported soft observation shape")
    all_parts=partitions(views)
    stage=[]
    for raw in likelihood.split(192):
        # One constant per observed view multiplies every complete partition equally.
        x=raw/raw.amax(-1,keepdim=True).clamp_min(1e-30)
        products={}
        for mask in range(1,1<<views):
            bit=mask&-mask
            single=x[:,bit.bit_length()-1]
            products[mask]=single if mask==bit else products[mask-bit]*single
        blocks=torch.stack(list(products.values()),0).reshape(-1,terms)@theta.T
        evidence=blocks.reshape((1<<views)-1,len(raw),len(theta))
        masses=[torch.zeros_like(evidence[0]) for _ in range(views)]
        totals=[0]*views
        for partition in all_parts:
            prob=evidence[partition[0]-1]
            weight=factorial(partition[0].bit_count()-1)
            for mask in partition[1:]:
                prob=prob*evidence[mask-1]
                weight*=factorial(mask.bit_count()-1)
            j=len(partition)-1
            masses[j].add_(prob,alpha=weight)
            totals[j]+=weight
        stage.append(torch.stack([v/t for v,t in zip(masses,totals,strict=True)],1))
    return torch.cat(stage),totals


def interpolate(coeff,totals,alpha):
    if alpha==0:return coeff[:,0]
    if alpha==float("inf"):return coeff[:,-1]
    weight=coeff.new_tensor([alpha**j*n for j,n in enumerate(totals,start=1)])
    return (coeff*(weight/weight.sum())[None,:,None]).sum(1)


def example():
    theta=torch.full((1,3),1/3)
    first=torch.eye(3)[[0,0,1]]
    other=torch.eye(3)
    coeff,totals=coefficients(torch.stack((first,other)),theta)
    value=interpolate(coeff,totals,1).flatten()
    assert coeff[0,0,0]==coeff[1,0,0]==0 and torch.allclose(coeff[0,-1],coeff[1,-1])
    assert value[0]>2*value[1]
    x=torch.tensor([[[.8,.2],[.2,.8]],[[.5,.5],[.5,.5]]])
    phi=torch.tensor([[.7,.3],[.3,.7]])
    c,t=coefficients(x,phi)
    actual=interpolate(c,t,1)*x.amax(-1).prod(1)[:,None]
    explicit=(.5*((x[:,0]@phi.T)*(x[:,1]@phi.T)+((x[:,0]*x[:,1])@phi.T)))
    assert torch.allclose(actual,explicit,atol=1e-7)
    assert torch.allclose(x[0].mean(0),x[1].mean(0)) and (actual[1]-actual[0]).abs().min()>.02
    return {"equal_extreme_models":coeff[:,[0,-1],0].tolist(),"partial":value.tolist(),
            "equal_average_observations":actual.tolist(),"partitions_of_five":len(partitions(5))}
