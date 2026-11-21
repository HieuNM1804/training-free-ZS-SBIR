"""Exactly marginalize a candidate-specific best-of-n concept selection for one query."""

import torch
import torch.nn.functional as F


def compile_index(p,target,scale,prior,k,transform):
    word=prior.topk(k).indices
    logprior=prior[word].clamp_min(1e-30).log()
    response=scale*(p@target[word].T)+logprior
    if "visual" in transform:
        init=response.softmax(1)
        centr=F.normalize(p-p.mean(0))
        mu=F.normalize(init.T@centr)
        response=.5*response+.5*(scale*(centr@mu.T)+logprior)
    b=response.softmax(1)
    if transform.endswith("rgb"):
        from types import SimpleNamespace
        from .cross_diffusion_map import Config,Index
        cfg=Config(neighbors=16,bandwidth=.36,iterations=2,mode="rgb")
        graph=Index(SimpleNamespace(p=p),rgb_only=True)
        b=graph.score(cfg,b.T.contiguous()).T.contiguous()
        b=b/b.sum(1,keepdim=True)
    m=b.mean(0)
    return word,b,m


def exact_best_of(utility,b,prior,order,draws):
    # E[max of n draws] equals sum over adjacent utility gaps times the chance
    # that at least one draw comes from the upper tail. No independent-query
    # batch statistics or stochastic sampling are involved.
    u=utility.gather(1,order)
    mids=u[:,:-1]-u[:,1:]
    # b is supported on public concepts with gallery mass. Its rows sum to one.
    out={}
    total=len(b)
    for n in draws: out[n]=utility.new_empty((len(u),total))
    for j in range(0,total,192):
        block=b[j:j+192]
        mass=block[:,order].permute(1,0,2)
        upper=mass[:,:,:-1].cumsum(2).clamp_(0,1)
        for n in draws:
            if n==1:
                chance=upper
            elif n==2:
                chance=upper*(2-upper)
            else:
                chance=-torch.expm1(n*torch.log1p(-upper))
            out[n][:,j:j+len(block)]=u[:,-1,None]+(mids[:,None,:]*chance).sum(2)
    return out


def scores(q,source,photo,prior,scale,temp,nlist,orderkind):
    logits=scale*temp*(q@source.T)
    u=(logits-logits.amax(1,keepdim=True)).exp()
    if orderkind=="query":
        order=u.argsort(1,descending=True,stable=True)
    elif orderkind=="prior":
        order=prior.argsort(descending=True,stable=True)[None].expand(len(q),-1)
    elif orderkind=="shuffled":
        random=torch.randperm(len(source),generator=torch.Generator().manual_seed(15299)).to(q.device)
        order=u[:,random].argsort(1,descending=True,stable=True)
    else:
        raise ValueError(orderkind)
    return exact_best_of(u,photo,prior,order,nlist)


def validate(q,source,b,prior,scale):
    # Brute-force every pair and every triple on small real category profiles.
    u=(q[:2]@source[:9].T)*scale
    u=(u-u.amax(1,keepdim=True)).exp().double()
    p=b[:5,:9].double()
    p=p/p.sum(1,keepdim=True)
    order=u.argsort(1,descending=True)
    pred=exact_best_of(u,p,prior[:9],order,(1,2,3))
    m2=torch.maximum(u[:,:,None],u[:,None,:])
    m3=torch.maximum(m2[:,:,:,None],u[:,None,None,:])
    tru={1:u@p.T,2:torch.einsum("qij,pi,pj->qp",m2,p,p),
         3:torch.einsum("qijk,pi,pj,pk->qp",m3,p,p,p)}
    err={n:float((pred[n]-tru[n]).abs().max()) for n in pred}
    if max(err.values())>2e-10: raise ValueError("Best-of-n identity failed")
    return err
