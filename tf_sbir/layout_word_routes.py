"""Compare evidence routes before and after a within-image patch-layout intervention."""

from math import log
from types import SimpleNamespace

import torch

from .cross_diffusion_map import Config,Index
from .shared_concept_likelihood import fit_prior


def layout_evidence(original,permuted,text,scale,subset=None):
    banks=permuted if subset is None else permuted[:,subset]
    logits=(original@text.T)*scale
    draws=(banks.float().reshape(-1,text.size(1))@text.T).reshape(len(original),banks.size(1),-1)*scale
    delta=logits-(draws.logsumexp(1)-log(banks.size(1)))
    return logits,delta


class RouteIndex:
    def __init__(self,p,nulls,target,scale,prior_mode):
        self.scale=scale
        self.target=target
        n=len(p)
        self.logits=p.new_empty((n,len(target)))
        self.delta=torch.empty_like(self.logits)
        for start in range(0,n,128):
            raw,contrast=layout_evidence(p[start:start+128],nulls[start:start+128],target,scale)
            self.logits[start:start+len(raw)]=raw
            self.delta[start:start+len(raw)]=contrast
        probs=self.logits.softmax(1)
        self.prior=fit_prior(probs)[0]["mle"] if prior_mode=="mle" else torch.ones_like(probs[0])
        self.logits+=self.prior.clamp_min(1e-30).log()[None]
        self.p=self.logits.softmax(1)
        self.config=Config(neighbors=16,bandwidth=.36,iterations=2,mode="rgb")
        self.graph=Index(SimpleNamespace(p=p),rgb_only=True)
        self.graph.prepare(self.config)

    def make(self,kind,value,temperature,seed=17341):
        self.kind=kind
        self.value=value
        self.route_temp=temperature
        p=self.p
        self.route_table=None
        self.mean=None
        self.photo=p
        if kind.startswith("route") or kind=="flat_query":
            order=None
            if kind=="route_shuffle":
                gen=torch.Generator().manual_seed(seed)
                order=torch.randperm(len(p),generator=gen).to(p.device)
            total=p.new_zeros(p.size(1),dtype=torch.double)
            mass=torch.zeros_like(total)
            self.route_table=torch.empty_like(p)
            for start in range(0,len(p),256):
                sl=slice(start,start+256)
                d=self.delta[sl] if order is None else self.delta[order[sl]]
                a=(d*temperature).sigmoid()
                self.route_table[sl]=a
                total+=p[sl].sum(0,dtype=torch.double)
                mass+=(p[sl]*a).sum(0,dtype=torch.double)
            self.mean=(mass/total.clamp_min(1e-30)).float().clamp(2e-6,1-2e-6)
            residual=torch.zeros_like(total)
            for start in range(0,len(p),256):
                sl=slice(start,start+256)
                block=self.route_table[sl]
                block.sub_(self.mean).mul_(p[sl])
                residual+=block.sum(0,dtype=torch.double)
            self.mass_check=float((residual.abs()/total.clamp_min(1e-30)).max())
        if kind in ("contrast_both","contrast_target"):
            self.photo=torch.empty_like(p)
            for start in range(0,len(p),256):
                sl=slice(start,start+256)
                self.photo[sl]=(self.logits[sl]+value*self.delta[sl]).softmax(1)

    def score(self,logits,delta):
        if self.kind in ("contrast_both","contrast_query"):
            logits=logits+self.value*delta
        l=(logits-logits.amax(1,keepdim=True)).exp()
        e=l@self.photo.T
        if self.route_table is not None:
            route=(delta*self.route_temp).sigmoid()
            if self.kind=="route_opposite":route=1-route
            elif self.kind=="flat_query":route=torch.full_like(route,.5)
            weight=l*(route-self.mean)/(self.mean*(1-self.mean))
            e=e+self.value*(weight@self.route_table.T)
        if not torch.isfinite(e).all():raise ValueError("Invalid route score")
        return self.graph.score(self.config,e)

    def check_routes(self,l,delta):
        if self.route_table is None:return {}
        a=(delta*self.route_temp).sigmoid()
        ai=(self.delta[:17]*self.route_temp).sigmoid()
        if self.kind=="route_shuffle": return {"weighted_word_mass_error":self.mass_check}
        if self.kind=="route_opposite":a=1-a
        elif self.kind=="flat_query":a=torch.full_like(a,.5)
        logits=l
        l=(logits-logits.amax(1,keepdim=True)).exp()
        p=self.p[:17]
        f=self.mean
        raw=l@p.T
        explicit=(l*a/f)@(p*ai).T+(l*(1-a)/(1-f))@(p*(1-ai)).T
        actual=raw+(l*(a-f)/(f*(1-f)))@self.route_table[:17].T
        rel=float(((explicit-actual).abs()/raw.amax(1,keepdim=True).clamp_min(1e-30)).max())
        if rel>2e-4:raise ValueError("Two-route likelihood ratio mismatch")
        return {"weighted_word_mass_error":self.mass_check,"two_channel_formula_error":rel}
