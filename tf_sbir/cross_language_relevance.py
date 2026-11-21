"""Estimate a relevance language model for one sketch from unlabeled photo documents."""

from dataclasses import dataclass, replace

import torch

from .shared_concept_likelihood import fit_prior, posterior


@dataclass(frozen=True)
class Config:
    proxies: int = 96
    background: float = .1
    prior: str = "uniform"
    first_pass: str = "concept"
    scoring: str = "cross_entropy"
    shuffle: bool = False


def canonical():
    return Config()


def candidates():
    return [Config(k,background,prior) for k in (20,96,256,0)
            for background in (0.,.1,.5) for prior in ("uniform","mle")]


def controls(c):
    return {"primary":c,"initial_query_likelihood":replace(c,scoring="initial"),
            "no_feedback":replace(c,scoring="query_only"),
            "linear_word_score":replace(c,scoring="linear"),
            "no_background":replace(c,background=0.),"uniform_prior":replace(c,prior="uniform"),
            "gallery_prior":replace(c,prior="mle"),"paper_canonical":canonical(),
            "rgb_first_pass":replace(c,first_pass="rgb"),"shuffled_proxy_identity":replace(c,shuffle=True)}


def cache_key(c):
    return c.prior,c.first_pass,c.shuffle,c.proxies,c.background,c.scoring


class Index:
    def __init__(self,data):
        self.data=data
        self.priors,self.info=fit_prior(data.a)
        self.permutation=torch.randperm(len(data.p),generator=torch.Generator().manual_seed(5609)).to(data.p.device)
        self.last_prior=None
        self.last_feedback=None
        self.language=None
        self.initial=None
        self.estimate=None

    def prepare(self,prior):
        if prior==self.last_prior:
            return
        self.language=None
        self.initial=None
        self.estimate=None
        self.language=self.data.a if prior=="uniform" else posterior(self.data.a,self.priors[prior])
        self.collection=self.language.mean(0)
        self.initial=self.data.base if prior=="uniform" else self.data.query_emission@self.language.T
        self.last_prior=prior
        self.last_feedback=None

    def relevance(self,c):
        key=(c.proxies,c.first_pass,c.shuffle)
        if key==self.last_feedback:
            return self.estimate
        self.estimate=None
        self.estimate=self.data.query_emission.new_empty(self.data.query_emission.shape)
        for start in range(0,len(self.estimate),192):
            block=self.initial[start:start+192]
            if c.first_pass=="rgb":
                visual=self.data.q[start:start+len(block)]@self.data.p.T
                logits=self.data.scale*(visual-visual.amax(1,keepdim=True))
                block=logits.exp()
            if c.proxies:
                values,ids=block.topk(c.proxies,dim=1)
                if c.shuffle:
                    ids=self.permutation[ids]
                weights=block.new_zeros(block.shape).scatter_(1,ids,values)
            else:
                weights=block[:,self.permutation] if c.shuffle else block
            normalized=weights/weights.sum(1,keepdim=True).clamp_min(1e-38)
            self.estimate[start:start+len(block)]=normalized@self.language
        self.last_feedback=key
        return self.estimate

    def score(self,c):
        self.prepare(c.prior)
        if c.scoring=="initial":
            return self.initial
        if c.scoring=="query_only":
            evidence=self.data.query_emission
            relevance=evidence/evidence.sum(1,keepdim=True).clamp_min(1e-38)
        else:
            relevance=self.relevance(c)
        result=self.initial.new_empty(self.initial.shape)
        for begin in range(0,len(self.language),512):
            doc=self.language[begin:begin+512].lerp(self.collection,c.background)
            if c.scoring!="linear":
                doc.clamp_min_(1e-38).log_()
            result[:,begin:begin+len(doc)]=relevance@doc.T
        return result
