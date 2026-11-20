"""A query-likelihood model that permits a gallery concept to generate a different query term."""

from dataclasses import dataclass,replace

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    channel: str="gallery"
    steps: int=1
    translated_mass: float=.5
    keep_identity: bool=False
    shuffled: bool=False


def canonical():
    return Config()


def candidates():
    return [Config(channel,steps,mass,keep) for channel in ("gallery","text")
            for steps in (1,2) for mass in (.25,.5,1.) for keep in ((False,True) if channel=="gallery" else (False,))]


def controls(c):
    return {"primary":c,"exact_terms":replace(c,translated_mass=0.),"translation_only":replace(c,translated_mass=1.),
            "include_identical_word":replace(c,keep_identity=True),"exclude_identical_word":replace(c,keep_identity=False),
            "one_translation":replace(c,steps=1),"two_translations":replace(c,steps=2),
            "semantic_text_kernel":replace(c,channel="text"),"gallery_cooccurrence":replace(c,channel="gallery"),
            "shuffle_translation_destination":replace(c,shuffled=True),"canonical":canonical()}


def cache_key(c):
    return c.channel,c.keep_identity,c.shuffled,c.steps,c.translated_mass


class Index:
    def __init__(self,data):
        self.d=data
        self.mass=data.a.sum(0).clamp_min(1e-38)
        self.diag=torch.zeros_like(self.mass)
        for chunk in data.a.split(384):
            self.diag+=chunk.square().sum(0)
        self.diag/=self.mass
        self.language=F.normalize(data.source+data.target)
        self.perm=torch.randperm(len(self.language),generator=torch.Generator().manual_seed(374)).to(self.language.device)
        self.last_key=None; self.intermediate=None; self.steps=0
        self.info={"mean_identity_probability_gallery":float(self.diag.mean()),
                   "posterior_weighted_identity_probability":float((self.mass*self.diag).sum()/self.mass.sum())}

    def move(self,f,c):
        # Each row is a conditional distribution of query terms for a gallery term.
        if c.channel=="gallery":
            propagated=((f@self.d.a.T)@self.d.a)/self.mass
            if not c.keep_identity:
                remaining=1-self.diag
                numerator=(propagated-f*self.diag).clamp_min(0.)
                propagated=torch.where(remaining>1e-6,numerator/remaining.clamp_min(1e-6),f)
            return propagated
        out=torch.empty_like(f)
        for i in range(0,len(self.language),384):
            bank=self.language[i:i+384]
            logits=self.d.scale*(bank@self.language.T)
            if not c.keep_identity:
                it=torch.arange(len(bank),device=f.device)
                logits[it,i+it]=-torch.inf
            out[:,i:i+len(bank)]=f@logits.softmax(1).T
        return out

    def score(self,c):
        if c.translated_mass==0:
            return self.d.base
        key=(c.channel,c.keep_identity,c.shuffled)
        if key!=self.last_key or c.steps<self.steps:
            self.steps=0; self.intermediate=self.d.query_emission; self.last_key=key
        while self.steps<c.steps:
            nxt=self.move(self.intermediate,c)
            self.intermediate=nxt[:,self.perm] if c.shuffled else nxt
            self.steps+=1
        translated=self.intermediate@self.d.a.T
        return translated if c.translated_mass==1 else self.d.base.lerp(translated,c.translated_mass)
