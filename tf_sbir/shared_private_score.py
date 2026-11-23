"""Predict query evidence from a held-out photo's own and group-shared latent names."""
from dataclasses import dataclass
import itertools
import math

import torch

from .shared_private_names import MODELS,DEPTHS,segment_shuffle


ALPHAS=(.2,.6,.9)
POWERS=(.35,1.)
FULL=("joint","mass","fixed","private","peer","wrong","opposite","reg-cov","reg-wrong","reg-mass","reg-joint")
GAINS=(-.6,.3,1.)
BASES=("base/direct","base/regional")


@dataclass(frozen=True)
class Row:
    model:str
    depth:int
    power:float
    alpha:float
    mode:str
    gain:float

    def label(self):
        return f"{self.model}/{self.depth}/{self.power:g}/{self.alpha:g}/{self.mode}/{self.gain:g}"

    @classmethod
    def parse(cls,k):
        m,d,e,a,x,g=k.split("/")
        return cls(m,int(d),float(e),float(a),x,float(g))


def grid():
    rows=set(BASES)
    for model,level,power,alpha,mode in itertools.product(MODELS,DEPTHS,POWERS,ALPHAS,FULL):
        if mode in ("private","peer") and (power!=1. or alpha!=.6):continue
        for gain in (GAINS if mode.startswith("reg-") else (1.,)):
            rows.add(Row(model,level,power,alpha,mode,gain).label())
    return sorted(rows)


class ConditionalIndex:
    def __init__(self,bank,geometry,peer,depth,power):
        self.bank=bank
        self.fold=geometry["folds"]
        self.group=geometry["groups"][depth]
        self.peer=peer
        self.ratio=bank.new_zeros(len(bank))
        self.shuffle=torch.arange(len(bank),device=bank.device)
        self.joint=torch.empty_like(bank)
        self.audit=0.;self.frozen_chance_error=0.
        for h,idx in enumerate(self.fold):
            which=self.group[idx];p=peer[h];m=geometry["priors"][h]
            local=segment_shuffle(which,idx,91961+h)
            self.shuffle[idx]=idx[local]
            for off in range(0,len(idx),256):
                ii=idx[off:off+256]
                latent=p[self.group[ii]];posterior=bank[ii]
                lik=(posterior/m[None]).clamp_min(1e-30)
                if power!=1.:lik=lik.pow(power)
                unnorm=latent*lik
                z=unnorm.sum(1).clamp_min(1e-30)
                cond=unnorm/z[:,None]
                self.ratio[ii]=z;self.joint[ii]=cond
                self.audit=max(self.audit,float((cond.sum(1)-1).abs().amax()))
            # Probability conservation under an uninformative candidate and priors identical to background.
            test=peer[h][:8]
            uniform=torch.ones_like(test)
            self.frozen_chance_error=max(self.frozen_chance_error,float((test*uniform/test.sum(1,keepdim=True)-test).abs().amax()))
        if self.audit>5e-5 or self.frozen_chance_error>5e-5:raise ValueError("Conditional name posterior is not normalized")

    def values(self,likelihood):
        # Real query over all 20,931 words. No truncation or top-word dispatch.
        b=len(likelihood);out=likelihood.new_empty((b,len(self.bank)))
        for h,idx in enumerate(self.fold):
            grouped=likelihood@self.peer[h].T
            out[:,idx]=grouped[:,self.group[idx]]
        joint=likelihood@self.joint.T
        return out,joint


def read_scores(direct,regional,peer,joint,index,recipes):
    groups={}
    for r in recipes:groups.setdefault(r.alpha,[]).append(r)
    positive=lambda x:x.clamp_min(1e-32).log()
    for alpha,items in groups.items():
        z=index.ratio
        h=alpha*z/((1-alpha)+alpha*z)
        # a=x's own lexical likelihood, e=peer expectation, j=candidate-verified expectation.
        # The ordinary and scalar counterfactual share the exact same posterior borrowing mass h.
        scalar=direct+h*(peer-direct)
        joined=direct+h*(joint-direct)
        wrong=direct+h*(joint[:,index.shuffle]-direct)
        correction=positive(joined)-positive(scalar)
        for r in items:
            if r.mode=="joint":val=positive(joined)
            elif r.mode=="mass":val=positive(scalar)
            elif r.mode=="fixed":val=positive(direct+alpha*(joint-direct))
            elif r.mode=="private":val=positive(direct)
            elif r.mode=="peer":val=positive(peer)
            elif r.mode=="wrong":val=positive(wrong)
            elif r.mode=="opposite":val=positive(scalar)-correction
            elif r.mode=="reg-cov":val=regional+r.gain*correction
            elif r.mode=="reg-wrong":val=regional+r.gain*(positive(wrong)-positive(scalar))
            elif r.mode=="reg-mass":val=regional+r.gain*(positive(scalar)-positive(direct))
            elif r.mode=="reg-joint":val=regional+r.gain*(positive(joined)-positive(direct))
            else:raise ValueError(r.mode)
            yield r.label(),val
