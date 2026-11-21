"""Use unlabeled photo neighborhoods as context for gallery concept likelihoods."""

from dataclasses import dataclass, replace

import torch

from .gallery_diffusion import nearest_gallery
from .shared_concept_likelihood import fit_prior, posterior


@dataclass(frozen=True)
class Config:
    k: int = 64
    calibration: str = "posterior"
    estimator: str = "mle"
    weighting: str = "uniform"
    shuffled: bool = False


def canonical():
    return Config()


def candidates():
    return [Config(k,order,prior,weight) for k in (16,64,128,256)
            for order,prior in (("posterior","mle"),("posterior","uniform"),("likelihood","mle"),
                                ("likelihood","one-update"),("posterior","one-update"))
            for weight in ("uniform","cosine")]


def controls(c):
    return {"primary":c,"no_neighborhood":replace(c,k=0),"uniform_prior":replace(c,estimator="uniform"),
            "before_neighborhood":replace(c,calibration="posterior"),"after_neighborhood":replace(c,calibration="likelihood"),
            "one_update":replace(c,estimator="one-update"),"mle":replace(c,estimator="mle"),
            "cosine_weighting":replace(c,weighting="cosine"),"uniform_weights":replace(c,weighting="uniform"),
            "random_neighborhood":replace(c,shuffled=True)}


def cache_key(c):
    return c.k,c.weighting,c.shuffled,c.calibration,c.estimator


class Index:
    def __init__(self,data):
        self.d=data
        self.priors,self.info=fit_prior(data.a)
        self.neighbors,self.similarity=nearest_gallery(data.p,256)
        self.permutation=torch.randperm(len(data.p),generator=torch.Generator().manual_seed(373)).to(data.p.device)
        self.context=None
        self.previous=None

    def pool(self,c,features):
        n=len(features)
        ids=self.neighbors[:,:c.k]
        if c.shuffled:
            ids=self.permutation[ids]
        if c.weighting=="cosine":
            distances=2*(1-self.similarity[:,:c.k]).clamp_min(0)
            bandwidth=.36*distances.sqrt().mean().square().clamp_min(1e-12)
            weights=(-distances/bandwidth).softmax(1)
        else:
            weights=features.new_full(ids.shape,1./c.k)
        crow=torch.arange(0,(n+1)*c.k,c.k,device=ids.device,dtype=torch.int32)
        columns=ids.flatten().to(torch.int32)
        graph=torch.sparse_csr_tensor(crow,columns,weights.flatten(),size=(n,n),check_invariants=False)
        return torch.sparse.mm(graph,features)

    def score(self,c):
        prior=self.priors[c.estimator]
        if c.k==0:
            predictive=posterior(self.d.a,prior)
        elif c.calibration=="posterior":
            self.context=None
            self.previous=None
            predictive=self.pool(c,posterior(self.d.a,prior))
        else:
            key=(c.k,c.weighting,c.shuffled)
            if key!=self.previous:
                self.context=None
                self.context=self.pool(c,self.d.a)
                self.previous=key
            predictive=posterior(self.context,prior)
        return self.d.query_emission@predictive.T
