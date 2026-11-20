"""Estimate a photo's predictive concept distribution from leave-one-photo-out RGB neighbors."""

from dataclasses import dataclass,replace

import torch
from torch.nn import functional as F

from .gallery_diffusion import nearest_gallery


COUNTS=(4,16,64,256,1024)


@dataclass(frozen=True)
class Config:
    neighbors: int=0  # Zero selects the size by unlabeled predictive log loss.
    estimator: str="predictive"
    geometry: str="rgb"
    alignment: str="paired"


def canonical():
    return Config()


def candidates():
    return [Config(k,mode) for k in (0,16,64,256,1024) for mode in ("predictive","mixture","context","geometric")]


def controls(c):
    return {"primary":c,"no_neighborhood":replace(c,estimator="direct"),
            "context_alone":replace(c,estimator="context"),"own_and_context_mixture":replace(c,estimator="mixture"),
            "product_with_local_prior":replace(c,estimator="predictive"),"geometric_pool":replace(c,estimator="geometric"),
            "label_free_bandwidth":replace(c,neighbors=0),"small_neighborhood":replace(c,neighbors=16),
            "larger_neighborhood":replace(c,neighbors=256),"shuffled_images":replace(c,alignment="shuffle"),
            "centered_rgb":replace(c,geometry="centered")}


def cache_key(c):
    return c.geometry,c.alignment,c.neighbors,c.estimator


class Index:
    def __init__(self,data):
        self.d=data
        self.graphs={}; self.cv={}
        self.previous=None; self.context=None
        self.shuffled=torch.randperm(len(data.p),generator=torch.Generator().manual_seed(1692)).to(data.p.device)
        self.reference=torch.randperm(len(data.p),generator=torch.Generator().manual_seed(7138))[:1024].to(data.p.device)
        self.info={}

    def graph(self,geometry):
        if geometry not in self.graphs:
            vectors=self.d.p if geometry=="rgb" else F.normalize(self.d.p-self.d.p.mean(0))
            self.graphs[geometry]=nearest_gallery(vectors,min(max(COUNTS),len(vectors)-1))[0]
        return self.graphs[geometry]

    def average(self,indices):
        # Tiling the dense multiplication avoids storing a square photo matrix.
        n=len(self.d.p); k=indices.size(1)
        output=self.d.a.new_empty((len(indices),self.d.a.size(1)))
        for start in range(0,len(indices),512):
            sub=indices[start:start+512]
            weights=self.d.a.new_zeros((len(sub),n))
            weights.scatter_(1,sub,1./k)
            torch.mm(weights,self.d.a,out=output[start:start+len(sub)])
        return output

    def calibrated_k(self,geometry):
        if geometry not in self.cv:
            idx=self.graph(geometry)[self.reference]
            truth=self.d.a[self.reference]
            errors={}
            for k in COUNTS:
                prediction=self.average(idx[:,:k])
                loss=-(truth*prediction.clamp_min(1e-38).log()).sum(1).mean()
                errors[k]=float(loss)
            self.cv[geometry]=min(errors,key=errors.get)
            self.info[geometry]={"loo_cross_entropy":errors,"selected_neighbors":self.cv[geometry],"reference_photos":len(idx)}
        return self.cv[geometry]

    def score(self,c):
        if c.estimator=="direct":
            return self.d.base
        k=c.neighbors or self.calibrated_k(c.geometry)
        key=(c.geometry,k,c.alignment)
        if key!=self.previous:
            ids=self.graph(c.geometry)[:,:k]
            if c.alignment=="shuffle":
                ids=self.shuffled[ids]
            self.context=None
            self.context=self.average(ids)
            self.previous=key
        if c.estimator=="context":
            return self.d.query_emission@self.context.T
        if c.estimator=="mixture":
            return .5*(self.d.query_emission@self.context.T)+.5*self.d.base
        # Two conditionally independent pieces of evidence, the current image
        # and its neighbors. The square root is the equal-weight log-pool control.
        scores=self.d.base.new_empty(self.d.base.shape)
        for i in range(0,len(self.context),384):
            b=self.d.a[i:i+384]*self.context[i:i+384]
            if c.estimator=="geometric":
                b=b.sqrt()
            b/=b.sum(1,keepdim=True).clamp_min(1e-38)
            scores[:,i:i+len(b)]=self.d.query_emission@b.T
        return scores
