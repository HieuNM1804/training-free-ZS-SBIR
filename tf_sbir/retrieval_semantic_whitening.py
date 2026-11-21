"""Center and whiten image descriptions in a fixed public semantic reference space."""

from dataclasses import dataclass, replace
import torch
from torch.nn import functional as F

from .image_retrieval_round import DECISION

USES_LITERAL = True
FULL_FINALISTS = 2


@dataclass(frozen=True)
class Config:
    dimensions: int = 256
    power: float = 1.
    reference: str = "words"
    mode: str = "semantic"


def canonical():
    return Config()


def candidates():
    return [Config(d,p) for d in (32,64,128,256,512) for p in (0.,.5,1.)]


def controls(c):
    return {"primary":c,"paper_whitening":canonical(),"no_whitening":replace(c,power=0.),
            "unprocessed_mean":replace(c,mode="plain"),"gallery_reference":replace(c,reference="gallery"),
            "visual_only_gallery_pca":replace(c,mode="visual",reference="gallery"),
            "permuted_word_identity":replace(c,mode="shuffle"),"exact_word_likelihood":replace(c,mode="seed")}


def cache_key(c):
    return c.mode,c.reference,c.dimensions,c.power


def basis(rows):
    centered=rows.double()
    mean=centered.mean(0)
    centered=centered-mean
    values,vectors=torch.linalg.eigh(centered.T@centered/(len(rows)-1))
    return mean.float(),values.flip(0).float(),vectors.flip(1).float()


class Index:
    def __init__(self,data):
        self.d=data
        self.query=F.normalize(data.query_emission@data.literal,dim=1)
        self.docs=F.normalize(data.a@data.literal,dim=1)
        self.public=basis(data.literal)
        self.reference={}
        eig=self.public[1]
        self.info={"public_words":len(data.literal),"variance_fraction_top_32":float(eig[:32].sum()/eig.sum()),
                   "primary_reference":"unit raw public word embeddings only"}

    def score(self,c):
        d=self.d
        if c.mode=="seed":
            return d.base
        if c.mode=="plain":
            return self.query@self.docs.T
        q,p=self.query,self.docs
        if c.mode=="visual":
            q,p=d.q,d.p
        if c.mode=="shuffle":
            perm=torch.randperm(len(d.literal),generator=torch.Generator().manual_seed(9587)).to(d.q.device)
            q=F.normalize(d.query_emission@d.literal[perm],dim=1)
        if c.reference=="words":
            mean,eig,U=self.public
        else:
            key=c.mode=="visual"
            if key not in self.reference:
                self.reference[key]=basis(p)
            mean,eig,U=self.reference[key]
        floor=torch.finfo(eig.dtype).eps*eig[0]
        axes=U[:,:c.dimensions]*eig[:c.dimensions].clamp_min(floor).pow(-.5*c.power)
        return F.normalize((q-mean)@axes,dim=1)@F.normalize((p-mean)@axes,dim=1).T
