"""Transfer WeiMoCIR's weighted query composition and weighted target similarities."""

from dataclasses import dataclass, replace

import open_clip
import torch
from torch.nn import functional as F

from .encoding import load_frozen_clip
from .image_retrieval_round import DECISION

FULL_FINALISTS=2


@dataclass(frozen=True)
class Config:
    alpha: float = .1
    beta: float = .8
    modifier: str = "domain"
    descriptions: str = "three"
    mode: str = "matched"


def canonical():
    return Config()


def candidates():
    variants=[Config(a,b,"content",cap) for a in (.1,.5,.8,1.) for b in (.2,.5,.8,1.)
              for cap in ("three","all")]
    variants.extend(Config(a,b) for a in (.1,.5,.8) for b in (.2,.5,.8,1.))
    return variants


def controls(c):
    return {"primary":c,"paper_coefficients_domain_modifier":canonical(),"fixed_domain":replace(c,modifier="domain"),
            "no_query_text":replace(c,alpha=0.),"only_query_text":replace(c,alpha=1.),
            "only_photo_similarity":replace(c,beta=0.),"only_description_similarity":replace(c,beta=1.),
            "alternative_descriptions":replace(c,descriptions="all" if c.descriptions=="three" else "three"),
            "shuffled_photo_descriptions":replace(c,mode="shuffle"),"exact_word_likelihood":replace(c,mode="seed")}


def cache_key(c):
    return c.descriptions,c.modifier,c.mode,c.alpha,c.beta


def prepare_ir_data(d,args,qpaths,ppaths,qi,pi):
    model=load_frozen_clip(args.clip_checkpoint,d.q.device)
    text=open_clip.get_tokenizer("ViT-B-32")(["a photo."]).to(d.q.device)
    d.domain_text=F.normalize(model.encode_text(text).float(),dim=1)


class Index:
    def __init__(self,d):
        self.d=d
        self.query_text=F.normalize(d.query_emission@d.target,dim=1)
        self.photo_text={"all":d.a@d.target,"three":d.target[d.a.topk(3,dim=1).indices].mean(1)}
        self.permutation=torch.randperm(len(d.p),generator=torch.Generator().manual_seed(5925)).to(d.p.device)
        self.info={"target_domain_modifier":"a photo.","caption_model":None,
                   "descriptions":"normalized CLIP text directions for public photo concepts",
                   "mean_norm_of_full_description":float(self.photo_text["all"].norm(dim=1).mean())}

    def score(self,c):
        if c.mode=="seed":
            return self.d.base
        text=self.d.domain_text if c.modifier=="domain" else self.query_text
        q=F.normalize(self.d.q.lerp(text,c.alpha),dim=1)
        descriptions=self.photo_text[c.descriptions]
        if c.mode=="shuffle":
            descriptions=descriptions[self.permutation]
        # Each visual or textual vector entering either cosine is unit length.
        # The mean of caption directions is intentionally not normalized again.
        return q@self.d.p.lerp(descriptions,c.beta).T
