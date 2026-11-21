"""Multiple Gaussian assignment to public words, with separate bandwidths for both image domains."""

import argparse
from dataclasses import dataclass, replace
from datetime import datetime,timezone
from pathlib import Path
import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json,checksum,read
from .crossdomain_benchmark import digest
from .data import SKETCHY_SPLITS
from .image_retrieval_round import DECISION as FIVE_DECISION, verified
from .map_study import score_key

FAMILY="retrieval-soft-assignment"
DECISION=FAMILY+"/extra-joint.json"
FULL_FINALISTS=3


@dataclass(frozen=True)
class Config:
    query_scale: float = 1.
    photo_scale: float = 1.
    terms: int = 3
    norm: str = "cosine"
    control: str = "uniform"


def canonical():
    return Config()


def candidates():
    scales=(.25,.5,1.,2.,4.)
    dense=[Config(a,b,0,n) for a in scales for b in scales for n in ("mass","cosine")]
    sparse=[Config(a,b,r,n) for a,b in ((.5,.5),(1.,1.),(2.,2.),(.5,2.),(2.,.5))
            for r in (3,16,64) for n in ("mass","cosine")]
    return dense+sparse


def controls(c):
    return {"primary":c,"paper_sparse":canonical(),"hard_assignment":replace(c,terms=1),
            "checkpoint_temperatures":replace(c,query_scale=1.,photo_scale=1.),
            "same_as_query":replace(c,photo_scale=c.query_scale),"same_as_photo":replace(c,query_scale=c.photo_scale),
            "no_truncation":replace(c,terms=0),"alternative_normalization":replace(c,norm="cosine" if c.norm=="mass" else "mass"),
            "original_gallery_idf_three":replace(canonical(),control="idf"),
            "shuffled_identity":replace(c,control="shuffle"),"exact_seed":Config(1.,1.,0,"mass","seed")}


def cache_key(c):
    return c.terms,c.norm,c.control,c.photo_scale,c.query_scale


def assign(raw,scale,terms,norm,idf=None):
    if terms:
        value,idx=raw.topk(terms,dim=1)
        value=(value/value.amax(1,keepdim=True).clamp_min(1e-30)).pow(scale)
        if idf is not None:
            value=value*idf[idx]
        value=F.normalize(value,p=norm,dim=1)
        return torch.zeros_like(raw).scatter_(1,idx,value)
    value=raw/raw.amax(1,keepdim=True).clamp_min(1e-30)
    value.pow_(scale)
    if idf is not None:
        value.mul_(idf)
    return F.normalize(value,p=norm,dim=1)


class Index:
    def __init__(self,data):
        self.d=data
        cols=data.a.topk(3,dim=1).indices.flatten()
        counts=cols.bincount(minlength=data.a.size(1))
        self.idf=(len(data.a)/counts.float().clamp_min(1)).log()
        self.perm=torch.randperm(data.a.size(1),generator=torch.Generator().manual_seed(2093)).to(data.q.device)
        self.last=None
        self.photo=None
        self.info={"vocabulary":data.a.size(1),"controls_gallery_idf_uses_top_three":True,
                   "primary_reference":"public words with constant IDF"}

    def score(self,c):
        if c.control=="seed":
            return self.d.base
        idf=self.idf if c.control=="idf" else None
        key=(c.terms,c.norm,c.photo_scale,c.control=="idf")
        norm=1 if c.norm=="mass" else 2
        if self.last!=key:
            self.photo=None
            self.photo=assign(self.d.a,c.photo_scale,c.terms,norm,idf)
            self.last=key
        q=assign(self.d.query_emission,c.query_scale,c.terms,norm,idf)
        if c.control=="shuffle":
            q=q[:,self.perm]
        return q@self.photo.T


def supplement(w):
    root=w/"results"
    output=root/DECISION
    if output.exists():
        return verified(output)
    previous=verified(root/FIVE_DECISION)
    folder=root/FAMILY
    if any((folder/"heldout").glob("tuberlin-*.json")):
        raise ValueError("TU preceded the assignment decision")
    lock=verified(folder/"freeze.json")
    entries=previous["families"].copy()
    scores=[]
    for s in SKETCHY_SPLITS:
        f=folder/"sketchy"/f"{s}-{lock['winner_id']}.json"
        if checksum(f)!=lock["sources"][f.name]:
            raise ValueError(f"Sketchy score changed: {f}")
        row=read(f)
        scores.append({"split":s,"sha256":checksum(f),"mAP":row["mAP"],"precision":row["precision"]})
    entries[FAMILY]={"scope":"pairwise","fresh":True,"name":lock["winner_id"],"freeze":digest(lock),
                     "settings":lock["winner"],"revision":lock["revision"],"scores":scores}
    def winner(scope):
        return max((n for n in entries if scope=="all" or entries[n]["scope"]==scope),
                   key=lambda n:score_key(entries[n]["scores"]))
    record={"utc":datetime.now(timezone.utc).isoformat(),"families":entries,
            "prior_five_lock":digest(previous),"primary":winner("all"),"pairwise":winner("pairwise"),
            "gallery":winner("gallery"),"criterion":previous["criterion"],"selection_data":list(SKETCHY_SPLITS),
            "new_assignment_tu_rows":0,"basis":"Sketchy scores only, retained five-way comparison"}
    atomic_json(output,record|{"sha256":digest(record)})
    return record


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    record=supplement(p.parse_args().workspace)
    print(record["pairwise"],record["primary"])
