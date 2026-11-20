"""Use CLIP's own patch partition to test whether incorrect ranks reflect framing."""

from dataclasses import dataclass, replace

import torch
from torch.nn import functional as F

from .reranking import standardize
from .reverse_prompt import nearest_words, term_average
from .spectral_views import get_views
from .view_consensus_retrieval import Index as NeighborhoodIndex


USE_VIEWS = True
MODES = ("late","padded","middle","uncentered","square")


@dataclass(frozen=True)
class Config:
    mode: str = "late"
    location: str = "object"
    query_weight: float = .65
    photo_weight: float = .35
    terms: int = 8
    neighbors: int = 96
    rounds: int = 2
    support: int = 96
    expansion: float = .65
    pairing: str = "paired"

    @property
    def name(self):
        return (f"{self.mode}-{self.location}-q{self.query_weight:g}-p{self.photo_weight:g}-w{self.terms}"
                f"-k{self.neighbors}x{self.rounds}-n{self.support}-e{self.expansion:g}-{self.pairing}")


def candidates():
    c=Config()
    configs={replace(c,mode=m,query_weight=q,photo_weight=p) for m in MODES
             for q in (.35,.65,1.) for p in (0.,.35,.7)}
    configs.update(replace(c,query_weight=0.,photo_weight=p) for p in (0.,.35,.7))
    for field,values in {"terms":(4,16),"neighbors":(48,192),"rounds":(0,1,3),"support":(48,192),
                         "expansion":(0.,.35,.9)}.items():
        configs.update(replace(c,**{field:v}) for v in values)
    return sorted(configs,key=lambda v:v.name)


def refine(c):
    return list(dict.fromkeys([c,*[replace(c,query_weight=q,photo_weight=p)
        for q in (.15,.4,.75,.9,1.) for p in (0.,.20,.5,.85)],
        *[replace(c,support=n,expansion=e) for n in (48,192,320) for e in (.45,.75,.9)],
        *[replace(c,neighbors=k,rounds=r) for k in (48,144,256) for r in (1,2)],
        *[replace(c,terms=v) for v in (4,12,24)]]))


def controls(c):
    return {"primary":c,"whole_sketch":replace(c,query_weight=0.),"whole_gallery":replace(c,photo_weight=0.),
        "both_full_frames":replace(c,query_weight=0.,photo_weight=0.),"same_size_center":replace(c,location="center"),
        "same_size_opposite":replace(c,location="opposite"),"no_averaging":replace(c,rounds=0),
        "no_feedback":replace(c,expansion=0.),"shuffled_words":replace(c,pairing="shuffled"),
        "native":replace(c,pairing="rgb")}


def vocab_size(c):
    return 0 if c.pairing=="rgb" else 20931


def choose(images,c,weight):
    if not weight:
        return images[:,0]
    slot=1+MODES.index(c.mode)+{"object":0,"center":6,"opposite":11}[c.location]
    return F.normalize(images[:,0].lerp(images[:,slot],weight))


class Index(NeighborhoodIndex):
    def base(self,view):
        if view not in self.bases:
            mode,location,weight=view
            rgb=choose(self.photos,Config(mode=mode,location=location),weight)
            ids,values=nearest_words(rgb,self.target,32)
            self.bases[view]=(rgb,F.normalize(rgb-rgb.mean(0)),ids,values)
        return self.bases[view]

    def banks(self,c):
        view=(c.mode,c.location,c.photo_weight) if c.photo_weight else ("late","object",0.)
        key=(view,c.terms,c.neighbors,c.rounds,c.pairing)
        if key not in self.augmented:
            rgb,geom,ids,values=self.base(view)
            mapping=self.shuffle if c.pairing=="shuffled" else self.identity
            words=rgb if c.pairing=="rgb" else term_average(self.source,mapping,ids,values,c.terms,.03)
            initial=torch.cat((words,geom),1)
            words,visual=words,geom
            for _ in range(c.rounds):
                updates=.30*initial+.70*torch.sparse.mm(self.operator(view,c.neighbors),torch.cat((words,visual),1))
                words,visual=(F.normalize(m) for m in updates.chunk(2,dim=1))
            self.augmented[key]=(words,visual)
        return self.augmented[key]

    def scores(self,queries,c,block_size=144):
        words,visual=self.banks(c)
        for views in queries.split(block_size):
            query=choose(views,c,c.query_weight)
            base=standardize(query@words.T)
            if c.expansion:
                vals,ids=base.topk(c.support)
                feedback=F.normalize(torch.bmm((vals/.6).softmax(1)[:,None],visual[ids]).squeeze(1))
                base=base.lerp(standardize(feedback@visual.T),c.expansion)
            yield base
