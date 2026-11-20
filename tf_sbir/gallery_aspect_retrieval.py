"""Rank photos by marginalizing latent visual groups named with a fixed CLIP vocabulary."""

from dataclasses import dataclass, replace

import torch
from torch.nn import functional as F

from .cluster_mine import cluster_assignments
from .reranking import standardize
from .reverse_prompt import nearest_words, term_average


USE_VIEWS = False


@dataclass(frozen=True)
class Config:
    groups: int = 128
    naming: str = "votes"
    words: int = 8
    distinct: float = .35
    geometry: float = .35
    query_temp: float = .025
    image_temp: float = .06
    prior: float = .5
    mix: float = .8
    pairing: str = "paired"

    @property
    def name(self):
        return (f"g{self.groups}-{self.naming}-w{self.words}-d{self.distinct:g}-v{self.geometry:g}"
                f"-q{self.query_temp:g}-i{self.image_temp:g}-p{self.prior:g}-m{self.mix:g}-{self.pairing}")


def candidates():
    base = Config()
    grid={replace(base,groups=g,naming=n,geometry=a) for g in (32,64,128,256,512)
          for n in ("votes","centroid","pooled") for a in (0.,.5)}
    for attr, values in {"words":(1,4,16,32),"distinct":(0.,.7,1.),"geometry":(0.,.75,1.),
                         "query_temp":(.008,.05,.1),"image_temp":(.025,.12),"prior":(0.,1.),
                         "mix":(0.,.35,.6,1.)}.items():
        grid.update(replace(base,**{attr:v}) for v in values)
    return sorted(grid,key=lambda c:c.name)


def refine(c):
    return list(dict.fromkeys([c,*[replace(c,groups=g,geometry=v) for g in (48,96,192,384,768)
        for v in (0.,.4,.75)], *[replace(c,query_temp=q,image_temp=t) for q in (.008,.02,.04)
        for t in (.025,.06,.10)], *[replace(c,mix=m) for m in (.45,.65,.95,1.)],
        *[replace(c,distinct=x) for x in (0.,.6,1.)],*[replace(c,naming=n) for n in ("votes","pooled","centroid")]]))


def controls(c):
    return {"primary":c,"no_groups":replace(c,mix=0.),"groups_only":replace(c,mix=1.),
        "pure_rgb_geometry":replace(c,geometry=0.),"uniform_word_prior":replace(c,distinct=0.),
        "uniform_group_prior":replace(c,prior=0.),"random_group_links":replace(c,pairing="random_groups"),
        "neutral_words":replace(c,pairing="neutral"),"rgb_only":replace(c,pairing="rgb",geometry=0.)}


def vocab_size(c):
    return 0 if c.pairing=="rgb" else 20931


def means(values, assignments, count):
    total = values.new_zeros(count,values.size(1)).index_add_(0,assignments,values)
    return F.normalize(total)


class Index:
    def __init__(self,photos,source,target):
        self.rgb,self.source,self.target=photos,source,target
        self.visual=F.normalize(photos-photos.mean(0))
        ids,values=nearest_words(photos,target,32)
        self.ids,self.values=ids,values
        self.identity=torch.arange(len(target),device=photos.device)
        self.raw_codes={key:term_average(bank,self.identity,ids,values,8,.03)
                        for key,bank in (("paired",source),("neutral",target))}
        self.semantic=F.normalize(self.raw_codes["paired"]-self.raw_codes["paired"].mean(0))
        self.partitions,self.codes,self.assignments={},{},{}

    def partition(self,c):
        key=(c.groups,c.geometry)
        if key not in self.partitions:
            features=F.normalize(torch.cat(((1-c.geometry)**.5*self.visual,c.geometry**.5*self.semantic),1))
            assigned=cluster_assignments(features,c.groups)
            centers=means(features,assigned,c.groups)
            proximity=features @ centers.T
            self.partitions[key]=(assigned,proximity)
        return self.partitions[key]

    def names(self,c,assigned):
        key=(c.groups,c.geometry,c.naming,c.words,c.distinct,c.pairing)
        if key in self.codes:
            return self.codes[key]
        bank=self.target if c.pairing=="neutral" else self.source
        raw=self.raw_codes["neutral" if c.pairing=="neutral" else "paired"]
        pooled=means(raw,assigned,c.groups)
        if c.pairing=="rgb":
            names=means(self.rgb,assigned,c.groups)
        elif c.naming=="pooled":
            names=pooled
        else:
            if c.naming=="votes":
                width=16
                pair=assigned[:,None]*len(bank)+self.ids[:,:width]
                weights=(self.values[:,:width]/.03).softmax(1)
                count=torch.bincount(pair.flatten(),weights.flatten(),minlength=c.groups*len(bank)).view(c.groups,-1)
                marginal=count.sum(0).clamp_min(1.)
                votes=count/marginal.pow(c.distinct)
                weight,ids=votes.topk(c.words)
                mass=weight/weight.sum(1,keepdim=True).clamp_min(1e-12)
            else:
                centers=means(self.rgb,assigned,c.groups)
                similarity=centers@self.target.T-c.distinct*(self.rgb.mean(0)@self.target.T)
                value,ids=similarity.topk(c.words)
                mass=(value/.03).softmax(1)
            names=F.normalize(torch.bmm(mass[:,None],bank[ids]).squeeze(1))
        if c.pairing=="random_groups":
            perm=torch.randperm(c.groups,generator=torch.Generator().manual_seed(5109)).to(names.device)
            names=names[perm]
        self.codes[key]=names
        return names

    def scores(self,queries,c,block_size=144):
        raw=self.rgb if c.pairing=="rgb" else self.raw_codes["neutral" if c.pairing=="neutral" else "paired"]
        if c.mix:
            assigned,proximity=self.partition(c)
            codes=self.names(c,assigned)
            key=(c.groups,c.geometry,c.image_temp,c.prior)
            if key not in self.assignments:
                affiliation=(proximity/c.image_temp).softmax(1)
                size=affiliation.sum(0).clamp_min(1e-8)
                self.assignments[key]=(affiliation/size.pow(c.prior)).T.contiguous()
            conditional=self.assignments[key]
        for query in queries.split(block_size):
            direct=standardize(query@raw.T)
            if c.mix:
                query_groups=((query@codes.T)/c.query_temp).softmax(1)
                likelihood=(query_groups@conditional).clamp_min(1e-30).log()
                direct=direct.lerp(standardize(likelihood),c.mix)
            yield direct
