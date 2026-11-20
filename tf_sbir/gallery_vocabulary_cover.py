"""Discover a compact public lexicon covering the unlabeled gallery and transfer its evidence."""

from dataclasses import dataclass,replace

import torch
from torch.nn import functional as F

from .gallery_diffusion import nearest_gallery
from .reranking import standardize
from .reverse_prompt import nearest_words,term_average


USE_VIEWS=False


@dataclass(frozen=True)
class Config:
    words: int=256
    selection: str="frequency"
    evidence_temp: float=.02
    direct: float=.25
    support: int=96
    expansion: float=.65
    neighbors: int=0
    pairing: str="paired"

    @property
    def name(self):
        return (f"vocab{self.words}-{self.selection}-t{self.evidence_temp:g}-d{self.direct:g}"
                f"-n{self.support}-e{self.expansion:g}-k{self.neighbors}-{self.pairing}")


def candidates():
    c=Config()
    grid={replace(c,words=n,selection=s,direct=d) for n in (48,96,192,320,512,768)
          for s in ("frequency","cover","hybrid") for d in (0.,.4)}
    for field,values in {"evidence_temp":(.006,.012,.035,.06),"direct":(.1,.6,.8),"neighbors":(32,96,192),
                         "expansion":(0.,.35,.85,1.),"support":(24,48,192,384)}.items():
        grid.update(replace(c,**{field:v}) for v in values)
    return sorted(grid,key=lambda v:v.name)


def refine(c):
    return list(dict.fromkeys([c,*[replace(c,words=n,evidence_temp=t) for n in (64,128,224,384,640,1024)
        for t in (.01,.025,.045)],*[replace(c,direct=d,expansion=e) for d in (0.,.2,.5,.7)
        for e in (.4,.7,.9)],*[replace(c,support=n) for n in (48,160,256)],
        *[replace(c,neighbors=k) for k in (0,48,144)],*[replace(c,selection=s) for s in ("cover","frequency","hybrid")]]))


def controls(c):
    return {"primary":c,"mass_only":replace(c,selection="frequency"),"coverage_only":replace(c,selection="cover"),
        "uniform_candidates":replace(c,selection="random"),"unrestricted":replace(c,direct=1.),
        "restricted_only":replace(c,direct=0.),"no_feedback":replace(c,expansion=0.),
        "no_gallery_averaging":replace(c,neighbors=0),"neutral":replace(c,pairing="neutral"),
        "shuffled_words":replace(c,pairing="shuffled"),"rgb_only":replace(c,pairing="rgb")}


def vocab_size(c):
    return 0 if c.pairing=="rgb" else 20931


class Index:
    def __init__(self,photos,source,target):
        self.p,self.source,self.target=photos,source,target
        self.visual=F.normalize(photos-photos.mean(0))
        self.ids,self.values=nearest_words(photos,target,16)
        freq=torch.bincount(self.ids.flatten(),weights=(self.values/.02).softmax(1).flatten(),minlength=len(target))
        self.freq=freq.argsort(descending=True,stable=True)
        self.randint=torch.randperm(len(target),generator=torch.Generator().manual_seed(781)).to(photos.device)
        self.shuffle=torch.randperm(len(target),generator=torch.Generator().manual_seed(8244)).to(photos.device)
        self.orders={"frequency":self.freq,"random":self.randint}
        self.banks,self.graphs,self.visual_banks,self.global_words={},{},{},{}

    def cover(self):
        # Submodular facility location. Each selected term covers reference
        # photos that have a weak match to all of the previously selected terms.
        candidates=self.freq[:2048]
        generator=torch.Generator().manual_seed(711)
        sample=torch.randperm(len(self.p),generator=generator)[:1536].to(self.p.device)
        logits=self.p[sample]@self.target[candidates].T
        utility=((logits-logits.amax(1,keepdim=True))/.035).exp()
        current=utility.new_zeros(len(sample),1)
        used=torch.zeros(len(candidates),dtype=torch.bool,device=utility.device)
        order=[]
        for _ in range(min(1024,len(candidates))):
            gain=(utility-current).clamp_min(0).sum(0).masked_fill(used,-torch.inf)
            idx=gain.argmax().view(1)
            order.append(idx); used.scatter_(0,idx,True)
            current=torch.maximum(current,utility.index_select(1,idx))
        return candidates[torch.cat(order)]

    def select(self,c):
        if c.selection not in self.orders:
            if "cover" not in self.orders:
                self.orders["cover"]=self.cover()
            if c.selection=="hybrid":
                # Alternating prefixes preserves both the high-frequency and
                # marginal-coverage candidate sets. The pure versions are controls.
                pairs=torch.stack((self.freq[:1024],self.orders["cover"]),1).flatten().tolist()
                self.orders["hybrid"]=torch.tensor(list(dict.fromkeys(pairs)),device=self.p.device)
        return self.orders[c.selection][:c.words]

    def augment(self,features,k):
        if not k:
            return features
        if k not in self.graphs:
            neighbors,similarity=nearest_gallery(self.visual,k)
            rows=torch.arange(len(features),device=features.device).repeat_interleave(k)
            self.graphs[k]=torch.sparse_coo_tensor(torch.stack((rows,neighbors.flatten())),
                (similarity/.08).softmax(1).flatten(),(len(features),len(features))).coalesce().to_sparse_csr()
        return F.normalize(.3*features+.7*torch.sparse.mm(self.graphs[k],features))

    def global_code(self,pairing):
        if pairing not in self.global_words:
            bank=self.target if pairing=="neutral" else self.source
            mapping=self.shuffle if pairing=="shuffled" else torch.arange(len(bank),device=bank.device)
            self.global_words[pairing]=term_average(bank,mapping,self.ids,self.values,8,.03)
        return self.global_words[pairing]

    def banks_for(self,c):
        if c.neighbors not in self.visual_banks:
            self.visual_banks[c.neighbors]=self.augment(self.visual,c.neighbors)
        if c.pairing=="rgb":
            return self.p,self.p,self.visual_banks[c.neighbors]
        ids=self.select(c)
        key=c.words,c.selection,c.evidence_temp,c.pairing,c.neighbors
        if key not in self.banks:
            src=self.target if c.pairing=="neutral" else self.source
            target_ids=self.shuffle[ids] if c.pairing=="shuffled" else ids
            weights=((self.p@self.target[ids].T)/c.evidence_temp).softmax(1)
            self.banks[key]=self.augment(F.normalize(weights@src[target_ids]),c.neighbors)
        return self.banks[key],self.global_code(c.pairing),self.visual_banks[c.neighbors]

    def scores(self,queries,c,block_size=144):
        restricted,original,visual=self.banks_for(c)
        for q in queries.split(block_size):
            initial=standardize(q@restricted.T).lerp(standardize(q@original.T),c.direct)
            if c.expansion:
                values,ids=initial.topk(c.support)
                feedback=F.normalize(torch.bmm((values/.6).softmax(1)[:,None],visual[ids]).squeeze(1))
                initial=initial.lerp(standardize(feedback@visual.T),c.expansion)
            yield initial
