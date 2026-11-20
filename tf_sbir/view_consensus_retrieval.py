"""Estimate a query from agreement across its own views and discriminative gallery feedback."""

from dataclasses import dataclass, replace

import torch
from torch.nn import functional as F

from .gallery_diffusion import nearest_gallery
from .reranking import standardize
from .reverse_prompt import nearest_words, term_average


USE_VIEWS = True


@dataclass(frozen=True)
class Config:
    query_views: int = 6
    photo_views: int = 6
    pool: str = "uniform"
    terms: int = 8
    neighbors: int = 64
    rounds: int = 2
    support: int = 96
    expansion: float = .65
    ridge: float = 0.
    pairing: str = "paired"

    @property
    def name(self):
        return (f"v{self.query_views}-{self.photo_views}-{self.pool}-c{self.terms}-k{self.neighbors}"
                f"x{self.rounds}-n{self.support}-e{self.expansion:g}-r{self.ridge:g}-{self.pairing}")


def candidates():
    base = Config()
    grid = {replace(base,query_views=q,photo_views=p,pool=pool)
            for q in (1,2,6) for p in (1,2,6) for pool in (("uniform",) if q==1 else ("uniform","agreement","entropy"))}
    for attr, values in {"terms":(4,16,32),"rounds":(0,1,3),"neighbors":(24,128,256),
                         "support":(24,48,192,384),"expansion":(0.,.35,.85,1.),"ridge":(.1,1.,10.)}.items():
        grid.update(replace(base, **{attr:v}) for v in values)
    return sorted(grid,key=lambda c:c.name)


def refine(c):
    return list(dict.fromkeys([c,*[replace(c,neighbors=k,rounds=t) for k in (32,96,192) for t in (1,2,3)],
        *[replace(c,support=n,expansion=e) for n in (32,96,192) for e in (.4,.7,.9)],
        *[replace(c,terms=v) for v in (4,12,24)], *[replace(c,ridge=r) for r in (.1,1.,5.)],
        *[replace(c,pool=p) for p in ("agreement","uniform","entropy")]]))


def controls(c):
    return {"primary":c,"one_sketch_view":replace(c,query_views=1,pool="uniform"),
        "one_photo_view":replace(c,photo_views=1),"one_view_both":replace(c,query_views=1,photo_views=1,pool="uniform"),
        "uniform_views":replace(c,pool="uniform"),"agreement":replace(c,pool="agreement"),
        "without_gallery_averaging":replace(c,rounds=0),"without_feedback":replace(c,expansion=0.),
        "centroid_feedback":replace(c,ridge=0.), "shuffled_words":replace(c,pairing="shuffled"),
        "rgb_only":replace(c,pairing="rgb")}


def vocab_size(c):
    return 0 if c.pairing == "rgb" else 20931


class Index:
    def __init__(self, photos, source, target):
        self.photos, self.source, self.target = photos, source, target
        self.identity = torch.arange(len(source),device=source.device)
        self.shuffle = torch.randperm(len(source),generator=torch.Generator().manual_seed(5109)).to(source.device)
        self.bases, self.graphs, self.augmented, self.inverses = {}, {}, {}, {}

    def base(self,n):
        if n not in self.bases:
            rgb = F.normalize(self.photos[:,:n].mean(1))
            geometry = F.normalize(rgb-rgb.mean(0))
            ids, values = nearest_words(rgb,self.target,32)
            self.bases[n] = (rgb,geometry,ids,values)
        return self.bases[n]

    def operator(self,n,k):
        key = n,k
        if key not in self.graphs:
            geom = self.base(n)[1]
            ids, similarities = nearest_gallery(geom,k)
            weights = (similarities/.08).softmax(1)
            rows = torch.arange(len(geom),device=geom.device).repeat_interleave(k)
            self.graphs[key] = torch.sparse_coo_tensor(torch.stack((rows,ids.flatten())),weights.flatten(),
                (len(geom),len(geom))).coalesce().to_sparse_csr()
        return self.graphs[key]

    def database(self,c,kind):
        key = (kind,c.photo_views,c.neighbors,c.rounds,
               c.terms if kind=="words" else 0,c.pairing if kind=="words" else "")
        if key not in self.augmented:
            rgb,geom,ids,values = self.base(c.photo_views)
            if kind=="visual":
                base=geom
            elif c.pairing=="rgb":
                base=rgb
            else:
                mapping=self.shuffle if c.pairing=="shuffled" else self.identity
                base=term_average(self.source,mapping,ids,values,c.terms,.03)
            current=base
            for _ in range(c.rounds):
                current=F.normalize(.30*base+.70*torch.sparse.mm(self.operator(c.photo_views,c.neighbors),current))
            self.augmented[key]=current
        return self.augmented[key]

    def discriminate(self,feedback,visual,c):
        if not c.ridge:
            return feedback
        key=(c.photo_views,c.neighbors,c.rounds,c.ridge)
        if key not in self.inverses:
            centered=visual-visual.mean(0)
            cov=(centered.T@centered)/len(centered)
            cov.diagonal().add_(c.ridge*cov.diagonal().mean())
            self.inverses[key]=(torch.linalg.cholesky(cov),visual.mean(0))
        chol,mean=self.inverses[key]
        return torch.cholesky_solve((feedback-mean).T,chol).T

    @staticmethod
    def merge(raw,pool):
        if raw.size(1)==1:
            return standardize(raw[:,0])
        if pool=="uniform":
            return standardize(raw.mean(1))
        probabilities=(raw/.02).softmax(-1)
        if pool=="agreement":
            # Bhattacharyya affinity between the photo distributions of this sketch's views.
            roots=probabilities.sqrt()
            reliability=(roots @ roots.mT).mean(-1)
            weights=(reliability/.075).softmax(1)
        else:
            certainty=(probabilities*probabilities.clamp_min(1e-30).log()).sum(-1)
            weights=(certainty/.5).softmax(1)
        return standardize((raw*weights[:,:,None]).sum(1))

    def scores(self,queries,c,block_size=128):
        words=self.database(c,"words")
        visual=self.database(c,"visual")
        for q in queries.split(block_size):
            base=self.merge(q[:,:c.query_views] @ words.T,c.pool)
            if c.expansion:
                mass,ids=base.topk(min(len(visual),c.support))
                feedback=torch.bmm((mass/.6).softmax(1)[:,None],visual[ids]).squeeze(1)
                feedback=F.normalize(self.discriminate(feedback,visual,c))
                base=base.lerp(standardize(feedback@visual.T),c.expansion)
            yield base
