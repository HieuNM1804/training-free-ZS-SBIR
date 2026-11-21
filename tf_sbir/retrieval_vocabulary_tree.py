"""Index soft CLIP word counts on a public vocabulary tree, with exact sparse intersection."""

from dataclasses import dataclass, replace
import torch
from torch.nn import functional as F

from .image_retrieval_round import DECISION

USES_LITERAL=True
FULL_FINALISTS=2


@dataclass(frozen=True)
class Config:
    terms: int = 16
    levels: str = "all"
    metric: str = "intersection"
    idf: bool = True
    control: str = "matched"


def canonical():
    return Config()


def candidates():
    return [Config(k,level,m) for k in (4,16,64) for level in ("all","internal")
            for m in ("intersection","cosine")]


def controls(c):
    return {"primary":c,"canonical":canonical(),"exact_leaf_only":replace(c,levels="leaf"),
            "no_idf":replace(c,idf=False),"shuffled_tree":replace(c,control="shuffle"),
            "permuted_tie_order":replace(c,control="ties"),"seed":replace(c,control="seed")}


def cache_key(c):
    return c.terms,c.levels,c.idf,c.control,c.metric


def split_group(x):
    k=min(10,len(x))
    if k==len(x):
        return torch.arange(k,device=x.device)
    # A reproducible farthest-first initialisation for spherical Lloyd iterations.
    centers=[x[0]]
    best=x@x[0]
    for _ in range(1,k):
        point=x[best.argmin()]
        centers.append(point)
        best=torch.maximum(best,x@point)
    centers=torch.stack(centers)
    old=None
    for _ in range(24):
        ids=(x@centers.T).argmax(1)
        if old is not None and torch.equal(ids,old):
            break
        counts=ids.bincount(minlength=k)
        sums=torch.zeros_like(centers).index_add_(0,ids,x)
        centers=torch.where(counts[:,None]>0,F.normalize(sums,dim=1),centers)
        old=ids
    return ids


def public_tree(words):
    parent=torch.zeros(len(words),dtype=torch.long,device=words.device)
    levels=[]
    for _ in range(3):
        child=torch.empty_like(parent)
        offset=0
        for node in parent.unique(sorted=True):
            group=torch.where(parent==node)[0]
            labels=split_group(words[group])
            unique,local=torch.unique(labels,sorted=True,return_inverse=True)
            child[group]=local+offset
            offset+=len(unique)
        levels.append(child)
        parent=child
    return levels


def top_terms(response,k):
    mass,ids=response.topk(k,dim=1)
    return mass/mass.sum(1,keepdim=True),ids


def representation(q,p,lookup,use_idf,atomic):
    qmass,qids=q; pmass,pids=p
    counts=lookup.bincount()
    w=(len(lookup)/counts.float()).log() if use_idf else torch.ones_like(counts,dtype=pmass.dtype)
    docs=pmass.new_zeros(len(counts),len(pmass))
    rows=lookup[pids]
    docs.scatter_add_(0,rows.T,(pmass*w[rows]).T)
    col=lookup[qids]
    if atomic:
        vals=qmass*w[col]
    else:
        dense=qmass.new_zeros(len(qmass),len(counts)).scatter_add_(1,col,qmass*w[col])
        vals,col=dense.topk(min(qmass.shape[1],len(counts)),dim=1)
    return vals,col,docs


class Index:
    def __init__(self,data):
        self.d=data
        self.levels=public_tree(data.literal)
        self.ids=torch.arange(len(data.literal),device=data.q.device)
        self.perm=torch.randperm(len(self.ids),generator=torch.Generator().manual_seed(47565)).to(data.q.device)
        self.info={"nodes":[int(v.max())+1 for v in self.levels],"reference_documents":"20931 standalone public words",
                   "branching":10,"max_lloyd_iterations":24,"sample_score_ties":{}}

    def score(self,c):
        if c.control=="seed":
            return self.d.base
        q=top_terms(self.d.query_emission,c.terms)
        p=top_terms(self.d.a,c.terms)
        levels=[] if c.levels=="leaf" else self.levels
        if c.control=="shuffle":
            levels=[lookup[self.perm] for lookup in levels]
        features=[representation(q,p,lookup,c.idf,False) for lookup in levels]
        if c.levels!="internal":
            features.append(representation(q,p,self.ids,c.idf,True))
        if c.metric=="intersection":
            nq=sum(v.sum(1) for v,_,_ in features).clamp_min(1e-30)
            np=sum(d.sum(0) for _,_,d in features).clamp_min(1e-30)
        else:
            nq=sum(v.square().sum(1) for v,_,_ in features).clamp_min(1e-30).sqrt()
            np=sum(d.norm(dim=0).square() for _,_,d in features).clamp_min(1e-30).sqrt()
        scores=self.d.q.new_zeros(len(nq),len(np))
        for mass,ids,docs in features:
            docs/=np[None,:]
            for start in range(0,len(nq),32):
                v=(mass[start:start+32]/nq[start:start+32,None])[:,:,None]
                hits=docs[ids[start:start+len(v)]]
                contribution=torch.minimum(v,hits) if c.metric=="intersection" else v*hits
                scores[start:start+len(v)]+=contribution.sum(1)
        self.info["sample_score_ties"][str(c)]={"fraction_zero":float((scores[:32]==0).float().mean())}
        if c.control=="ties":
            order=torch.randperm(scores.size(1),generator=torch.Generator().manual_seed(64393)).to(scores.device)
            for row in scores.split(64):
                ranks=order[row[:,order].argsort(dim=1,descending=True,stable=True)]
                values=torch.arange(ranks.size(1),0,-1,device=row.device,dtype=row.dtype)
                row.scatter_(1,ranks,values.expand_as(row))
        return scores
