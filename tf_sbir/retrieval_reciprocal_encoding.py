"""One-query-at-a-time weighted reciprocal re-ranking against a fixed photo index."""

from dataclasses import dataclass, replace
import os

import numpy as np
import torch
from numba import njit, prange, set_num_threads, get_num_threads
from scipy.sparse import csr_matrix, eye
from torch.nn import functional as F

from .image_retrieval_round import DECISION

USES_LITERAL=True
FULL_FINALISTS=2
MAX_K=120


@dataclass(frozen=True)
class Config:
    k: int = 20
    expansion: int = 6
    original_weight: float = .3
    backoff: str = "geometry"
    geometry: str = "semantic"
    neighborhoods: str = "reciprocal"


def canonical():
    return Config()


def candidates():
    settings=[(20,6),(60,6),(120,6),(60,1),(60,12)]
    return [Config(k,e,a,b) for k,e in settings for a in (.3,.7) for b in ("geometry","likelihood")]


def controls(c):
    return {"primary":c, "paper_settings":canonical(),"no_reranking":replace(c,original_weight=1.),
            "no_original_score":replace(c,original_weight=0.), "one_way":replace(c,neighborhoods="one_way"),
            "no_half_set_expansion":replace(c,neighborhoods="no_half"),"no_local_query_expansion":replace(c,expansion=1),
            "raw_rgb_geometry":replace(c,geometry="rgb",backoff="geometry"),
            "shuffled_gallery_context":replace(c,neighborhoods="shuffle"),"exact_word_seed":replace(c,backoff="likelihood",original_weight=1.)}


def cache_key(c):
    return c.geometry,c.k,c.neighborhoods,c.expansion,c.backoff,c.original_weight


def nearest(x,p,exclude_self):
    ids,dist=[],[]
    for start in range(0,len(x),256):
        d=(1.-x[start:start+256]@p.T).mul_(.5).clamp_(0.,1.)
        if exclude_self:
            idx=torch.arange(len(d),device=p.device)
            d[idx,start+idx]=torch.inf
        vals,columns=d.topk(MAX_K,largest=False)
        ids.append(columns.int().cpu().numpy());dist.append(vals.cpu().numpy())
    return np.concatenate(ids),np.concatenate(dist)


def binary(neighbors,n,keep=None):
    rows=np.broadcast_to(np.arange(len(neighbors),dtype=np.int32)[:,None],neighbors.shape)
    r,c=(rows.ravel(),neighbors.ravel()) if keep is None else (rows[keep],neighbors[keep])
    return csr_matrix((np.ones(len(r),np.float32),(r,c)),shape=(len(neighbors),n))


@njit(cache=True)
def expand_indices(ptr,idx,hptr,hidx,n):
    """Zhong Eq. 4, using the original reciprocal set for all overlap tests."""
    outptr=np.zeros(len(ptr),np.int32)
    output=np.empty(0,np.int32)
    mark=np.full(n,-1,np.int64)
    seen=np.full(n,-1,np.int64)
    for phase in range(2):
        total=0
        for i in range(len(ptr)-1):
            tag=phase*len(ptr)+i
            for pos in range(ptr[i],ptr[i+1]):
                j=idx[pos]
                mark[j]=tag;seen[j]=tag
                if phase: output[total]=j
                total+=1
            for pos in range(ptr[i],ptr[i+1]):
                j=idx[pos]
                left,right=hptr[j],hptr[j+1]
                count=0
                for t in range(left,right):
                    count+=mark[hidx[t]]==tag
                if 3*count<=2*(right-left):
                    continue
                for t in range(left,right):
                    v=hidx[t]
                    if seen[v]!=tag:
                        seen[v]=tag
                        if phase: output[total]=v
                        total+=1
            if not phase: outptr[i+1]=total
        if not phase: output=np.empty(total,np.int32)
    return outptr,output


def encoded(core,half,x,p,expand):
    core.sort_indices()
    if expand:
        ptr,ids=expand_indices(core.indptr,core.indices,half.indptr,half.indices,len(p))
    else:
        ptr,ids=core.indptr,core.indices
    origin=np.repeat(np.arange(len(x),dtype=np.int32),np.diff(ptr))
    weights=np.empty(len(ids),np.float32)
    for start in range(0,len(ids),16384):
        rr=torch.from_numpy(origin[start:start+16384]).to(x.device,dtype=torch.long)
        cc=torch.from_numpy(ids[start:start+len(rr)]).to(x.device,dtype=torch.long)
        weights[start:start+len(rr)]=((x[rr]*p[cc]).sum(1).sub_(1.).mul_(.5).exp_()).cpu().numpy()
    out=csr_matrix((weights,ids,ptr),shape=core.shape)
    sums=np.asarray(out.sum(1)).ravel()
    out.data/=np.repeat(np.maximum(sums,1e-30),np.diff(out.indptr))
    out.sort_indices()
    return out


@njit(parallel=True,cache=True)
def overlap_block(qptr,qids,qval,gptr,gids,gval,g_sums,start,stop):
    result=np.zeros((stop-start,len(g_sums)),np.float32)
    for i in prange(stop-start):
        row=start+i
        mass=np.float32(0.)
        for j in range(qptr[row],qptr[row+1]):
            c=qids[j]; val=qval[j]; mass+=val
            for h in range(gptr[c],gptr[c+1]):
                result[i,gids[h]]+=min(val,gval[h])
        for d in range(len(g_sums)):
            intersection=result[i,d]
            result[i,d]=intersection/max(np.float32(1e-30),mass+g_sums[d]-intersection)
    return result


class Index:
    def __init__(self,data):
        self.d=data
        self.hq=F.normalize(data.query_emission@data.literal,dim=1)
        self.hp=F.normalize(data.a@data.literal,dim=1)
        self.info={"gallery":"fixed across independent queries","distance":"half of one minus cosine",
                   "graph_components":{}}
        set_num_threads(max(1,min(8,len(os.sched_getaffinity(0)),get_num_threads())))
        self.last_geometry=None
        self.last_key=None
        self.cached=None

    def geometry(self,c):
        if c.geometry==self.last_geometry:
            return
        self.cached=None;self.last_key=None
        self.q,self.p=(self.hq,self.hp) if c.geometry=="semantic" else (self.d.q,self.d.p)
        self.gi,self.gd=nearest(self.p,self.p,True)
        self.qi,self.qd=nearest(self.q,self.p,False)
        self.direct=(self.q@self.p.T).add_(1.).mul_(.5)
        self.last_geometry=c.geometry

    def context(self,c):
        treatment="reciprocal" if c.neighborhoods=="shuffle" else c.neighborhoods
        key=c.k,c.expansion,treatment
        if self.last_key==key:
            return self.cached
        self.cached=None
        k=c.k;n=len(self.p)
        g=binary(self.gi[:,:k],n);half=binary(self.gi[:,:k//2],n)
        qkeep=None
        if treatment!="one_way":
            g=g.minimum(g.T);half=half.minimum(half.T)
            qkeep=self.qd[:,:k]<=self.gd[self.qi[:,:k],k-1]
        g=g+eye(n,format="csr",dtype=np.float32)
        half=half+eye(n,format="csr",dtype=np.float32)
        q=binary(self.qi[:,:k],n,qkeep)
        g0=encoded(g,half,self.p,self.p,treatment!="no_half")
        q0=encoded(q,half,self.q,self.p,treatment!="no_half")
        if c.expansion>1:
            count=c.expansion-1
            q0=(q0+binary(self.qi[:,:count],n)@g0)/c.expansion
            g0=(g0+binary(self.gi[:,:count],n)@g0)/c.expansion
        inverted=g0.tocsc()
        g_sums=np.asarray(g0.sum(1)).ravel()
        self.cached=self.direct.new_empty(self.direct.shape)
        for start in range(0,len(self.q),256):
            stop=min(start+256,len(self.q))
            block=overlap_block(q0.indptr,q0.indices,q0.data,inverted.indptr,inverted.indices,inverted.data,g_sums,start,stop)
            self.cached[start:stop]=torch.from_numpy(block).to(self.p.device)
        self.info["graph_components"][str((c.geometry,*key))]={
            "mean_reciprocal_query_core":float(np.diff(q.indptr).mean()),
            "queries_without_core":int((np.diff(q.indptr)==0).sum()),
            "mean_nonzero_photo_weights":float(g0.nnz/n)}
        self.last_key=key
        return self.cached

    def score(self,c):
        self.geometry(c)
        direct=self.direct if c.backoff=="geometry" else self.d.base
        if c.original_weight==1.:
            return direct
        jaccard=self.context(c)
        if c.neighborhoods=="shuffle":
            ids=torch.randperm(len(self.p),generator=torch.Generator().manual_seed(71633)).to(self.p.device)
            jaccard=jaccard[:,ids]
        return jaccard.lerp(direct,c.original_weight)
