"""Infer query-dependent cuts of a nested, unlabeled photo hierarchy and transfer evidence between gallery folds."""
from dataclasses import dataclass
import math

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch.nn import functional as F


def aggregate(x,groups,number):
    total=x.new_zeros((len(x),number))
    return total.index_add_(1,groups,x)


def nearest_components(features):
    n=len(features);device=features.device
    if n==1:return torch.zeros(1,device=device,dtype=torch.long)
    direction=F.normalize(features,dim=1)
    ids=[]
    for start in range(0,n,512):
        sims=direction[start:start+512]@direction.T
        row=torch.arange(len(sims),device=device)
        sims[row,start+row]=-torch.inf
        ids.append(sims.argmax(1))
    adj=csr_matrix((np.ones(n,dtype=np.int8),(np.arange(n),torch.cat(ids).cpu().numpy())),shape=(n,n))
    nnew,group=connected_components(adj,directed=False)
    if nnew>=n:raise ValueError("First-neighbor clustering failed to coarsen the gallery")
    return torch.as_tensor(group.astype(np.int64),device=device)


@dataclass
class Hierarchy:
    ids:list
    parents:list
    sizes:list

    @classmethod
    def build(cls,photos):
        # Photo order is randomized for exact-neighbor ties before any label is read.
        n=len(photos)
        perm=torch.randperm(n,generator=torch.Generator().manual_seed(86117)).to(photos.device)
        group=torch.empty_like(perm);group[perm]=torch.arange(n,device=photos.device)
        features=photos[perm];ids=[];parents=[];sizes=[]
        while len(features)>1:
            mapping=nearest_components(features)
            if ids:parents.append(mapping)
            group=mapping[group];m=int(mapping.max())+1
            count=torch.bincount(group,minlength=m)
            features=photos.new_zeros((m,photos.size(1))).index_add_(0,group,photos)/count[:,None]
            ids.append(group);sizes.append(m)
        if sizes[-1]!=1 or any(not torch.equal(up[child],parent) for up,child,parent in zip(parents,ids[:-1],ids[1:],strict=True)):
            raise ValueError("Gallery cells are not nested")
        return cls(ids,parents,sizes)

    def shuffle(self):
        perm=torch.randperm(len(self.ids[0]),generator=torch.Generator().manual_seed(37831)).to(self.ids[0].device)
        return Hierarchy([i[perm] for i in self.ids],self.parents,self.sizes)


def zscore(x):
    return (x-x.mean(1,keepdim=True))/x.std(1,keepdim=True,unbiased=False).clamp_min(1e-10)


def rank_gaussian(x):
    ordered,ix=x.sort(dim=1,stable=True)
    left=torch.searchsorted(ordered.contiguous(),ordered.contiguous())
    right=torch.searchsorted(ordered.contiguous(),ordered.contiguous(),right=True)
    percentile=(left+right).float()/(2*x.size(1))
    val=math.sqrt(2)*torch.erfinv(2*percentile-1)
    return torch.empty_like(val).scatter_(1,ix,val)


def fold_sums(values,tree,indices):
    ids=[ix[indices] for ix in tree.ids]
    sums=[aggregate(values,ix,n) for ix,n in zip(ids,tree.sizes,strict=True)]
    counts=[torch.bincount(ix,minlength=n)[None].to(values.dtype) for ix,n in zip(ids,tree.sizes,strict=True)]
    return sums,counts


def stop_weights(tree,sums,counts,noise,prior=.5):
    # Unit-variance Gaussian prior on the cell mean; noise is within-cell observation variance.
    # The pointwise null log density cancels exactly between every admissible cut.
    ll=[.5*(s.square()/(noise*(n+noise))-torch.log1p(n/noise)) for s,n in zip(sums,counts,strict=True)]
    post=[torch.ones_like(ll[0])]
    evidence=ll[0]
    for mapping,m,value in zip(tree.parents,tree.sizes[1:],ll[1:],strict=True):
        split=aggregate(evidence,mapping,m)
        joint=math.log(prior)+value
        evidence=torch.logaddexp(joint,math.log1p(-prior)+split)
        post.append((joint-evidence).exp())
    return post


def means(tree,sums,counts):
    # If this fold has no donor in a cell, fall back to its nearest populated ancestor.
    out=[None]*len(sums)
    out[-1]=sums[-1]/counts[-1].clamp_min(1)
    for j in range(len(out)-2,-1,-1):
        own=sums[j]/counts[j].clamp_min(1)
        out[j]=torch.where(counts[j]>0,own,out[j+1][:,tree.parents[j]])
    return out


def cut_read(tree,means,stops,indices):
    top=means[-1]
    prefix=stops[-1]
    total=prefix*top
    remaining=1-prefix
    for j in range(len(means)-2,-1,-1):
        parent=tree.parents[j]
        rem=remaining[:,parent];take=rem*stops[j]
        total=total[:,parent]+take*means[j]
        remaining=rem-take
    if remaining.abs().max()>1e-5:raise ValueError("Cut posterior mass did not sum to one")
    return total[:,tree.ids[0][indices]]


def folds(n,device,seed):
    perm=torch.randperm(n,generator=torch.Generator().manual_seed(seed)).to(device)
    f=torch.empty_like(perm);f[perm]=torch.arange(n,device=device)%3
    return [torch.where(f==i)[0] for i in range(3)]


def enumerate_check():
    # A two-child tree admits exactly the cut at the root and the cut at both leaves.
    ids=[torch.tensor([0,0,1]),torch.zeros(3,dtype=torch.long)]
    tree=Hierarchy(ids,[torch.tensor([0,0])],[2,1])
    data=torch.tensor([[1.,-.3,2.]],dtype=torch.float64)
    ix=torch.arange(3)
    s,n=fold_sums(data,tree,ix);st=stop_weights(tree,s,n,1.)
    logm=lambda z:-.5*math.log1p(len(z))+.5*float(z.sum())**2/(1+len(z))
    a=math.exp(logm(data[0]));b=math.exp(logm(data[0,:2])+logm(data[0,2:]))
    brute=a/(a+b)
    predicted=cut_read(tree,means(tree,s,n),st,ix)
    expected=brute*data.mean()+(1-brute)*torch.tensor([[.35,.35,2.]],dtype=torch.float64)
    err=max(abs(float(st[-1])-brute),float((predicted-expected).abs().max()))
    if err>1e-12:raise ValueError("Cut marginal differs from enumeration")
    return err
