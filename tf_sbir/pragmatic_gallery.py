"""Score an image through locally competing descriptions of unlabeled gallery groups."""

import numpy as np
from scipy.cluster.hierarchy import linkage
import torch
import torch.nn.functional as F

from .cluster_mine import cluster_assignments


def member_sum(values,assignment,count):
    result=values.new_zeros((count,values.size(1)))
    return result.index_add_(0,assignment,values)


def unit_rows(x):
    return x/x.sum(1,keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)


class GalleryGame:
    def __init__(self,p,photo,words,view,clusters):
        rgb=F.normalize(p-p.mean(0))
        if view=="joint":
            semantic=photo@words
            features=F.normalize(torch.cat((rgb,F.normalize(semantic-semantic.mean(0))),1))
        else:
            features=rgb
        assigned=cluster_assignments(features,clusters)
        _,self.leaf=torch.unique(assigned,sorted=True,return_inverse=True)
        self.nleaf=int(self.leaf.max())+1
        count=torch.bincount(self.leaf,minlength=self.nleaf).float()
        centers=member_sum(features,self.leaf,self.nleaf)/count[:,None]
        # Ward treats the oversegmented prototypes as equally weighted when building
        # topology; actual photo counts enter every subsequent likelihood and prior.
        tree=linkage(centers.cpu().numpy().astype(np.float64),method="ward")
        children=tree[:,:2].astype(int)
        self.left=torch.from_numpy(children[:,0].copy()).to(p.device)
        self.right=torch.from_numpy(children[:,1].copy()).to(p.device)
        self.nodes=2*self.nleaf-1
        sizes=count.tolist()+[0.]*(self.nleaf-1)
        side=np.zeros((self.nodes,self.nleaf),dtype=np.float32)
        side[:self.nleaf]=np.eye(self.nleaf,dtype=np.float32)
        self.children=children.tolist()
        for j,(a,b) in enumerate(self.children,self.nleaf):
            sizes[j]=sizes[a]+sizes[b]
            side[j]=side[a]+side[b]
        self.size=p.new_tensor(sizes)
        self.on_path=torch.from_numpy(side).to(p.device)
        self.cloud=photo

    def summarize(self,pooling):
        # Transform only one row block at a time, as the vocabulary and gallery
        # remain complete at every split of the tree.
        n=self.nleaf
        shape=(self.nodes,self.cloud.size(1))
        sums=self.cloud.new_zeros(shape)
        lower=self.cloud.new_full(shape,torch.inf)
        upper=self.cloud.new_full(shape,-torch.inf)
        for i in range(0,len(self.cloud),256):
            x=self.cloud[i:i+256]
            if pooling=="geom":
                x=x.clamp_min(1e-30).log()
            ix=self.leaf[i:i+len(x)]
            sums.index_add_(0,ix,x)
            index=ix[:,None].expand_as(x)
            lower.scatter_reduce_(0,index,x,reduce="amin",include_self=True)
            upper.scatter_reduce_(0,index,x,reduce="amax",include_self=True)
        for j,(a,b) in enumerate(self.children,n):
            torch.add(sums[a],sums[b],out=sums[j])
            torch.minimum(lower[a],lower[b],out=lower[j])
            torch.maximum(upper[a],upper[b],out=upper[j])
        counts=self.size[:,None]
        mean=sums/counts
        denom=(counts-1).clamp_min(1)
        lo=torch.where(counts>1,(sums-upper)/denom,mean)
        hi=torch.where(counts>1,(sums-lower)/denom,mean)
        if pooling=="arith":
            return unit_rows(mean),lo.clamp_min(0),hi.clamp_min(0)
        density=mean.softmax(1)
        # The log normalizer of every leave-one-photo-out word distribution lies
        # between these two logsumexp bounds.
        loglo=torch.logsumexp(lo,1,keepdim=True)
        loghi=torch.logsumexp(hi,1,keepdim=True)
        return density,(lo-loghi).exp(),(hi-loglo).exp().clamp_max(1.)

    def speakers(self,summary,alpha,bound,contrast="sibling"):
        density,lo,hi=summary
        a,b=self.left,self.right
        own=torch.cat((a,b)); other=torch.cat((b,a))
        if alpha==0: return density[own]
        low=lo[own] if bound else density[own]
        high=hi[other] if bound else density[other]
        if contrast=="unrelated":
            # Same pool of competitor descriptions, even though their assignment to
            # the decisions is now unrelated. Candidate descriptions are untouched.
            generator=torch.Generator().manual_seed(5021)
            perm=torch.randperm(len(high),generator=generator).to(high.device)
            high=high[perm]
        elif contrast=="global":
            high=density[-1:].expand_as(low)
        listener=low/(low+high).clamp_min(1e-30)
        log_speech=density[own].clamp_min(1e-30).log()+alpha*listener.clamp_min(1e-30).log()
        speech=log_speech.softmax(1)
        weak=(self.size[own]<=1)|(self.size[other]<=1)
        return torch.where(weak[:,None],density[own],speech)

    def path_likelihood(self,l,speaker):
        pair=(l@speaker.T).clamp_min(1e-30)
        a,b=self.left,self.right
        nc=len(a)
        lp=pair[:,:nc]*self.size[a]
        rp=pair[:,nc:]*self.size[b]
        normal=(lp+rp).clamp_min(1e-30)
        return (lp/normal).clamp_min(1e-30).log()@self.on_path[a]+(rp/normal).clamp_min(1e-30).log()@self.on_path[b]

    def literal_path(self,l,arith):
        # Binary literal posteriors telescope to p(leaf | query).
        e=(l@arith[0][:self.nleaf].T).clamp_min(1e-30)*self.size[:self.nleaf]
        return unit_rows(e).clamp_min(1e-30).log()

    def mean_base(self,base):
        means=member_sum(base.T.contiguous(),self.leaf,self.nleaf)/self.size[:self.nleaf,None]
        return means.T.clamp_min(1e-30).log()

    def ranking(self,logbase,leafmean,correction,within,strength):
        return within*logbase+((1-within)*leafmean+strength*correction)[:,self.leaf]


def check_removed_photo_bounds(game,summary,pooling):
    density,lo,hi=summary
    best=0.
    order=game.size[:game.nleaf].argsort(descending=True)[:5]
    for node in order.tolist():
        ids=(game.leaf==node).nonzero().flatten()
        if len(ids)<2: continue
        x=game.cloud[ids].double()
        if pooling=="geom": x=x.clamp_min(1e-30).log()
        ids=torch.linspace(0,len(x)-1,min(4,len(x)),device=x.device).long()
        direct=(x.sum(0)-x[ids])/(len(x)-1)
        if pooling=="geom": direct=direct.softmax(1)
        violation=torch.maximum(lo[node].double()-direct,direct-hi[node].double()).clamp_min(0)
        best=max(best,float(violation.max()))
    if best>2e-5: raise ValueError(f"Leave-one-image bounds violated {best}")
    return best
