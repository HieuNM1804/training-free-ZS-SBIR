"""Transport reproducible ordinal word tests between rival unlabeled gallery modes."""

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

from .cluster_mine import cluster_assignments


class OrdinalRivals:
    def __init__(self,p,prob,target,source,prior,groups,view):
        rgb=F.normalize(p-p.mean(0))
        if view=="joint":
            semantic=prob@target
            rgb=F.normalize(torch.cat((rgb,F.normalize(semantic-semantic.mean(0))),1))
        assigned=cluster_assignments(rgb,groups)
        _,self.groups=torch.unique(assigned,sorted=True,return_inverse=True)
        self.count=int(self.groups.max())+1
        self.size=torch.bincount(self.groups,minlength=self.count).float()
        coarse=prob.new_zeros((self.count,prob.size(1))).index_add_(0,self.groups,prob)/self.size[:,None]
        vocab=torch.cat((prior.topk(160).indices,coarse.topk(10,dim=1).indices.flatten())).unique()
        self.source=source[vocab]
        self.terms=vocab
        self.center=p.mean(0)
        self.proportions=coarse[:,vocab]
        t=target[vocab]
        self.glogits=p@t.T
        self.ordering=torch.triu_indices(self.count,self.count,offset=1,device=p.device)
        # A random balanced split inside each discovered group never consults class labels.
        rng=torch.Generator().manual_seed(45613)
        half=torch.zeros(len(p),dtype=torch.long)
        cpu=self.groups.cpu()
        for c in range(self.count):
            ix=(cpu==c).nonzero().flatten()
            half[ix[torch.randperm(len(ix),generator=rng)[::2]]]=1
        self.fold=half.cuda()
        self.frequency=self._frequencies()
        self.indices,self.pp,self.pn,self.stability=self._matched()

    def _frequencies(self):
        nword=self.glogits.size(1)
        result=self.glogits.new_zeros((2*self.count,nword*nword))
        category=self.groups+self.fold*self.count
        for i in range(0,len(category),64):
            x=self.glogits[i:i+64]
            prefer=(x[:,:,None]>x[:,None,:]).flatten(1).float()
            result.index_add_(0,category[i:i+len(x)],prefer)
        size=torch.bincount(category,minlength=2*self.count).float()
        return ((result+.5)/(size[:,None]+1)).reshape(2,self.count,nword,nword)

    def _matched(self):
        indices=[]; positive=[]; negative=[]; stability=[]
        size=24
        for block in self.ordering.split(96,dim=1):
            a,b=block.unbind(0)
            pa,pb=self.proportions[a],self.proportions[b]
            mid=.5*(pa+pb)
            fa=pa*(pa.clamp_min(1e-30)/mid.clamp_min(1e-30)).clamp_min(1e-30).log()
            fb=pb*(pb.clamp_min(1e-30)/mid.clamp_min(1e-30)).clamp_min(1e-30).log()
            aa=fa.topk(size,dim=1).indices
            bb=fb.topk(size,dim=1).indices
            wa=aa[:,:,None].expand(-1,-1,size)
            wb=bb[:,None,:].expand(-1,size,-1)
            pos=self.frequency[:,a[:,None,None],wa,wb]
            neg=self.frequency[:,b[:,None,None],wa,wb]
            # Signed reproducibility means the relation favors the same group
            # on both halves. JS support chooses words; empirical bits choose edges.
            d=pos-neg
            mag=torch.minimum(d[0].abs(),d[1].abs())
            distinct_support=(fa.gather(1,aa)>0)[:,:,None] & (fb.gather(1,bb)>0)[:,None,:]
            cred=mag*(d[0]*d[1]>0)*(wa!=wb)*distinct_support
            weights=cred.cpu().numpy()
            columns=np.stack([linear_sum_assignment(w,maximize=True)[1] for w in weights])
            cols=torch.from_numpy(columns).to(a.device)
            k=cred.gather(2,cols[:,:,None]).squeeze(2)
            order=k.argsort(dim=1,descending=True,stable=True)
            left=aa.gather(1,order)
            right=bb.gather(1,cols.gather(1,order))
            term_pos=self.frequency[:,a[:,None],left,right].mean(0)
            term_neg=self.frequency[:,b[:,None],left,right].mean(0)
            indices.append(torch.stack((left,right),dim=2))
            positive.append(term_pos);negative.append(term_neg);stability.append(k.gather(1,order))
        return tuple(torch.cat(v) for v in (indices,positive,negative,stability))

    def comparisons(self,q,length,mode):
        x=q@self.source.T
        if mode=="offset":x=x-(self.center@self.source.T)
        return self.from_word_scores(x,length,mode)

    def from_word_scores(self,x,length,mode):
        pairs=self.indices[:,:length]
        a=x[:,pairs[:,:,0]]
        b=x[:,pairs[:,:,1]]
        sign=a>b
        pos,neg=self.pp[:,:length],self.pn[:,:length]
        clip=1e-6
        yes=pos.clamp_min(clip).log()-neg.clamp_min(clip).log()
        no=(1-pos).clamp_min(clip).log()-(1-neg).clamp_min(clip).log()
        if mode=="identity_shuffle":
            perm=torch.randperm(x.size(1),generator=torch.Generator().manual_seed(891)).to(x.device)
            sign=x[:,perm[pairs[:,:,0]]]>x[:,perm[pairs[:,:,1]]]
        if mode=="signed":
            value=(2*sign.float()-1)*torch.sign(pos-neg)
        elif mode=="continuous":
            # The bit probability is a soft readout at the fixed CLIP logit scale.
            sm=torch.sigmoid((a-b)*100)
            value=sm*yes+(1-sm)*no
        else:
            value=torch.where(sign,yes,no)
        weight=self.stability[:,:length]
        if mode=="uniform_edges":weight=(weight>0).to(weight)
        return (value*weight).sum(2)/weight.sum(1).clamp_min(1e-6)

    def group_mean(self,base):
        return (base.new_zeros((self.count,len(base))).index_add_(0,self.groups,base.T)/self.size[:,None]).T

    def calibration(self,base,odds,foil_power):
        prior=self.group_mean(base).clamp_min(1e-30).log()
        w=(prior*foil_power).softmax(1)
        pair=base.new_zeros((len(base),self.count,self.count))
        a,b=self.ordering
        pair[:,a,b]=odds
        pair[:,b,a]=-odds
        projected=(pair@w[:,:,None]).squeeze(2)
        return projected,prior,w

    def rank(self,base,odds,foil_power,strength,operation):
        score,prior,w=self.calibration(base,odds,foil_power)
        if operation=="replace":
            score=score-(prior-(w*prior).sum(1,keepdim=True))
        return base.clamp_min(1e-30).log()+strength*score[:,self.groups]
