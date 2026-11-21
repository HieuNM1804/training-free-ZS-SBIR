"""Match scale dependence in query and gallery with an exact same-marginal null."""

import torch
from types import SimpleNamespace

from .cross_diffusion_map import Config,Index
from .regional_interventions import windows
from .shared_concept_likelihood import fit_prior


def edges(device):
    e=torch.zeros(4,9,device=device)
    for a,(y,x) in enumerate(windows(5)):
        for b,(v,u) in enumerate(windows(3)):
            e[a,b]=int(y<=v and v+3<=y+5 and x<=u and u+3<=x+5)
    e/=e.sum()
    if int((e>0).sum())!=16:raise ValueError("Unexpected number of nested windows")
    independent=e.sum(1)[:,None]*e.sum(0)
    return e,independent


def word_bank(photos, crops, target, scale, n):
    em=photos.new_empty((len(photos),len(target)))
    for start in range(0,len(photos),256):
        sl=slice(start,start+256)
        em[sl]=((photos[sl]@target.T)*scale).softmax(1)
    prior=fit_prior(em)[0]["mle"]
    heads=prior.topk(n).indices
    del em
    ix=torch.randperm(len(photos),generator=torch.Generator().manual_seed(99371))[:2048]
    variance=torch.zeros(len(target),device=photos.device,dtype=torch.float64)
    for chunk in ix.split(32):
        x=crops[chunk,:9].cuda().float()
        z=(x@target.T)*scale
        prob=z.softmax(-1)
        variance+=prob.var(1,correction=0).sum(0,dtype=torch.float64)
    return heads,variance.topk(2*n).indices


def picture(crops,outer_words,inner_words,scale,temp):
    e,ind=edges(crops.device)
    outer=(crops[:,9:].float()@outer_words.T*scale*temp).softmax(-1)
    inner=(crops[:,:9].float()@inner_words.T*scale*temp).softmax(-1)
    paired=(outer.transpose(1,2)@e)@inner
    null=(outer.transpose(1,2)@ind)@inner
    shifted=(outer.transpose(1,2)@e.roll(1,0))@inner
    return null,paired,shifted


class PairIndex:
    def __init__(self, photos, crops, target, scale, heads, parts, temp, weighting, with_wrong=False):
        graph=Index(SimpleNamespace(p=photos),rgb_only=True)
        config=Config(neighbors=16,bandwidth=.36,iterations=2,mode="rgb")
        self.marginal_error=0.
        chunked=[[] for _ in range(3 if with_wrong else 2)]
        for start in range(0,len(photos),192):
            j=picture(crops[start:start+192].cuda(),target[heads],target[parts],scale,temp)
            for v in j[1:]:
                self.marginal_error=max(self.marginal_error,float((v.sum(1)-j[0].sum(1)).abs().max()),
                                        float((v.sum(2)-j[0].sum(2)).abs().max()))
            for bag,val in zip(chunked,j):
                bag.append(val.flatten(1).cpu())
        if self.marginal_error>2e-5:raise ValueError("Changing containment altered a word marginal")
        self.profiles=[]
        for blocks in chunked:
            bank=torch.cat(blocks).cuda()
            out=torch.empty_like(bank)
            for start in range(0,bank.size(1),256):
                out[:,start:start+256]=graph.score(config,bank[:,start:start+256].T.contiguous()).T
            norm=out.sum(1,keepdim=True)
            if not torch.isfinite(norm).all() or (norm<=0).any():raise ValueError("Invalid gallery scale profiles")
            self.profiles.append(out/norm)
            del bank,out
            blocks.clear()
        self.heads=heads
        self.parts=parts
        self.temp=temp
        self.scale=scale
        mean=self.profiles[0].mean(0).reshape(len(heads),len(parts))
        outer=mean.sum(1)[:,None]
        inner=mean.sum(0)[None,:]
        self.inv=(outer*inner).clamp_min(1e-15).flatten().pow(-weighting)

    def overlaps(self,crops,source,wrong=False):
        u,p,s=picture(crops,source[self.heads],source[self.parts],self.scale,self.temp)
        q={name:val.flatten(1)*self.inv for name,val in (("0",u),("1",p),("x",s))}
        slots={"00":("0",0),"10":("1",0),"01":("0",1),"11":("1",1)}
        if wrong:slots.update({"x0":("x",0),"x1":("x",1),"0x":("0",2),"xx":("x",2)})
        return {name:(q[part]@self.profiles[i].T).clamp_min(1e-30).log() for name,(part,i) in slots.items()}


def correction(z,kind):
    crossed=z["11"]+z["00"]-z["10"]-z["01"]
    if kind=="cross":return crossed
    if kind=="null":return 0.
    if kind=="direct":return z["11"]-z["00"]
    if kind=="query":return z["10"]-z["00"]
    if kind=="gallery":return z["01"]-z["00"]
    if kind=="wrong_query":return z["x1"]+z["00"]-z["x0"]-z["01"]
    if kind=="wrong_both":return z["xx"]+z["00"]-z["x0"]-z["0x"]
    if kind=="reverse":return -crossed
    raise ValueError(kind)
