"""Ask a query view for the current gallery distinction and integrate all public words before moving down the tree."""
from collections import defaultdict
from dataclasses import dataclass

import torch


POLICIES=("commit","uniform","global-commit","global-fresh","cross","same","moved","wrong-node","reverse","agreement","pose","panels")
GAMMAS=(1.,4.,16.,64.)
PRIORS=(1/14,.5,.8)


def sum_to_parent(x,up,n,prob):
    shape=(1,len(up))+((1,)*(x.ndim-2))
    return x.new_zeros((len(x),n,*x.shape[2:])).index_add_(1,up,x*prob.reshape(shape))


def build_levels(density,tree):
    # [query, cell, view, pose, independent gallery]. Normalize each observation over gallery images.
    xs=[density.permute(0,4,1,2,3).contiguous()]
    count=[torch.bincount(a,minlength=k).float() for a,k in zip(tree.ids,tree.sizes,strict=True)]
    priors=[]
    for j,up in enumerate(tree.parents):
        pr=count[j]/count[j+1][up];priors.append(pr)
        xs.append(sum_to_parent(xs[-1],up,tree.sizes[j+1],pr))
    return xs,count,priors


def decisions(xs,tree,priors):
    rates=[];signals=[];identity=0.
    for j,up in enumerate(tree.parents):
        n=tree.sizes[j+1];pr=priors[j]
        ca=xs[j];pa=xs[j+1][:,up]
        ratios=ca/pa.clamp_min(1e-30)
        full=ca[:,:,:,:,0].mean(3)/pa[:,:,:,:,0].mean(3).clamp_min(1e-30)
        a,b=ratios[:,:,:,:,1],ratios[:,:,:,:,2]
        ao,af=a.unbind(3);bo,bf=b.unbind(3)
        q0,q1=ratios[:,:,:,:,0].unbind(3)
        def cross(left,right):
            return .5*sum_to_parent(left*right.clamp_min(1e-30).log()+right*left.clamp_min(1e-30).log(),up,n,pr)
        cv=.5*(cross(ao,bf)+cross(af,bo))
        same=.25*sum_to_parent(ao*ao.clamp_min(1e-30).log()+af*af.clamp_min(1e-30).log()+
                 bo*bo.clamp_min(1e-30).log()+bf*bf.clamp_min(1e-30).log(),up,n,pr)
        error=(same-cv).amin().clamp_max(0).abs()
        identity=max(identity,float(error))
        signals.append({"cross":cv,"same":same,"agreement":cv-same,"pose":cross(q0,q1),"panels":.5*(cross(ao,bo)+cross(af,bf))})
        rates.append(full)
    if identity>2e-5:raise ValueError("Cross-validated information exceeded its optimistic self-information bound")
    return rates,signals,identity


@dataclass(frozen=True,order=True)
class Row:
    depth:int
    whole:float
    policy:str
    gamma:float
    readout:str
    gain:float

    def label(self):return f"{self.depth}/{self.whole:.5f}/{self.policy}/{self.gamma:g}/{self.readout}/{self.gain:g}"

    @classmethod
    def parse(cls,name):
        l,h,p,g,r,a=name.split("/")
        return cls(int(l),float(h),p,float(g),r,float(a))


def grid():
    rows={"base/direct","base/regional"}
    for depth in (1,2,3):
        for h in PRIORS:
            for mode in POLICIES:
                for ga in ((1.,) if mode in ("commit","uniform") else GAMMAS):
                    for readout,choices in (("cell",(0.,.45,1.)),("regional-delta",(-.5,.3,1.)),("regional-cell",(.15,.4,.8))):
                        for gain in choices:rows.add(Row(depth,h,mode,ga,readout,gain).label())
    return sorted(rows)


def weights(prior,score,gamma):
    return (prior.log()+gamma*score).softmax(2)


def apply_path(tree,xs,count,prior,rates,signals,mode,gamma,depths,trace=None):
    b=len(xs[0]);device=xs[0].device
    # Full view, nine small crops, four larger crops; the shuffled control fixes the full view and both scales.
    perm=torch.tensor([0,4,5,6,7,8,9,1,2,3,12,13,10,11],device=device)
    root=weights(prior,signals[-1]["cross"],gamma)
    state=prior.expand(b,1,-1)
    out={}
    for j in reversed(range(min(depths)-1,len(rates))):
        up=tree.parents[j];n=tree.sizes[j+1]
        if mode=="commit":
            mixed=state
        elif mode=="global-commit":
            mixed=root if n==1 else state
        else:
            if mode=="uniform":pol=prior.expand(b,n,-1)
            elif mode=="global-fresh":pol=root.expand(b,n,-1)
            elif mode in ("cross","same","pose","panels","agreement"):pol=weights(prior,signals[j][mode],gamma)
            else:
                cross=signals[j]["cross"]
                if mode=="moved":cross=cross[:,:,perm]
                elif mode=="wrong-node":
                    rng=torch.randperm(n,generator=torch.Generator().manual_seed(46537+j)).to(device)
                    cross=cross[:,rng]
                elif mode=="reverse":cross=-cross
                else:raise ValueError(mode)
                pol=weights(prior,cross,gamma)
            if trace is not None and mode=="cross":
                mass=state.sum(2)*count[j+1][None]/count[-1]
                kl=(pol*(pol.clamp_min(1e-20).log()-root.clamp_min(1e-20).log())).sum(2)
                trace[j+1]=torch.stack(((mass*kl).sum(1),(mass*pol[:,:,0]).sum(1),
                    (mass*(pol.argmax(2)!=root.argmax(2))).sum(1)),1).cpu()
            mixed=state.sum(2,keepdim=True)*pol
        state=mixed[:,up]*rates[j]
        if j+1 in depths:out[j+1]=state.sum(2).clamp_min(1e-30).log()
    return out


def outputs(tree,data,lex,reg,rows,diagnostic=None):
    if "base/direct" in rows:yield "base/direct",lex,"direct"
    if "base/regional" in rows:yield "base/regional",reg,"regional"
    sets=defaultdict(list)
    for k in rows:
        if not k.startswith("base/"):
            r=Row.parse(k);sets[(r.whole,r.policy,r.gamma)].append(r)
    xs,count,priors=build_levels(data,tree);rates,signals,err=decisions(xs,tree,priors)
    native=xs[0][:,:,0,0,0]
    for (h,mode,gamma),grp in sets.items():
        p=data.new_full((1,1,14),(1-h)/13);p[:,:,0]=h
        depths={x.depth for x in grp}
        trace={} if diagnostic is not None and (h,mode,gamma)==diagnostic["policy"] else None
        act=apply_path(tree,xs,count,p,rates,signals,mode,gamma,depths,trace)
        if trace is not None:diagnostic["trace"]=trace
        for depth in sorted(depths):
            j=depth-1;ids=tree.ids[j]
            v=xs[j][:,:,:,:,0].mean(3)
            frozen=(v*p).sum(2).clamp_min(1e-30).log()
            if mode=="commit" and (act[depth]-frozen).abs().max()>5e-4:raise ValueError("One fixed view failed to telescope")
            delta=(act[depth]-frozen)[:,ids]
            cell=act[depth][:,ids]
            # Individual photo detail uses the same direct full-word score at all settings.
            whole=xs[j][:,:,0,0,0].clamp_min(1e-30).log()[:,ids]
            local=lex-whole
            for r in (x for x in grp if x.depth==depth):
                if r.readout=="cell":v=cell+r.gain*local
                elif r.readout=="regional-delta":v=reg+r.gain*delta
                else:
                    m=cell-cell.mean(1,keepdim=True)
                    scale=reg.std(1,keepdim=True,unbiased=False)/m.std(1,keepdim=True,unbiased=False).clamp_min(1e-8)
                    v=reg+r.gain*scale*m
                yield r.label(),v,("direct" if r.readout=="cell" else "regional")
