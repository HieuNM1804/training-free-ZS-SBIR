"""Measure the tail of each rival's best apparent window on intact, disjoint gallery photos."""
from collections import defaultdict
import math

import torch


STAT=("whole","mean","maximum","gain","mean_gain","single_pose")
SPECS=("whole","mean","mean_gain","maximum","gain","gain-centered","gain-unconditional","gain-mixed","gain-pooled","gain-raw","whole-raw","single_pose")


def margins(x,g):
    diff=x-x[:,:,:,g:g+1]
    whole=diff[:,0].mean(1)
    local=diff[:,1:]
    gains=(local-diff[:,:1]).amin(2)
    return {"whole":whole,"mean":local.mean((1,2)),"maximum":local.amin(2).amax(1),"gain":gains.amax(1),
            "mean_gain":gains.mean(1),"single_pose":(local[:,:,0]-diff[:,0:1,0]).amax(1)},gains


def shuffled_max(gain,order,seed):
    n,h=order.shape
    order=order.T
    # Each pair has its own order by whole-image margin. Both poses of a window travel together.
    vals=gain.permute(2,0,1)
    sorted_vals=vals.gather(1,order[:,:,None].expand(-1,-1,vals.size(2)))
    changed=torch.empty_like(sorted_vals)
    generator=torch.Generator().manual_seed(seed)
    for seg in torch.tensor_split(torch.arange(n),min(4,n)):
        start=int(seg[0]);stop=int(seg[-1])+1
        for view in range(vals.size(2)):
            perm=torch.randperm(len(seg),generator=generator).to(vals.device)+start
            changed[:,start:stop,view]=sorted_vals[:,perm,view]
    return changed.amax(2).gather(1,order.argsort(1)).T


def tail(sorted_ref,query):
    # Each row is a rival. Ties count as at least as extreme.
    n=sorted_ref.size(1)
    below=torch.searchsorted(sorted_ref.contiguous(),query.T.contiguous(),right=False)
    return ((1+n-below).float()/(n+1)).log().T


def calibrators(samples,g,seed):
    stats,all_gain=margins(samples,g)
    n=len(samples)
    rank=stats["whole"].argsort(dim=0,stable=True)
    stats["gain-mixed"]=shuffled_max(all_gain,rank,seed)
    sorted_whole=stats["whole"].T.sort(1).values
    group=rank.T
    nbin=min(4,n)
    intervals=[(int(q[0]),int(q[-1])+1) for q in torch.tensor_split(torch.arange(n),nbin)]
    out={}
    for k,x in stats.items():
        z=x.T
        grouped=z.gather(1,group)
        out[k]=[grouped[:,lo:hi].sort(1).values for lo,hi in intervals]
        if k in ("gain","whole"):out[k+"/uncond"]=z.sort(1).values
    return sorted_whole,intervals,out


def conditioned(query,rank,limits,tables):
    value=torch.zeros_like(query)
    for (lo,hi),z in zip(limits,tables,strict=True):
        v=tail(z,query)
        value=torch.where((rank>=lo)&(rank<hi),v,value)
    return value


class Calibration:
    def __init__(self,photos,groups,donors):
        self.tables=[]
        self.counts=[]
        groups=groups[donors];k=photos.size(-1)
        for g in range(k):
            relevant=torch.where(groups==g)[0];n=len(relevant)
            self.counts.append(n)
            if n==0:
                self.tables.append(None);continue
            rng=torch.Generator().manual_seed(62811+g)
            unrelated=torch.randperm(len(donors),generator=rng)[:n].to(donors.device)
            real=calibrators(photos[donors[relevant]],g,71117+g)
            fake=calibrators(photos[donors[unrelated]],g,71117+g)
            self.tables.append((real,fake))

    def score(self,x,prior):
        b=len(x);k=x.size(-1)
        # Matrix rows are null groups g, columns are rival groups h.
        logs={v:x.new_zeros((b,k,k)) for v in SPECS}
        for g,item in enumerate(self.tables):
            if item is None:continue
            (ref,limits,tables),(fref,flimits,ftables)=item
            d,_=margins(x,g)
            lo=torch.searchsorted(ref.contiguous(),d["whole"].T.contiguous()).clamp_max(ref.size(1)-1).T
            flo=torch.searchsorted(fref.contiguous(),d["whole"].T.contiguous()).clamp_max(fref.size(1)-1).T
            for name in ("mean","mean_gain","maximum","gain","single_pose"):
                logs[name][:,g]=conditioned(d[name],lo,limits,tables[name])
            logs["whole"][:,g]=tail(tables["whole/uncond"],d["whole"])
            # The expected log of a discrete exchangeable rank depends on how many donors the bin contains.
            center=torch.zeros_like(d["gain"])
            for begin,stop in limits:
                size=stop-begin+1
                expected=math.lgamma(size+1)/size-math.log(size)
                center=torch.where((lo>=begin)&(lo<stop),expected,center)
            logs["gain-centered"][:,g]=logs["gain"][:,g]-center
            logs["gain-unconditional"][:,g]=tail(tables["gain/uncond"],d["gain"])
            logs["gain-mixed"][:,g]=conditioned(d["gain"],lo,limits,tables["gain-mixed"])
            logs["gain-pooled"][:,g]=conditioned(d["gain"],flo,flimits,ftables["gain"])
            logs["gain-raw"][:,g]=-d["gain"]
            logs["whole-raw"][:,g]=-d["whole"]
        whole=x[:,0].mean(1);all_views=torch.logsumexp(x.mean(2),1)-math.log(x.size(1))
        results={"whole":whole,"multi":.5*whole+.5*all_views}
        rivals={"equal":torch.ones_like(whole)/k,"query":(whole+prior.clamp_min(1e-12).log()).softmax(1)}
        for name,v in logs.items():
            diff=v-v.transpose(1,2)
            for who,weight in rivals.items():
                results[name+"/"+who]=(diff*weight[:,None]).sum(2)
        return results,logs


def assemble_level(original,labels,parent,level):
    if level==2:return original,labels
    # Densities rather than log scores must be merged when aggregating fine photo groups.
    original=original.float()
    ct=torch.bincount(labels,minlength=len(parent)).float()
    nc=int(parent.max())+1
    count=ct.new_zeros(nc).index_add_(0,parent,ct)
    data=original.exp()
    merged=data.new_zeros((*data.shape[:-1],nc)).index_add_(-1,parent,data*(ct/count[parent]))
    return merged.clamp_min(1e-30).log(),parent[labels]
