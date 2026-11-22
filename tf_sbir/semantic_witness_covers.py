"""Exact marginal likelihood of covering a sketch's public clues with witnesses from a gallery population."""

import math
import torch


FAMILY="semantic-witness-covers"
SIZES=(6,8)
THRESHOLDS=(.90,.97)
DRAW_COUNTS=(1,2,4)
MODES=("cover","excess","marginal","independent","one","mean")


def subset_tables(k,device):
    ids=torch.arange(1<<k,device=device)
    bit=torch.arange(k,device=device)
    bits=(ids[:,None]>>bit[None])&1
    sign=1-2*(bits.sum(1)%2)
    return ids,bits.double(),sign.double(),1<<bit


def subset_cdf(masses,k):
    out=masses.double()
    n=out.shape[-1]
    if n!=1<<k:raise ValueError("Invalid Boolean observation histogram")
    for j in range(k):
        shape=out.shape
        x=out.reshape(*shape[:-1],-1,2,1<<j)
        out=torch.stack((x[...,0,:],x[...,0,:]+x[...,1,:]),-2).reshape(shape)
    return out


def scores(hist,k):
    _,bits,sign,_=subset_tables(k,hist.device)
    hist=hist.double()
    probability=hist/hist.sum(-1,keepdim=True).clamp_min(1)
    cdf=subset_cdf(probability,k)
    marg=probability@bits
    empty=cdf.flip((-1,))
    result={}
    # "one" sums the exact probability of seeing every missing set in a single draw.
    one=cdf[...,0]*0+probability[...,-1]
    for r in DRAW_COUNTS:
        independent=(1-(1-marg).pow(r)).prod(-1)
        cover=(empty.pow(r)*sign).sum(-1)
        if float(cover.min())< -1e-8 or float(cover.max())>1.0000001:raise ValueError("Invalid exact witness union")
        cover=cover.clamp(0,1)
        # The root gives a per-clue probability on [0,1]; zero remains exactly zero.
        full=cover.pow(1/k)
        null=independent.clamp(0,1).pow(1/k)
        result[r]={"cover":full,"excess":full-null,"marginal":null,"independent":independent,
                   "one":one.clamp(0,1).pow(1/k),"mean":marg.mean(-1)}
    return result


def audit(device):
    x=torch.tensor([[.5,0.,0.,.5],[0.,.5,.5,0.]],device=device,dtype=torch.float64)
    c=subset_cdf(x,2)
    _,bits,sign,_=subset_tables(2,device)
    different=(c.flip((-1,)).square()*sign).sum(1)
    err=float((different-x.new_tensor([.75,.5])).abs().max())
    probs=torch.tensor([.2,.45,.8],device=device,dtype=torch.float64)
    _,b,s,_=subset_tables(3,device)
    hist=(probs**b*(1-probs)**(1-b)).prod(1)
    actual=subset_cdf(hist,3).flip((-1,)).pow(4)@s
    ref=(1-(1-probs).pow(4)).prod()
    independent_error=float((actual-ref).abs())
    if max(err,independent_error)>1e-10:raise ValueError("Exact cover probability differs from enumeration")
    return {"two_complementary_vs_redundant_words":different.tolist(),"same_marginals_identity_error":err,
            "independent_clue_identity_error":independent_error}


def word_margins(photo,query,src,tgt,k,thresholds):
    names=(query@src.T).topk(k,1).indices
    text=tgt[names]
    b=len(query)
    raw=(photo@text.flatten(0,1).T).reshape(len(photo),b,k).permute(1,0,2).contiguous()
    masks={}
    for level in thresholds:
        cutoff=raw.kthvalue(max(1,math.ceil(level*len(photo))),dim=1,keepdim=True).values
        masks[level]=raw>cutoff
    return masks


def group_counts(photo_bits,assignments,groups):
    b,n,k=photo_bits.shape
    _,_,_,weights=subset_tables(k,photo_bits.device)
    masks=(photo_bits.long()*weights).sum(-1)
    states=1<<k
    # Group membership is a target-only quantity, fit before seeing the sketch.
    idx=assignments[None,:]*states+masks
    counts=torch.zeros((b,groups*states),device=masks.device,dtype=torch.int32)
    counts.scatter_add_(1,idx,torch.ones_like(idx,dtype=counts.dtype))
    return counts.reshape(b,groups,states)
