"""Minimal posterior changes compatible with two generic renderings of a domain description."""

import argparse
from pathlib import Path

import open_clip
import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json, checksum, read
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT, runtime
from .encoding import load_frozen_clip, save_tensor
from .foundation import load_words
from .semantic import read_words


FAMILY = "depiction-posterior-bounds"
MODES = ("box","matched_kl","shared_mixture","mean","shuffled_width","drawing","sketch")


def kl(p, log_prior):
    return (p*(p.clamp_min(1e-35).log()-log_prior)).sum(-1,keepdim=True).clamp_min(0.)


def projection(prior, lower, upper):
    lp=prior.clamp_min(1e-35).log()
    lo=(lower.clamp_min(1e-35).log()-lp).amin(-1,keepdim=True)
    hi=(upper.clamp_min(1e-35).log()-lp).amax(-1,keepdim=True)
    for _ in range(22):
        mid=(lo+hi)*.5
        mass=(prior*mid.exp()).clamp(min=lower,max=upper).sum(-1,keepdim=True)
        small=mass<1
        lo,hi=torch.where(small,mid,lo),torch.where(small,hi,mid)
    p=(prior*((lo+hi)*.5).exp()).clamp(min=lower,max=upper)
    return p/p.sum(-1,keepdim=True).clamp_min(1e-35)


def on_segment(first, second, log_prior):
    diff=second-first
    lo=torch.zeros_like(first[:,:1])
    hi=torch.ones_like(lo)
    for _ in range(17):
        mid=(hi+lo)*.5
        row=first+mid*diff
        sign=(diff*(row.clamp_min(1e-35).log()-log_prior)).sum(-1,keepdim=True)
        lo,hi=torch.where(sign<0,mid,lo),torch.where(sign<0,hi,mid)
    return first+(.5*(hi+lo))*diff


def equal_information(mean, log_prior, budget):
    gap=mean.clamp_min(1e-35).log()-log_prior
    lo=torch.zeros_like(budget)
    hi=torch.ones_like(budget)
    for _ in range(17):
        mid=(hi+lo)*.5
        test=(log_prior+mid*gap).softmax(-1)
        small=kl(test,log_prior)<budget
        lo,hi=torch.where(small,mid,lo),torch.where(small,hi,mid)
    return (log_prior+(.5*(hi+lo))*gap).softmax(-1)


def rank_permutation(length, device, group_size=32):
    gen=torch.Generator().manual_seed(17621)
    parts=[torch.randperm(min(group_size,length-i),generator=gen)+i for i in range(0,length,group_size)]
    return torch.cat(parts).to(device)


def solve(prior, first, second, modes, permutation=None):
    low=torch.minimum(first,second)
    high=torch.maximum(first,second)
    mid=(first+second)*.5
    logp=prior.clamp_min(1e-35).log()
    pstar=projection(prior,low,high)
    info=kl(pstar,logp)
    benefit=kl(mid,logp)-info
    violation=torch.maximum((low-pstar).amax(-1),(pstar-high).amax(-1)).clamp_min(0)
    checks=[violation.amax(),(-benefit).clamp_min(0).amax()]
    # The original posterior mass that must move before satisfying both bounds
    # is at least the larger of these two one-sided sums.
    deficit=(low-prior).clamp_min(0).sum(-1)
    excess=(prior-high).clamp_min(0).sum(-1)
    tv=.5*(pstar-prior).abs().sum(-1)
    diagnostics=torch.stack((info[:,0],kl(mid,logp)[:,0],tv,
                             .5*(first-second).abs().sum(-1),
                             torch.maximum(deficit,excess)),1)
    output={}
    for mode in modes:
        if mode=="box":
            output[mode]=pstar
        elif mode=="mean":
            output[mode]=mid
        elif mode=="drawing":
            output[mode]=first
        elif mode=="sketch":
            output[mode]=second
        elif mode=="shared_mixture":
            output[mode]=on_segment(first,second,logp)
        elif mode=="matched_kl":
            alternative=equal_information(mid,logp,info)
            matched=(kl(alternative,logp)-info).abs().amax()
            checks.append(matched)
            output[mode]=alternative
        elif mode=="shuffled_width":
            if permutation is None:
                raise ValueError("Missing matched-rank control permutation")
            rank=mid.argsort(dim=-1,descending=True,stable=True)
            donor=rank[:,permutation]
            width=(high-low)*.5/mid.clamp_min(1e-35)
            remapped=torch.empty_like(width).scatter_(1,rank,width.gather(1,donor))
            # These null bounds contain the identical arithmetic midpoint.
            out=projection(prior,(mid*(1-remapped)).clamp_min(0),mid*(1+remapped))
            output[mode]=out
        else:
            raise ValueError(mode)
    return output,diagnostics,torch.stack(checks).max()


def invariants(device):
    r=torch.tensor([[.62,.23,.15]],dtype=torch.float64,device=device)
    lo=torch.tensor([[.31,.24,.45]],dtype=r.dtype,device=device)
    hi=torch.tensor([[.61,.30,.09]],dtype=r.dtype,device=device)
    results,_,err=solve(r,lo,hi,MODES,rank_permutation(3,device))
    # Enumerate feasible simplex points to bound the forward KL from above.
    pos=torch.arange(501,device=device,dtype=r.dtype)/500
    x,y=torch.meshgrid(pos,pos,indexing="ij")
    members=torch.stack((x,y,1-x-y),-1).reshape(-1,3)
    lower,upper=torch.minimum(lo,hi),torch.maximum(lo,hi)
    keep=((members>=lower-1e-12)&(members<=upper+1e-12)).all(1)
    coarse=kl(members[keep],r.log()).min()
    numeric=kl(results["box"],r.log()).item()
    interior=(lo+hi)*.5
    equiv=projection(interior,lower,upper)
    singleton=projection(r,lo,lo)
    summary={"max_invariant_error":float(err),"projection_KL":numeric,"enumerated_feasible_min_KL":float(coarse),
             "interior_identity_error":float((interior-equiv).abs().max()),
             "singleton_error":float((lo-singleton).abs().max())}
    if numeric>coarse+1e-6 or max(summary[k] for k in ("max_invariant_error","interior_identity_error","singleton_error"))>1e-5:
        raise ValueError(f"KL-box audit failed {summary}")
    return summary


@torch.inference_mode()
def cache(w):
    cfg=runtime(w,"sketchy")
    names=read_words(cfg.vocabulary)
    if len(names)!=20931:
        raise ValueError("Different public dictionary")
    words=("drawing","sketch","photo")
    templates=[[f"a {domain} of {article} {{}}." for article in ("a","the")] for domain in words]
    spec={"checkpoint":CHECKPOINT,"vocabulary":checksum(cfg.vocabulary),"templates":templates,
          "contrasts":"two source domain names crossed with both of the existing articles"}
    sig=digest(spec)
    file=w/"cache"/FAMILY/"public.pt"
    desc=file.with_suffix(".json")
    checks=invariants(torch.device("cuda"))
    if file.is_file() and desc.is_file() and read(desc)["signature"]==sig:
        return
    model=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda"))
    tok=open_clip.get_tokenizer("ViT-B-32")
    banks=[[] for _ in words]
    legacy=[]
    for start in range(0,len(names),384):
        ls=names[start:start+384]
        embeddings=[]
        for pair in templates:
            emb=[]
            for tmpl in pair:
                tokens=tok([tmpl.format(x) for x in ls]).cuda()
                emb.append(F.normalize(model.encode_text(tokens).float()))
            embeddings.append(emb)
        for b,x in zip(banks,embeddings,strict=True):
            b.append(F.normalize(x[0]+x[1]).cpu())
        legacy.append(F.normalize(embeddings[0][0]+embeddings[1][1]).cpu())
    drawing,sketch,photo=(torch.cat(v) for v in banks)
    source,target=load_words(cfg,names)
    old=torch.cat(legacy)
    err=max(float((1-(old*source).sum(1)).abs().max()), float((1-(photo*target).sum(1)).abs().max()))
    if err>.002:
        raise ValueError("Existing public template bank changed")
    save_tensor({"drawing":drawing,"sketch":sketch,"photo":photo},file)
    atomic_json(desc,spec|{"signature":sig,"native_cosine_error":err,"numeric_audit":checks})
    print("public prompts done",len(names),checks,"native_error",err,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    cache(ap.parse_args().workspace)
