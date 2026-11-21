"""Word-level evidence on a common orbit of visual layouts across query poses."""

from math import log
from types import SimpleNamespace

import torch

from .cross_diffusion_map import Config,Index
from .encoding import save_tensor
from .pose_layout_cache import FAMILY
from .shared_concept_likelihood import fit_prior


def target_index(w,split,images,words,scale,prior):
    path=w/"cache"/FAMILY/(split+"-gallery-"+prior+".pt")
    if path.exists():
        return torch.load(path,weights_only=True,mmap=True,map_location="cpu").cuda()
    p=(scale*(images@words.T)).softmax(1)
    if prior=="mle":
        z=fit_prior(p)[0]["mle"]
        p*=z
        p/=p.sum(1,keepdim=True).clamp_min(1e-30)
    index=Index(SimpleNamespace(p=images),rgb_only=True)
    cfg=Config(neighbors=16,bandwidth=.36,iterations=2,mode="rgb")
    b=torch.empty_like(p)
    for start in range(0,len(words),384):
        b[:,start:start+384]=index.score(cfg,p[:,start:start+384].T.contiguous()).T
    b/=b.sum(1,keepdim=True)
    save_tensor(b.cpu(),path)
    return b


def code(z,poses,pool,mode,power):
    h=z[:,:poses]
    true=h[:,:,0]
    if pool=="native":
        base=z[:,0,0]
    elif pool=="arithmetic":
        base=true.logsumexp(1)-log(poses)
    elif pool=="geometric":
        base=true.mean(1)
    else:raise ValueError(pool)
    if mode=="original": raw=base
    elif mode=="unbounded":
        null=h[:,:,1:].logsumexp(2)-log(h.size(2)-1)
        raw=base+power*(true-null).mean(1)
    else:
        r=h.softmax(2)*h.size(2)
        if mode=="mean": all_mass=r.mean(1)
        elif mode=="squared": all_mass=r.square().mean(1)
        elif mode in ("cross","misaligned","phase_shift"):
            if mode=="misaligned":
                # Keep the native response, but assign different counterfactual
                # worlds to every other pose. Marginal distributions are unchanged.
                shift=torch.tensor([0,2,3,1],device=h.device)
                r=r.clone()
                r[:,1::2]=r[:,1::2].index_select(2,shift)
            elif mode=="phase_shift":
                r=r.clone()
                r[:,1::2]=r[:,1::2].roll(1,2)
            all_mass=(r.sum(1).square()-r.square().sum(1)).clamp_min(0)
            if poses==1:all_mass=r[:,0]
        else:raise ValueError(mode)
        e=all_mass*(h.size(2)/all_mass.sum(1,keepdim=True).clamp_min(1e-30))
        raw=base+power*e[:,0].clamp_min(1e-12).log()
    return (raw-raw.amax(1,keepdim=True)).exp()


def check_worlds(z):
    h=z[:3,:,:,:127].double()
    r=h.softmax(2)*4
    pair=(r.sum(1).square()-r.square().sum(1)).clamp_min(0)
    direct=sum(r[:,a]*r[:,b] for a in range(r.size(1)) for b in range(r.size(1)) if a!=b)
    algebra=float((direct-pair).abs().max()/direct.abs().max().clamp_min(1e-30))
    e=4*pair/pair.sum(1,keepdim=True).clamp_min(1e-30)
    unity=float((e.mean(1)-1).abs().max())
    if max(algebra,unity)>2e-10:raise ValueError("Finite-orbit concordance check failed")
    return {"explicit_pose_pair_error":algebra,"all_layout_designations_mean_one_error":unity}
