"""Compute the four-cell shared-word interaction between two whole images and their own fragmented readings."""

from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .cluster_mine import cluster_assignments
from .concept_benchmark import read
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .encoding import save_tensor
from .gallery_region_panels import CONFIG,Index
from .paired_fragment_inputs import FAMILY,PARTS,read_features
from .paired_fragment_area_control import read_area


SIZES=(256,2048)
ALL_PARTS=(*PARTS,"area_shuffle")
PAIRS=(("grid","grid"),("appearance","appearance"),("appearance","shuffled"),("shuffled","appearance"),("shuffled","shuffled"),
       ("appearance","area_shuffle"),("area_shuffle","appearance"),("area_shuffle","area_shuffle"))
METHODS=("interaction","source","target","additive","whole_minus_parts","parts","whole","cluster_shuffled","cluster_mean","logratio_cos","excess")
GAINS=(-.18,.06,.18,.5)
GRAPH=("native","diffused")


def grouped_permutation(groups):
    gen=torch.Generator().manual_seed(67541)
    perm=torch.arange(len(groups))
    for label in groups.unique():
        ix=torch.where(groups==label)[0]
        if len(ix)>1:
            order=ix[torch.randperm(len(ix),generator=gen)]
            perm[order]=order.roll(1)
    return perm


def catalog(w,s,inputs):
    _,_,_,pp,gs,_,raw,_,target,scale=inputs
    pids=gs[s][0]
    p=raw[pids].cuda()
    stamp=digest({"photos":[str(pp[int(i)]) for i in pids],"model":CHECKPOINT,"order":"mean whole-photo normalized probabilities over all public names v1"})
    directory=w/"cache"/FAMILY/"word-readout"
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/(s+"-catalog-"+stamp[:16]+".pt")
    if path.is_file():return torch.load(path,mmap=True,weights_only=True)
    mass=torch.zeros(len(target),device=p.device,dtype=torch.float64)
    for block in p.split(256):
        prob=(scale*(block@target.T)).softmax(1)
        mass+=prob.double().sum(0)
    prior=mass/len(p)
    order=prior.argsort(descending=True,stable=True).cpu()
    groups=cluster_assignments(F.normalize(p-p.mean(0),dim=1),64).cpu()
    data={"order":order,"prior":prior.cpu(),"groups":groups,"permutation":grouped_permutation(groups)}
    save_tensor(data,path)
    return data


def prob(z,axes,scale):
    return (scale*(z.float()@axes.T)).softmax(-1)


def target_tables(w,s,ds,inputs,cat,n):
    _,_,_,_,gs,_,p,_,tgt,scale=inputs
    pi=gs[s][0]
    # Save probabilities after their operations; returning float32 on CUDA keeps subsequent ratio calculations reliable.
    folder=w/"cache"/FAMILY/"word-readout"
    key=digest({"catalog":cat["order"][:n].tolist(),"parts":ALL_PARTS,"graph_config":str(CONFIG),"gallery":s,"model":CHECKPOINT})
    file=folder/f"{s}-{n}-{key[:12]}.pt"
    if file.is_file():
        cached=torch.load(file,mmap=True,weights_only=True)
        return {k:v.cuda() for k,v in cached.items()}
    fragments,_=read_features(w,ds,"photo",pi)
    matched_areas=read_area(w,ds,"photo",pi)
    rgb=p[pi].cuda();axes=tgt[cat["order"][:n]]
    g=Index(SimpleNamespace(p=rgb),rgb_only=True)
    data={}
    for part in ("whole",*ALL_PARTS):
        blocks=[]
        for start in range(0,len(pi),160):
            sl=slice(start,start+160)
            if part=="whole":b=prob(rgb[sl],axes,scale)
            elif part=="area_shuffle":b=prob(matched_areas[sl].cuda(),axes,scale).mean(1)
            else:b=prob(fragments[sl,PARTS.index(part)].cuda(),axes,scale).mean(1)
            blocks.append(b)
        b=torch.cat(blocks)
        smooth=g.score(CONFIG,b.T.contiguous()).T.contiguous()
        smooth=smooth/smooth.sum(1,keepdim=True).clamp_min(1e-30)
        if (b.sum(1)-1).abs().max()>.001 or (smooth.sum(1)-1).abs().max()>.001:
            raise ValueError("Conditionals are not normalized")
        data[part,"native"]=b.cpu();data[part,"diffused"]=smooth.cpu()
        del blocks,b,smooth
    save_tensor(data,file)
    del g,fragments
    return {k:v.cuda() for k,v in data.items()}


def log_overlap(a,b):
    return (a@b.T).clamp_min(1e-30).log()


def mean_by_group(b,group):
    size=int(group.max())+1
    count=group.bincount(minlength=size).clamp_min(1)
    mean=b.new_zeros((size,b.shape[1])).index_add_(0,group,b)/count[:,None]
    return mean[group]


def score_parts(a1,a0,b1,b0,permuted,averaged):
    s11=log_overlap(a1,b1);s10=log_overlap(a1,b0)
    s01=log_overlap(a0,b1);s00=log_overlap(a0,b0)
    joint=(s11-s01)-(s10-s00)
    def indirect(alt):
        return (s11-s01)-(log_overlap(a1,alt)-log_overlap(a0,alt))
    shuffled=indirect(permuted)
    rq=a1.clamp_min(1e-30).log()-a0.clamp_min(1e-30).log()
    rp=b1.clamp_min(1e-30).log()-b0.clamp_min(1e-30).log()
    cosine=F.normalize(rq-rq.mean(1,keepdim=True))@F.normalize(rp-rp.mean(1,keepdim=True)).T
    return {"interaction":joint,"source":s11-s01,"target":s11-s10,"additive":(s11-s01)+(s11-s10),
            "whole_minus_parts":s11-s00,"parts":s00,"whole":s11,"cluster_shuffled":shuffled,
            "cluster_mean":indirect(averaged),"logratio_cos":cosine,"excess":joint-shuffled}


def algebra():
    g=torch.Generator().manual_seed(7301)
    q0,q1=[torch.rand((3,13),generator=g,dtype=torch.float64).softmax(1) for _ in range(2)]
    p0,p1=[torch.rand((7,13),generator=g,dtype=torch.float64).softmax(1) for _ in range(2)]
    def joint(a,b,c,d):return log_overlap(a,c)-log_overlap(b,c)-log_overlap(a,d)+log_overlap(b,d)
    expected=joint(q1,q0,p1,p0)
    ref=q0[:,None]*p0[None];ref=ref/ref.sum(-1,keepdim=True)
    left=(q1/q0)[:,None];right=(p1/p0)[None,:]
    avg=lambda x:(ref*x).sum(-1)
    derived=(avg(left*right)/(avg(left)*avg(right))).log()
    err=float((derived-expected).abs().max())
    null=max(float(joint(q0,q0,p1,p0).abs().max()),float(joint(q1,q0,p0,p0).abs().max()))
    if max(err,null)>1e-10:raise ValueError("Whole/fragment contrast identity failed")
    return {"weighted_covariance_identity":err,"no_context_on_either_side":null}
