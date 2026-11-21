"""Convert uncertain pairwise ranks back to the original retrieval score scale."""

from math import sqrt

import torch


def quantile(p,center,width):
    low=(center.amin(1)-10*width)[:,None].expand_as(center)
    high=(center.amax(1)+10*width)[:,None].expand_as(center)
    for _ in range(26):
        mid=(low+high)*.5
        rank=torch.special.ndtr((mid[:,:,None]-center[:,None,:])/width[:,None,None]).mean(2)
        low=torch.where(rank<p,mid,low)
        high=torch.where(rank>=p,mid,high)
    return (high+low)*.5


def field(nominal,cov,size,temperature,pair_fraction,kind,common=None):
    ids=nominal.topk(min(size,nominal.size(1)),dim=1).indices
    rows=torch.arange(len(ids),device=ids.device)[:,None,None]
    small=nominal.gather(1,ids).double()
    c=cov[rows,ids[:,:,None],ids[:,None,:]].double()
    if kind=="fixed_gallery":
        c=common[ids[:,:,None],ids[:,None,:]].double()
    v=c.diagonal(dim1=-2,dim2=-1)
    paired=(v[:,:,None]+v[:,None,:]-2*c).clamp_min(1e-12)
    diag=torch.eye(ids.size(1),dtype=torch.bool,device=ids.device)
    sigma=paired[:,~diag].median(1).values.clamp_min(1e-8).sqrt()
    if kind=="independent":
        measured=(v[:,:,None]+v[:,None,:]).clamp_min(1e-12)
        # Match the reference width to the variance assumed by this ablation.
        sigma=measured[:,~diag].median(1).values.clamp_min(1e-8).sqrt()
    elif kind=="reverse_pairs":
        # Independent of group identities while retaining this query's variance scale.
        measured=paired.roll((1,1),(1,2))
    elif kind=="homogeneous":measured=sigma[:,None,None].square().expand_as(paired)
    else:measured=paired
    width=(pair_fraction*measured+(1-pair_fraction)*sigma[:,None,None].square()).sqrt()*temperature
    delta=small[:,:,None]-small[:,None,:]
    p=torch.special.ndtr(delta/width.clamp_min(1e-12)).mean(2)
    ref_width=(sigma*temperature).clamp_min(1e-8)
    if kind=="rank_only":
        ref=torch.special.ndtr(delta/ref_width[:,None,None]).mean(2)
        correction=(p-ref)*sqrt(2*torch.pi)*ref_width[:,None]
    elif kind=="linear":
        # Mean standardized margin with the common temperature scaled back out.
        relative=delta*ref_width[:,None,None]/width.clamp_min(1e-12)
        correction=(relative-delta).mean(2)
    else:
        correction=quantile(p,small,ref_width)-small
    if not torch.isfinite(correction).all():raise ValueError("Invalid inverse expected rank")
    return ids,correction.float()


def check(nominal,cov,common):
    _,identity=field(nominal[:4],cov[:4],16,1.,1.,"homogeneous",common)
    _,base=field(nominal[:4],cov[:4],16,1.,1.,"paired",common)
    _,shift=field(nominal[:4],cov[:4]+.5,16,1.,1.,"paired",common)
    _,translated=field(nominal[:4]+2,cov[:4],16,1.,1.,"paired",common)
    errors={"equal_variance_identity":float(identity.abs().max()),"common_query_noise_cancels":float((base-shift).abs().max()),
            "common_score_shift_cancels":float((base-translated).abs().max())}
    if max(errors.values())>5e-4:raise ValueError(f"Expected-rank calibration failed {errors}")
    return errors
