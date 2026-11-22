"""Compare residual dependence of named concepts under separately permutable visual observations."""

import math
import torch
from torch.nn import functional as F


ARMS = ("interaction","cosine","without_reference","full_residual","additive","raw_tau","pearson",
        "word_geometry","mean","dispersion","marginals","permuted","permuted_cosine","without_projection")
SIZES=(8,16,32)
GAINS=(.2,.7,2.,6.)


def offdiag(matrix):
    k=matrix.shape[-1]
    pairs=torch.triu_indices(k,k,offset=1,device=matrix.device)
    return matrix[...,pairs[0],pairs[1]]


def decompose(matrix):
    k=matrix.shape[-1]
    row=(matrix.sum(-1)-matrix.diagonal(dim1=-2,dim2=-1))/(k-1)
    mu=row.mean(-1,keepdim=True)
    h=(row-mu)*((k-1)/(k-2))
    additive=mu.unsqueeze(-1)+h.unsqueeze(-1)+h.unsqueeze(-2)
    return matrix-additive,additive


def sign_reference(text,native,project=True):
    if project:
        text=text-(text*native.unsqueeze(-2)).sum(-1,keepdim=True)*native.unsqueeze(-2)
    text=F.normalize(text,dim=-1,eps=1e-8)
    gram=(text@text.transpose(-1,-2)).clamp(-1,1)
    return (2/math.pi)*gram.asin()


def permute_views(values,names,offset,key):
    # A shared permutation would retain all joint relationships. Use a different
    # deterministic permutation for each name, identical for its two orientations.
    n=values.shape[-3]
    ids=torch.arange(n,device=names.device)
    word=names.long().unsqueeze(-1)+key+31*offset
    hashed=((word*2654435761+ids*2246822519).remainder(2147483647)*1274126177).remainder(2147483647)
    hashed=((hashed ^ (hashed>>12))*1597334677).remainder(2147483647)
    order=hashed.argsort(-1).transpose(-1,-2)
    return values.gather(-3,order.unsqueeze(-2).expand_as(values))


def moments(values,visual,include_pearson=False):
    # Inputs are (...,regions,poses,names). Treat each scale and pose as a separate experiment.
    n=values.shape[-3]
    pairs=torch.triu_indices(n,n,offset=1,device=values.device)
    diff=values[...,pairs[0],:,:]-values[...,pairs[1],:,:]
    signs=diff.sign().transpose(-2,-3)
    tau=(signs.transpose(-1,-2)@signs).mean(-3)/len(pairs[0])
    # When an actual crop difference vanishes, its rotated counterpart does too.
    delta=(visual[...,pairs[0],:,:]-visual[...,pairs[1],:,:]).square().sum(-1)
    active=(delta>1e-12).float().mean((-1,-2))
    pearson=None
    if include_pearson:
        z=F.normalize((values-values.mean(-3,keepdim=True)).transpose(-2,-3),dim=-2,eps=1e-8)
        pearson=(z.transpose(-1,-2)@z).mean(-3)
    return tau,active,pearson


def descriptors(views,native,text,names,side):
    # Broadcast-free path for either N image/name-set pairs or a query repeated against many photos.
    native=F.normalize(native.float(),dim=-1)
    views=views.float()
    original=(views.flatten(-3,-2)@text.transpose(-1,-2)).unflatten(-2,(13,2))
    radial=(views*native[:,None,None,:]).sum(-1,keepdim=True)
    tangential=views-radial*native[:,None,None,:]
    projected=original-radial*(native[:,None,:]*text).sum(-1)[:,None,None,:]
    tau=raw=shuffled=pearson=active=rawactive=0
    sorted_marginals=[]
    means=[]
    spread=[]
    for left,right in ((0,9),(9,13)):
        sl=slice(left,right)
        vals=projected[:,sl]
        subset=tangential[:,sl]
        estimate,prob,pc=moments(vals,subset,True)
        raw_tau,raw_prob,_=moments(original[:,sl],views[:,sl])
        perm=permute_views(vals,names,left,311 if side=="source" else 971)
        null,_,_=moments(perm,subset)
        tau=tau+.5*estimate
        active=active+.5*prob
        shuffled=shuffled+.5*null
        raw=raw+.5*raw_tau
        rawactive=rawactive+.5*raw_prob
        pearson=pearson+.5*pc
        ordered=original[:,sl].sort(dim=1).values.mean(2)
        sorted_marginals.append(ordered/(right-left)**.5)
        means.append(original[:,sl].mean((1,2)))
        spread.append(original[:,sl].std(1,unbiased=False).mean(1))
    reference=sign_reference(text,native)*active[:,None,None]
    residual,additive=decompose(tau-reference)
    centered,_=decompose(tau)
    shuffled_residual,_=decompose(shuffled-reference)
    raw_residual,_=decompose(raw-sign_reference(text,native,False)*rawactive[:,None,None])
    def centered_flat(x):
        return (x-x.mean(-1,keepdim=True)).flatten(1)
    return {"interaction":offdiag(residual),"cosine":offdiag(residual),"without_reference":offdiag(centered),
            "full_residual":offdiag(tau-reference),"additive":offdiag(additive),"raw_tau":offdiag(tau),
            "pearson":offdiag(pearson),"word_geometry":offdiag(reference),
            "mean":centered_flat(torch.stack(means,1)),"dispersion":centered_flat(torch.stack(spread,1)),
            "marginals":centered_flat(torch.cat(sorted_marginals,1)),"permuted":offdiag(shuffled_residual),
            "permuted_cosine":offdiag(shuffled_residual),"without_projection":offdiag(raw_residual)}


def match(q,p):
    return {k:((q[k]*p[k]).mean(-1) if k in ("interaction","permuted","without_reference","without_projection")
               else F.cosine_similarity(q[k],p[k],dim=-1,eps=1e-8)) for k in ARMS}


def checks(device):
    gen=torch.Generator(device=device).manual_seed(61041)
    def r(*shape):
        return torch.randn(shape,device=device,generator=gen)
    z=F.normalize(r(3,512),dim=-1)
    views=F.normalize(r(3,13,2,512),dim=-1)
    name=F.normalize(r(3,16,512),dim=-1)
    ids=torch.arange(16,device=device).expand(3,-1)
    a=descriptors(views,z,name,ids,"source")
    perm=torch.tensor([2,4,6,8,0,1,3,5,7,12,11,9,10],device=device)
    b=descriptors(views[:,perm],z,name,ids,"source")
    immutable=max(float((a[k]-b[k]).abs().max()) for k in ARMS if k not in ("permuted","permuted_cosine"))
    h=r(3,16)
    mat=r(3,16,16)
    mat=mat+mat.transpose(-1,-2)
    d,_=decompose(mat)
    dd,_=decompose(mat+h[:,:,None]+h[:,None,:])
    off=float((offdiag(d)-offdiag(dd)).abs().max())
    row=float((d.sum(-1)-d.diagonal(dim1=-1,dim2=-2)).abs().max())
    meas=r(3,9,2,16)
    shuffled=permute_views(meas,ids,0,311)
    margins=float((meas.sort(1).values-shuffled.sort(1).values).abs().max())
    coef=r(3,1,1,16)
    monotone=meas+coef+.2*meas.pow(3)
    t,_,_=moments(meas,views[:,:9])
    t2,_,_=moments(monotone,views[:,:9])
    inv=float((t-t2).abs().max())
    if immutable>2e-4 or off>1e-5 or row>3e-5 or margins or inv:
        raise ValueError(f"Geometric checks failed: {immutable}, {off}, {row}, {margins}, {inv}")
    return {"permuting_shared_locations":immutable,"adding_arbitrary_single_name_terms":off,
            "residual_offdiag_row_sums":row,"permuted_marginal_distributions":margins,"monotonic_recalibration":inv}
