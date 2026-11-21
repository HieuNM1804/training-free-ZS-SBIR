"""Render a gallery candidate subject to the patch moments of one independent query."""

import math
import torch

from .reciprocal_reassembly import encode


def helmert(n,device):
    h=torch.zeros(n,n-1,device=device,dtype=torch.float64)
    for j in range(1,n):
        scale=(j*(j+1))**-.5
        h[:j,j-1]=scale
        h[j,j-1]=-j*scale
    return h


def transport(source,photo,h):
    x=source.double()
    y=photo.double()
    mu=x.mean(1,keepdim=True)
    xv=h.T@(x-mu)
    yv=h.T@(y-y.mean(1,keepdim=True))
    u,s,vh=torch.linalg.svd(yv@xv.transpose(1,2),full_matrices=False)
    z=mu+h@((u@vh)@xv)
    common=s.sum(1)/(xv.flatten(1).norm(dim=1)*yv.flatten(1).norm(dim=1)).clamp_min(1e-30)
    return z,common.float()


def check_moments(x,y,z,h):
    x,y,z=x[:2].double(),y[:2].double(),z[:2].double()
    c=x-x.mean(1,keepdim=True)
    t=z-z.mean(1,keepdim=True)
    cov=c.transpose(1,2)@c
    cap=(t.transpose(1,2)@t-cov).abs().amax()/cov.abs().amax().clamp_min(1e-20)
    mean=(z.mean(1)-x.mean(1)).abs().amax()
    self_z,_=transport(x,x,h)
    error=(self_z-x).norm()/x.norm().clamp_min(1e-20)
    moved,_=transport(x.roll(5,1),y,h)
    # A rank-deficient cross-covariance can admit several optimal embeddings.
    # The minimized objective is invariant even if a numerical solver selects different minimizers.
    objectives=((moved-y).square().sum((1,2))-(z-y).square().sum((1,2))).abs()
    optim=(objectives/(z-y).square().sum((1,2)).clamp_min(1e-18)).max()
    same=(moved-z).norm()/z.norm().clamp_min(1e-18)
    values={"covariance_relative_max":float(cap),"mean_absolute_max":float(mean),
            "self_relative_error":float(error),"reordering_optimum_relative_error":float(optim),
            "reordering_representation_relative_change":float(same)}
    if max(values[k] for k in ("covariance_relative_max","mean_absolute_max","self_relative_error","reordering_optimum_relative_error"))>3e-6:
        raise ValueError(f"Observation geometry changed {values}")
    return values


def diagonal(source,photo):
    v,m=torch.var_mean(photo.float(),dim=1,correction=0,keepdim=True)
    vx,mx=torch.var_mean(source.float(),dim=1,correction=0,keepdim=True)
    return mx+(photo-m)*(vx.sqrt()/v.sqrt().clamp_min(1e-9))


def read_layouts(v,patch,permutations):
    return [encode(v,patch.to(v.conv1.weight.dtype))]+[
        encode(v,patch[:,g].to(v.conv1.weight.dtype)) for g in permutations]


def affinities(q,renders,src,words,scale,heads):
    all_q=(q@src.T*scale).log_softmax(-1)
    small_q=(q@src[heads].T*scale).log_softmax(-1)
    out={"all":[],"compact":[],"dot":[]}
    for z in renders:
        all_obs=(z@words.T*scale).log_softmax(-1)
        sm_obs=(z@words[heads].T*scale).log_softmax(-1)
        out["all"].append((all_q+all_obs).logsumexp(-1))
        out["compact"].append((small_q+sm_obs).logsumexp(-1))
        out["dot"].append((q*z).sum(-1)*scale)
    return {key:torch.stack(arr,1) for key,arr in out.items()}


def changes(native,observed):
    result={}
    for key,initial in native.items():
        after=observed[key]
        native_null=initial[:,1:].logsumexp(1)-math.log(initial.size(1)-1)
        new_null=after[:,1:].logsumexp(1)-math.log(after.size(1)-1)
        delta=after[:,0]-initial[:,0]
        result[key+"-response"]=after[:,0]
        result[key+"-change"]=delta
        result[key+"-layout"]=delta-(new_null-native_null)
        result[key+"-native"]=initial[:,0]
    return result
