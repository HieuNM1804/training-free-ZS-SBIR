"""Measure how an independent sketch reading moves score differences between rival gallery groups."""

import torch

from .crossdomain_benchmark import digest
from .encoding import save_tensor
from .reciprocal_reassembly_study import layout_groups
from .regional_intervention_study import index as compact_index


def group_mean(data,groups):
    total=data.new_zeros((int(groups.max())+1,data.size(1)))
    for start in range(0,len(groups),256):
        ix=slice(start,start+256)
        total.index_add_(0,groups[ix],data[ix].to(total))
    return total/groups.bincount()[:,None]


def group_inputs(w,s,photo,pi,pp,target,scale,word_sha):
    ident=digest({"gallery":digest([str(pp[i]) for i in pi.tolist()]),"words":word_sha,"model":"regional-64-rgb16-groups64-v1"})[:16]
    file=w/"cache/redraw-rank-stability"/(s+"-"+ident+".pt")
    if file.is_file():
        d=torch.load(file,weights_only=True,mmap=True)
        return {k:v.cuda() for k,v in d.items()}
    ids,_=layout_groups(photo)
    heads,compact=compact_index(w,s,photo,target,scale,ident)
    profiles=torch.load(w/"cache/pose-conditional-layout"/(s+"-gallery-uniform.pt"),map_location="cpu",weights_only=True,mmap=True)
    if profiles.size()!=(len(pi),len(target)):raise ValueError("Unrelated gallery word cache")
    full=photo.new_zeros((int(ids.max())+1,len(target)))
    for start in range(0,len(ids),256):
        sl=slice(start,start+256); full.index_add_(0,ids[sl],profiles[sl].cuda())
    full/=ids.bincount()[:,None]
    result={"groups":ids,"heads":heads,"compact":compact,"group_compact":group_mean(compact,ids),"full":full}
    save_tensor({k:v.cpu() for k,v in result.items()},file)
    return result


def logmatch(z,p):
    z=z-z.amax(-1,keepdim=True)
    return (z.exp()@p.T).clamp_min(1e-30).log()


def read_group(query,views,source,scale,bank):
    h=bank["heads"]
    native=logmatch(query@source.T*scale,bank["full"])
    local=views@source[h].T*scale
    a=local[:,:9].logsumexp(1)-local.new_tensor(9.).log()
    b=local[:,9:].logsumexp(1)-local.new_tensor(4.).log()
    zoom=torch.logaddexp(a,b)-local.new_tensor(2.).log()
    nominal=.6*native+.4*logmatch(zoom,bank["group_compact"])
    z=views.flatten(0,1)@source.T*scale
    high=logmatch(z,bank["full"]).reshape(len(query),13,-1)
    low=logmatch(local,bank["group_compact"])
    observed=.6*high+.4*low
    weights=observed.new_tensor([1/18]*9+[1/8]*4)
    # Each view gives one shared measurement of every group. Center each
    # measurement to remove an irrelevant query-dependent additive offset.
    observed=observed-observed.mean(-1,keepdim=True)
    centered=observed-(weights[None,:,None]*observed).sum(1,keepdim=True)
    covariance=centered.transpose(1,2)@(centered*weights[None,:,None])/(1-weights.square().sum())
    return nominal,covariance,zoom
