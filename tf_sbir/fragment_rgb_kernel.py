"""Cross the same whole-image and partial-image observations with a visual kernel and no vocabulary."""

import math
import torch


TEMPERATURES=(1.,8.,32.)
READOUTS=("interaction","embedding_difference","nonlinear_only","source","target","additive","fragment_only","parts_suppression")
PARTITION_PAIRS=(("grid","grid"),("appearance","appearance"),("shuffled","shuffled"),("area_shuffle","area_shuffle"),
                 ("appearance","area_shuffle"),("area_shuffle","appearance"))
GAINS=(-.4,.2,.8,1.6)


def four_inputs(query,query_parts,photo,photo_parts):
    q=torch.cat((query[:,None],query_parts),1)
    p=torch.cat((photo[:,None],photo_parts),1)
    # Computing an inner product in FP32 keeps the signed small-temperature difference well defined.
    joint=q.flatten(0,1).float()@p.flatten(0,1).float().T
    return joint.view(len(q),q.size(1),len(p),p.size(1)).permute(0,2,1,3).contiguous()


def logmeanexp(x,dim):
    return x.logsumexp(dim)-math.log(x.size(dim))


def readout(matrix,t):
    whole=matrix[:,:,0,0]*t
    pside=logmeanexp(t*matrix[:,:,0,1:],-1)
    qside=logmeanexp(t*matrix[:,:,1:,0],-1)
    both=logmeanexp(t*matrix[:,:,1:,1:].flatten(2),-1)
    cross=(whole-qside)-(pside-both)
    linear=t*(matrix[:,:,0,0]-matrix[:,:,1:,0].mean(-1)-matrix[:,:,0,1:].mean(-1)+matrix[:,:,1:,1:].mean((2,3)))
    return {"interaction":cross,"embedding_difference":linear,"nonlinear_only":cross-linear,
            "source":whole-qside,"target":whole-pside,"additive":(whole-qside)+(whole-pside),"fragment_only":both,
            "parts_suppression":whole-both}


def audit():
    rng=torch.Generator().manual_seed(10479)
    q=torch.nn.functional.normalize(torch.randn(3,512,generator=rng),dim=-1)
    p=torch.nn.functional.normalize(torch.randn(4,512,generator=rng),dim=-1)
    s=torch.nn.functional.normalize(torch.randn(3,6,512,generator=rng),dim=-1)
    t=torch.nn.functional.normalize(torch.randn(4,6,512,generator=rng),dim=-1)
    dots=four_inputs(q,s,p,t)
    err=0.;null=0.
    for scale in TEMPERATURES:
        result=readout(dots,scale)["embedding_difference"]
        vec=scale*(q-s.mean(1))@(p-t.mean(1)).T
        err=max(err,float((result-vec).abs().max()))
        for m in (four_inputs(q,q[:,None].expand_as(s),p,t),four_inputs(q,s,p,p[:,None].expand_as(t))):
            null=max(null,float(readout(m,scale)["interaction"].abs().max()))
    if max(null,err)>2e-5:raise ValueError("Visual kernel identities failed")
    return {"geometric_mean_equals_dot_of_differences":err,"interaction_null_for_unchanged_image":null}
