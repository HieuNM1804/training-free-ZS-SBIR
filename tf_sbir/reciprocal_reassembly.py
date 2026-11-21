"""Reassemble the same visual patch contents in two estimated corresponding layouts."""

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F


def seq(v,patches):
    raw=patches.to(v.conv1.weight.dtype)
    cls=v.class_embedding.to(raw).expand(len(raw),1,-1)
    return v.ln_pre(torch.cat((cls,raw),1)+v.positional_embedding.to(raw))


def local(v,patches):
    x=seq(v,patches)
    for layer in list(v.transformer.resblocks)[:8]:x=layer(x)
    return x[:,1:].float()


def encode(v,patches):
    x=v.transformer(seq(v,patches))
    pooled,_=v._pool(x)
    return F.normalize((pooled@v.proj).float())


def read_bank(raw,ids):
    return torch.from_numpy(raw[ids.detach().cpu().numpy()]).cuda()


def matching_features(x,reference):
    # Position-dependent gallery means remove the common native positional code.
    signal=x-reference
    return F.normalize(signal-signal.mean(1,keepdim=True),dim=-1)


def correspond(photo,query):
    sim=photo@query.transpose(1,2)
    assignments=[linear_sum_assignment(row,maximize=True)[1] for row in sim.float().cpu().numpy()]
    perm=torch.from_numpy(np.stack(assignments)).to(photo.device)
    cost=sim.gather(2,perm[:,:,None]).mean((1,2))
    return perm,cost


def affinity(a,b,src,tgt,scale,subset):
    if subset=="dot":return scale*(a*b).sum(-1)
    s,t=src,tgt
    qa=(a@s.T)*scale
    pb=(b@t.T)*scale
    return (qa.log_softmax(-1)+pb.log_softmax(-1)).logsumexp(-1)


def reassemble(v,qraw,praw,q,p,perm,src,tgt,scale,heads):
    grid=torch.arange(49,device=perm.device).reshape(7,7)
    maps=(grid.flatten(),grid.rot90(1).flatten(),grid.flip((1,)).flatten())
    ar=torch.arange(len(q),device=perm.device)[:,None]
    scores={}
    worst=0.
    for j,h in enumerate(maps):
        move=h[perm[:,h.argsort()]]
        inverse=move.argsort(dim=1)
        if not (move.gather(1,inverse)==grid.flatten()).all():raise ValueError("Non-bijective patch match")
        placed=encode(v,qraw[ar,move])
        other=encode(v,praw[ar,inverse])
        for name,subset in (("dot","dot"),("all","all"),("compact",heads)):
            if name=="compact":
                # Use the same vocabulary on all four corners of this pair.
                ss,tt=src[heads],tgt[heads]
                score=lambda a,b:affinity(a,b,ss,tt,scale,"all")
            else:score=lambda a,b:affinity(a,b,src,tgt,scale,subset)
            original=score(q,p)
            first=score(placed,p)-original
            second=score(q,other)-original
            swap=score(placed,other)-original
            interaction=.5*(first+second-swap)
            if name=="dot":
                direct=.5*scale*((q-placed)*(other-p)).sum(-1)
                worst=max(worst,float((direct-interaction).abs().max()))
            scores[name,j,"interaction"]=interaction
            scores[name,j,"one"]=first
            scores[name,j,"aligned"]=(first+second)/2
        scores["self",j,"recovery"]=.5*((placed*q).sum(-1)+(other*p).sum(-1))*scale
    if worst>2e-4:raise ValueError(f"Four-corner dot product failed, {worst}")
    return scores,worst


def adjust(scores,group_ids,nominated,correction,weight):
    """Preserve total score mass on the nominated support for each independent query."""
    delta=torch.zeros((len(scores),int(group_ids.max())+1),device=scores.device)
    delta.scatter_(1,nominated,weight*correction)
    valid=torch.zeros_like(delta,dtype=torch.bool)
    valid.scatter_(1,nominated,True)
    mask=valid[:,group_ids]
    proposal=scores+delta[:,group_ids]
    base_mass=scores.masked_fill(~mask,-torch.inf).logsumexp(1,keepdim=True)
    new_mass=proposal.masked_fill(~mask,-torch.inf).logsumexp(1,keepdim=True)
    return torch.where(mask,proposal+base_mass-new_mass,scores)
