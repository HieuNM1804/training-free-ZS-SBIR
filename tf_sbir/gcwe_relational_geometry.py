"""Separate GCWE patch eccentricity matching from genuine two-correspondence geometry."""
import torch

from .gcwe_conditioned_donor_test import run
from .geometry_word_flat import distance_agreement, rank_density, reweight


def geometry_parts(c,dx,dy):
    px,py=c.sum(-1),c.sum(-2)
    def split_distance(p,d):
        expected=p@d
        mean=(expected*p).sum(-1)
        eccentric=expected-mean[...,None]
        centered=d-expected[...,None,:]-expected[...,:,None]+mean[...,None,None]
        variance=(((p@d.square())*p).sum(-1)-mean.square()).clamp_min(0)
        relation_var=(p[...,None,:]*p[...,:,None]*centered.square()).sum((-1,-2))
        return eccentric,centered,variance,relation_var
    ux,ax,vx,rx=split_distance(px,dx)
    uy,ay,vy,ry=split_distance(py,dy)
    denom=(vx*vy).sqrt().clamp_min(1e-7)
    first=2*(c*ux[..., :,None]*uy[...,None,:]).sum((-1,-2))/denom
    second=(c*(ax@c@ay.transpose(-1,-2))).sum((-1,-2))
    original=distance_agreement(c,dx,dy)
    if (original-first-second/denom).abs().max()>3e-5:
        raise ValueError('Distance covariance decomposition failed')
    return {'original':original,'eccentricity':first,'relational':second/denom,
            'relational_normalized':second/(rx*ry).sqrt().clamp_min(1e-7)}


def update(query,blank,patches,dist,scale,words,bank,nominees,fold,priors,unused):
    perm=torch.randperm(len(dist),generator=torch.Generator().manual_seed(42)).to(dist.device)
    def verify(tokens):
        logits=scale*(tokens[:,None]@patches.transpose(-1,-2)).float()
        c=(logits.softmax(-1)+logits.softmax(-2))/98
        values=geometry_parts(c,dist,dist)
        values['wrong_coordinates']=geometry_parts(c,dist,dist[perm][:,perm])['relational']
        return values
    real,null=verify(query),verify(blank)
    evidence=bank[nominees]
    outputs={}
    for t in range(2):
        eligible=fold[nominees]!=bool(t)
        reference=eligible.float()[:,None].bmm(evidence).squeeze(1)+priors[1-t]
        dest=fold==bool(t)
        for name in real:
            weights=rank_density(real[name]-null[name],eligible,8)
            observed=weights[:,None].bmm(evidence).squeeze(1)+priors[1-t]
            score=reweight(words,observed/reference.clamp_min(1e-30))@bank.T
            if name not in outputs:outputs[name]=torch.empty_like(score)
            outputs[name][:,dest]=score[:,dest]
    return outputs


if __name__=='__main__':
    run(family='gcwe-relational-geometry',start=6912,count=1536,update_fn=update,controls=False,
        comparisons=[('relational','original'),('relational','eccentricity'),('relational','wrong_coordinates'),
                     ('relational','semantic_regional'),('relational_normalized','relational'),
                     ('original','semantic_regional')])
