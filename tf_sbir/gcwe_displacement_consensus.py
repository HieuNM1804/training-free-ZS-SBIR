"""Test whether disjoint sets of CLIP matches support a common displacement beyond marginal occupancy."""
from functools import lru_cache

import torch

from .gcwe_conditioned_donor_test import run
from .geometry_word_flat import distance_agreement, rank_density, reweight


@lru_cache(None)
def offset_geometry(device):
    coord = torch.cartesian_prod(torch.arange(7),torch.arange(7)).to(device)
    perm = torch.randperm(49,generator=torch.Generator().manual_seed(42)).to(device)
    blocks = []
    for parity in (0,1):
        rows = (coord.sum(1).remainder(2)==parity).nonzero().flatten()
        group = []
        for destination in (coord,coord[perm]):
            dy,dx = (destination[None]-coord[rows,None,:]).unbind(2)
            offset = (dy+6)*13+dx+6
            group.append(torch.nn.functional.one_hot(offset.flatten(),169).float())
        blocks.append((rows,*group))
    return blocks


def centered_histograms(coupling, geometry):
    observed, adjusted, shuffled = [], [], []
    for rows,offset,wrong_offset in geometry:
        c = coupling.index_select(-2,rows)
        c = c / c.sum((-1,-2),keepdim=True)
        independent = c.sum(-1,keepdim=True)*c.sum(-2,keepdim=True)
        mass = c.flatten(-2) @ offset
        residual = (c-independent).flatten(-2)
        centered = residual @ offset
        sham = residual @ wrong_offset
        if (mass.sum(-1)-1).abs().max() > 1e-5 or centered.sum(-1).abs().max() > 1e-5:
            raise ValueError('Displacement histogram lost mass')
        observed.append(mass)
        adjusted.append(centered)
        shuffled.append(sham)
    def agreement(pair):
        return (pair[0]*pair[1]).sum(-1)
    return {'cross':agreement(adjusted),'uncentered':agreement(observed),'permuted':agreement(shuffled),
            'same_partition':sum(v.square().sum(-1) for v in adjusted)/2}


def update(query,blank,patches,dist,scale,words,bank,nominees,fold,priors,unused):
    geometry=offset_geometry(query.device)
    def verify(tokens):
        z=scale*(tokens[:,None]@patches.transpose(-1,-2)).float()
        c=(z.softmax(-1)+z.softmax(-2))/98
        result=centered_histograms(c,geometry)
        result['original']=distance_agreement(c,dist,dist)
        return result
    real,null=verify(query),verify(blank)
    evidence=bank[nominees]
    outputs={}
    for t in range(2):
        eligible=fold[nominees]!=bool(t)
        normal=eligible.float()[:,None].bmm(evidence).squeeze(1)+priors[1-t]
        for name in real:
            mass=rank_density(real[name]-null[name],eligible,8)
            obs=mass[:,None].bmm(evidence).squeeze(1)+priors[1-t]
            score=reweight(words,obs/normal.clamp_min(1e-30))@bank.T
            if name not in outputs:outputs[name]=torch.empty_like(score)
            targets=fold==bool(t)
            outputs[name][:,targets]=score[:,targets]
    return outputs


if __name__=='__main__':
    run(update_fn=update,family='gcwe-displacement-consensus',start=8448,count=1536,controls=False,
        comparisons=[('cross','original'),('cross','semantic_regional'),('cross','uncentered'),
                     ('cross','permuted'),('cross','same_partition'),('original','semantic_regional')])
