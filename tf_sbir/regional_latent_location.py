"""Candidate-conditioned marginalization of overlapping crops at a common latent query location."""
from functools import lru_cache

import torch

from .regional_held_word_attention import run
from .regional_interventions import SIDES, windows


@lru_cache(None)
def kernel(device):
    positions=torch.cartesian_prod(torch.arange(7,device=device),torch.arange(7,device=device))
    blocks=[]
    for side in SIDES:
        masks=[]
        for y,x in windows(side):
            dy,dx=positions[:,0]-y,positions[:,1]-x
            masks.append((dy>=0)&(dy<side)&(dx>=0)&(dx<side))
        cover=torch.stack(masks,1).float()
        assert cover.sum(1).min()>0
        blocks.append(cover/cover.sum(1,keepdim=True)/len(SIDES))
    aligned=torch.cat(blocks,1)
    opposed=torch.cat((blocks[0],blocks[1].flip((0,))),1)
    assert torch.allclose(aligned.sum(1),torch.ones(49,device=device))
    assert torch.allclose(aligned.mean(0),opposed.mean(0))
    return aligned,opposed


def pool(likelihood,prior):
    positive=likelihood.clamp_min(1e-30)
    weight=(prior.log()+positive.log()-positive.mean(-1,keepdim=True).clamp_min(1e-30).log()).softmax(2)
    return (weight*positive).sum(2).mean(1).clamp_min(1e-30).log()


def read(reader):
    l=reader.l
    aligned,opposed=kernel(l.device)
    locations=aligned@l
    prior=l.new_full((1,1,49,1),1/49)
    cover=aligned.mean(0)[None,None,:,None]
    average=locations.mean(2)
    displaced=opposed@l
    if not torch.allclose(average,displaced.mean(2),rtol=2e-5,atol=1e-8):
        raise ValueError('The displacement changed a candidate marginal')
    alpha=(reader.prior.log()+reader.logl-l.mean(-1,keepdim=True).clamp_min(1e-30).log()).softmax(2)
    common=alpha.mean(-1,keepdim=True)
    scale_gate=[];position_gate=[]
    for region_ids in (slice(0,9),slice(9,13)):
        a=alpha[:,:,region_ids];m=common[:,:,region_ids]
        scale_gate.append(a.sum(2,keepdim=True)*m/m.sum(2,keepdim=True).clamp_min(1e-30))
        position_gate.append(m.sum(2,keepdim=True)*a/a.sum(2,keepdim=True).clamp_min(1e-30))
    def read_gate(gate):
        return (gate*l).sum(2).mean(1).clamp_min(1e-30).log()
    return {'self':pool(l,reader.prior),'location':pool(locations,prior),
            'displaced':pool(displaced,prior),'cover_prior':pool(l,cover),
            'shared':read_gate(common),'scale_only':read_gate(torch.cat(scale_gate,2)),
            'position_only':read_gate(torch.cat(position_gate,2))}


if __name__=='__main__':
    run(family='regional-latent-location',start=5376,alternate=read,
        contrasts=(('location','self'),('location','displaced'),('location','cover_prior'),
                   ('self','scale_only'),('self','position_only'),('self','shared')))
