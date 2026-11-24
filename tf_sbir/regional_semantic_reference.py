"""Compare regional windows against the gallery alternatives made plausible by this sketch."""
from functools import lru_cache

import torch

from .regional_held_word_attention import run


@lru_cache(None)
def permuted_gallery(size,device):
    return torch.randperm(size,generator=torch.Generator().manual_seed(42)).to(device)


def route(l,logl,prior,prob):
    if prob is None:
        mean=l.mean(-1,keepdim=True)
    else:
        mean=(l.flatten(1,2)@prob[:,:,None]).reshape(*l.shape[:3],1)
    return (prior.log()+logl-mean.clamp_min(1e-30).log()).softmax(2)


def within_scale(scale_from,position_from):
    parts=[]
    for sl in (slice(0,9),slice(9,13)):
        position=position_from[:,:,sl]
        conditional=position/position.sum(2,keepdim=True).clamp_min(1e-30)
        parts.append(scale_from[:,:,sl].sum(2,keepdim=True)*conditional)
    return torch.cat(parts,2)


def read(reader,global_score):
    l=reader.l
    prob=global_score.softmax(1)
    original=route(l,reader.logl,reader.prior,None)
    posterior=route(l,reader.logl,reader.prior,prob)
    perm=permuted_gallery(prob.size(1),prob.device)
    sham=route(l,reader.logl,reader.prior,prob[:,perm])
    gates={'self':original,'position':within_scale(original,posterior),'sham_position':within_scale(original,sham),
           'scale':within_scale(posterior,original),'both':posterior}
    if not torch.allclose(within_scale(original,original),original,rtol=1e-5,atol=2e-7):
        raise ValueError('Scale factorization changed the original attention')
    return {name:(gate*l).sum(2).mean(1).clamp_min(1e-30).log() for name,gate in gates.items()}


if __name__=='__main__':
    run(family='regional-semantic-reference',alternate=read,semantic_reference=True,start=2304,
        count=1536,batch_size=24,contrasts=(('position','self'),('position','sham_position'),
                                            ('position','scale'),('both','self')))
