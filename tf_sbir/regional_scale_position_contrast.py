"""Disentangle standalone positional evidence from the scale-position interaction in a candidate's regional score."""
import argparse

import torch

from .regional_held_word_attention import run


def complete_windows(w, dataset, inputs):
    from .gallery_panel_resume import region_views
    from .loo_source_windows import source_windows
    if dataset != 'sketchy':
        return source_windows(w,dataset,inputs)
    cfg,paths,splits,_,_,features,*_=inputs
    orders=[ids[torch.randperm(len(ids),generator=torch.Generator().manual_seed(42))]
            for ids,_ in splits.values()]
    covered=torch.zeros(len(paths),dtype=torch.bool)
    result={}
    for sl in (slice(0,1536),slice(1536,4608),None):
        ids=(~covered).nonzero().flatten() if sl is None else torch.cat([order[sl] for order in orders]).unique(sorted=True)
        if not len(ids):
            continue
        cache,_=region_views(w,'sampled-sketchy',cfg,paths,ids,features)
        for key in ('regions','flipped'):
            value=cache[key]
            if key not in result:
                result[key]=value.new_empty((len(paths),*value.shape[1:]))
            unseen=~covered[ids]
            result[key][ids[unseen]]=value[unseen]
        covered[ids]=True
    if not covered.all():
        raise ValueError('Missing frozen query windows')
    return result


def factorial_scores(reader):
    likelihood = reader.l
    evidence = reader.logl - likelihood.mean(-1, keepdim=True).clamp_min(1e-30).log()
    attention = (reader.prior.log() + evidence).softmax(2)
    shared = attention.mean(-1, keepdim=True)
    scale_gate, position_gate = [], []
    for rows in (slice(0, 9), slice(9, 13)):
        individual, average = attention[:, :, rows], shared[:, :, rows]
        scale = individual.sum(2, keepdim=True)
        shared_scale = average.sum(2, keepdim=True)
        scale_gate.append(scale * average / shared_scale.clamp_min(1e-30))
        position_gate.append(shared_scale * individual / scale.clamp_min(1e-30))

    def read(weights):
        return (weights * likelihood).sum(2).mean(1).clamp_min(1e-30).log()

    full = read(attention)
    scale = read(torch.cat(scale_gate, 2))
    position = read(torch.cat(position_gate, 2))
    base = read(shared)
    return full, base, scale, position


def decompose(reader):
    full, base, scale, position = factorial_scores(reader)
    interaction = (full-scale) - (position-base)
    additive = scale + position - base
    if (full - (additive+interaction)).abs().max() > 2e-5:
        raise ValueError('Log factorial decomposition changed the full score')

    # Substitute another candidate's interaction without altering its per-query marginal distribution.
    generator = torch.Generator().manual_seed(42)
    order = torch.randperm(interaction.size(1), generator=generator).to(interaction.device)
    donor = torch.empty_like(order).scatter_(0, order, order.roll(1))
    return {'self': full, 'shared': base, 'scale': scale, 'position': position,
            'additive': additive, 'scale_interaction': scale+interaction,
            'permuted_interaction': scale+interaction[:,donor]}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--confirmation',action='store_true')
    confirm=parser.parse_args().confirmation
    family='regional-additive-confirmation' if confirm else 'regional-scale-position-contrast'
    ref='regional-additive-device-reference' if confirm else 'regional-position-device-reference'
    comparisons=(('additive','self'),('additive','scale'),('additive','position')) if confirm else (
        ('scale_interaction','self'),('scale_interaction','scale'),
        ('scale_interaction','permuted_interaction'),('self','additive'),('position','shared'))
    run(family=family,alternate=decompose,start=1536 if confirm else 0,count=3072 if confirm else 1536,
        batch_size=24,selected_windows=True,device_reference=ref,contrasts=comparisons)
