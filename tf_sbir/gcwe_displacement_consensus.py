"""Test whether disjoint sets of CLIP matches support a common displacement beyond marginal occupancy."""
from functools import lru_cache
import argparse

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


def concentration_controls(coupling, geometry):
    native, shuffled, raw, individual = [], [], [], []
    for rows, offset, sham_offset in geometry:
        c=coupling.index_select(-2,rows)
        c=c/c.sum((-1,-2),keepdim=True)
        residual=(c-c.sum(-1,keepdim=True)*c.sum(-2,keepdim=True)).flatten(-2)
        native.append(residual@offset)
        shuffled.append(residual@sham_offset)
        raw.append(c.flatten(-2)@offset)
        individual.append(residual.square().sum(-1))
    def energy(histograms):
        return sum(v.square().sum(-1) for v in histograms)/2
    whole_vote=(native[0]+native[1])/2
    difference=(native[0]-native[1])/2
    shared=whole_vote.square().sum(-1)
    partition=difference.square().sum(-1)
    full=energy(native)
    if (full-shared-partition).abs().max()>1e-6:
        raise ValueError('Displacement-energy identity failed')
    return {'same_partition':full,'wrong_coordinates':energy(shuffled),'uncentered_energy':energy(raw),
            'individual_matches':sum(individual)/2, 'consensus_energy':shared,'difference_energy':partition}


GEOMETRY_ABLATIONS = ('without_blank_reference', 'without_donor_pseudocount',
                      'without_uniform_donor_correction', 'hard_best_donor', 'without_gallery_crossfit')


def score_update(measure,query,blank,patches,dist,scale,words,bank,nominees,fold,priors,unused,
                 *, ablate_internal=False):
    geometry=offset_geometry(query.device)
    def verify(tokens):
        z=scale*(tokens[:,None]@patches.transpose(-1,-2)).float()
        c=(z.softmax(-1)+z.softmax(-2))/98
        result=measure(c,geometry)
        if not ablate_internal:
            result['original']=distance_agreement(c,dist,dist)
        return result
    real,null=verify(query),verify(blank)
    evidence=bank[nominees]
    outputs={}
    for t in range(2):
        eligible=fold[nominees]!=bool(t)
        support=eligible.float()[:,None].bmm(evidence).squeeze(1)
        reference=priors[1-t]
        normal=support+reference
        targets=fold==bool(t)
        for name in real:
            signal=real[name]-null[name]
            mass=rank_density(signal,eligible,8)
            observed=mass[:,None].bmm(evidence).squeeze(1)
            obs=observed+reference
            score=reweight(words,obs/normal.clamp_min(1e-30))@bank.T
            if name not in outputs:outputs[name]=torch.empty_like(score)
            outputs[name][:,targets]=score[:,targets]
            if ablate_internal and name=='same_partition':
                blank_free=rank_density(real[name],eligible,8)
                maxima=signal.masked_fill(~eligible,-torch.inf).amax(1,keepdim=True)
                ties=(signal==maxima) & eligible
                hard=ties.float()*(eligible.sum(1,keepdim=True)/ties.sum(1,keepdim=True).clamp_min(1))
                ratios={
                    'without_blank_reference':
                        (blank_free[:,None].bmm(evidence).squeeze(1)+reference)/normal.clamp_min(1e-30),
                    'without_donor_pseudocount':
                        torch.where(support>0,observed/support,torch.ones_like(support)),
                    'without_uniform_donor_correction': obs,
                    'hard_best_donor':
                        (hard[:,None].bmm(evidence).squeeze(1)+reference)/normal.clamp_min(1e-30),
                }
                for control,ratio in ratios.items():
                    if control not in outputs:outputs[control]=torch.empty_like(score)
                    candidate=reweight(words,ratio.clamp_min(1e-30))@bank.T
                    outputs[control][:,targets]=candidate[:,targets]
    if ablate_internal:
        eligible=torch.ones_like(nominees,dtype=torch.bool)
        signal=real['same_partition']-null['same_partition']
        mass=rank_density(signal,eligible,8)
        reference=bank.mean(0)
        numerator=mass[:,None].bmm(evidence).squeeze(1)+reference
        denominator=evidence.sum(1)+reference
        outputs['without_gallery_crossfit']=reweight(words,numerator/denominator.clamp_min(1e-30))@bank.T
    return outputs


if __name__=='__main__':
    from functools import partial
    parser=argparse.ArgumentParser()
    parser.add_argument('--confirm-concentration',action='store_true')
    parser.add_argument('--full', choices=['sketchy','tuberlin'])
    args=parser.parse_args()
    confirm=args.confirm_concentration or args.full
    if confirm:
        primary='same_partition'
        comparisons=[(primary,c) for c in ('original','wrong_coordinates','individual_matches',
                                          'uncentered_energy','consensus_energy','difference_energy','semantic_regional')]
        measure=concentration_controls
        if args.full:
            def measure(coupling, geometry):
                stats=concentration_controls(coupling, geometry)
                return {k:stats[k] for k in ('same_partition','wrong_coordinates','uncentered_energy','individual_matches')}
            comparisons=[(a,b) for a,b in comparisons if b not in ('consensus_energy','difference_energy')]
        family='gcwe-displacement-full' if args.full else 'gcwe-displacement-concentration'
        run(update_fn=partial(score_update,measure),family=family,dataset=args.full or 'sketchy',
            start=9984,count=None if args.full else 1536,controls=False,comparisons=comparisons)
    else:
        run(update_fn=partial(score_update,centered_histograms),family='gcwe-displacement-consensus',
            start=8448,count=1536,controls=False,comparisons=[('cross','original'),('cross','semantic_regional'),
            ('cross','uncentered'),('cross','permuted'),('cross','same_partition'),('original','semantic_regional')])
