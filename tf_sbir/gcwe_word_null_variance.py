"""Discount GCWE word lifts using their exact variance under randomized donor identity."""
from itertools import permutations

import torch

from .gcwe_conditioned_donor_test import run
from .geometry_word_flat import distance_agreement, rank_density, reweight


def covariance_null(sum_w, sum_sq_w, n, variance_weight_sum):
    centered=(sum_sq_w.double()-sum_w.double().square()/n).clamp_min(0.)
    return variance_weight_sum*centered/(n-1)


def check_variance():
    a=torch.tensor([.12,.08,1.3,2.5],dtype=torch.double)
    b=torch.tensor([[.3,.9],[1.,.01],[.22,.1],[.61,.5]],dtype=torch.double)
    exact=torch.stack([a[list(perm)]@b for perm in permutations(range(4))]).var(0,unbiased=False)
    calculated=covariance_null(b.sum(0),b.square().sum(0),4,(a-a.mean()).square().sum())
    torch.testing.assert_close(calculated,exact,atol=1e-12,rtol=1e-12)


def make_update(src,tgt,bank):
    freq_order=bank.mean(0).argsort(stable=True)
    ids=torch.arange(len(freq_order),device=bank.device)
    perm=torch.empty_like(freq_order).scatter_(0,freq_order,freq_order[(ids^1).clamp_max(len(ids)-1)])

    def update(query,blank,patches,dist,scale,words,gallery,nominees,fold,priors,unused):
        def geometry(tokens):
            z=scale*(tokens[:,None]@patches.transpose(-1,-2)).float()
            return distance_agreement((z.softmax(-1)+z.softmax(-2))/98,dist,dist)
        signal=geometry(query)-geometry(blank)
        evidence=gallery[nominees]
        squared=evidence.square()
        outputs={}
        for t in range(2):
            valid=fold[nominees]!=bool(t)
            n=valid.sum(1,keepdim=True).double()
            if n.min()<=1:raise ValueError('Too few donors')
            mask=valid.float()[:,None]
            mass=rank_density(signal,valid,8)
            first=mask.bmm(evidence).squeeze(1)
            second=mask.bmm(squared).squeeze(1)
            base=first+priors[1-t]
            obs=mass[:,None].bmm(evidence).squeeze(1)+priors[1-t]
            ratio=obs/base.clamp_min(1e-30)
            delta=ratio.double()-1.
            centered=(mass.double()-mass.double().sum(1,keepdim=True)/n)*valid
            null=covariance_null(first,second,n,centered.square().sum(1,keepdim=True))/base.double().square().clamp_min(1e-60)
            tau2=(words*(delta.square()-null)).sum(1,keepdim=True).clamp_min(0)
            def trust(variance):
                return tau2/(tau2+variance).clamp_min(1e-60)
            weight=trust(null)
            energy=(words*delta.square()).sum(1,keepdim=True)
            common=((words*(weight*delta).square()).sum(1,keepdim=True)/energy.clamp_min(1e-60)).sqrt()
            scales={'precision':weight,'same_energy':common,'wrong_precision':trust(null[:,perm])}
            variants={'original':ratio}
            for name,coef in scales.items():
                variants[name]=(1.+coef*delta).clamp_min(0).float()
            if not torch.isfinite(null).all():raise ValueError('Invalid permutation variance')
            targets=fold==bool(t)
            for name,factor in variants.items():
                score=reweight(words,factor)@gallery.T
                if name not in outputs:outputs[name]=torch.empty_like(score)
                outputs[name][:,targets]=score[:,targets]
        return outputs
    return update


if __name__=='__main__':
    check_variance()
    run(family='gcwe-word-null-variance',start=5376,count=1536,controls=False,update_factory=make_update,
        comparisons=[('precision','original'),('precision','same_energy'),('precision','wrong_precision'),
                     ('precision','semantic_regional'),('original','semantic_regional')])
