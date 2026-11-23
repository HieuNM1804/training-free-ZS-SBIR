"""Marginalize GCWE donor selection and exclude each nominated target from its own evidence."""
from functools import lru_cache
from itertools import combinations, product
from math import comb

import torch

from .gcwe_conditioned_donor_test import run
from .geometry_word_flat import distance_agreement, rank_density, reweight


@lru_cache(None)
def maximum_cdf(n, m, draws):
    """CDF over the maximum donor rank after a uniform m-subset and iid draws within it."""
    if not 1 <= m <= n or draws < 1:
        raise ValueError((n,m,draws))
    distinct = [1.]
    for step in range(draws):
        following = [0.] * (len(distinct) + 1)
        for k, prob in enumerate(distinct):
            following[k] += prob * k/m
            following[k+1] += prob * (1-k/m)
        distinct = following
    cdf = [sum(prob*comb(i,k)/comb(n,k) for k,prob in enumerate(distinct) if prob and k<=i)
           for i in range(n+1)]
    assert abs(cdf[-1]-1.) < 1e-12
    return torch.tensor(cdf,dtype=torch.float64)


def check_identity():
    # All subsets and all ordered draws on a small population give an independent exact oracle.
    n,m,d=5,3,4
    maxima = torch.tensor([max(t) for subset in combinations(range(1,n+1),m)
                           for t in product(subset,repeat=d)])
    exact = (maxima[:,None] <= torch.arange(n+1)).double().mean(0)
    assert torch.allclose(exact,maximum_cdf(n,m,d),atol=2e-15,rtol=0)
    full = (torch.arange(n+1).double()/n)**d
    assert torch.allclose(full,maximum_cdf(n,n,d),atol=2e-15,rtol=0)


def tie_mass(cdf, low, high):
    return ((cdf[high]-cdf[low])/(high-low).clamp_min(1)).float()


def deterministic_update(words, bank, nominees, signal, prior, m):
    """Compute all-gallery and candidate-excluded scores. The shared smoothing prior is fixed."""
    evidence = bank[nominees]
    n = nominees.size(1)
    ordered = signal.sort(1).values.contiguous()
    low = torch.searchsorted(ordered,signal.contiguous())
    high = torch.searchsorted(ordered,signal.contiguous(),right=True)
    cdf = maximum_cdf(n,m,8).to(signal.device)
    weights = tie_mass(cdf,low,high)
    assert (weights.sum(1)-1).abs().max()<1e-5
    uniform_sum = evidence.sum(1)
    updated = reweight(words,(m*weights[:,None,:].bmm(evidence).squeeze(1)+prior)/(m/n*uniform_sum+prior).clamp_min(1e-30))
    inclusive = updated @ bank.T
    exclusive = inclusive.clone()
    other_m = min(m,n-1)
    cdf_held = maximum_cdf(n-1,other_m,8).to(signal.device)
    columns = torch.arange(n, device=signal.device)
    for group in columns.split(16):
        held = signal[:,group,None]
        # If the removed donor precedes this rank it shifts both boundaries. If tied it shifts only the upper.
        lower = low[:,None] - (held<signal[:,None]).long()
        upper = high[:,None] - (held<=signal[:,None]).long()
        weights = tie_mass(cdf_held,lower,upper)
        weights.masked_fill_(columns[None,None,:]==group[None,:,None],0)
        assert (weights.sum(2)-1).abs().max()<1e-5
        observed = other_m*weights.bmm(evidence)+prior
        normal = (other_m/(n-1))*(uniform_sum[:,None]-evidence[:,group]).clamp_min(0)+prior
        posterior = words[:,None]*observed/normal.clamp_min(1e-30)
        posterior = posterior / posterior.sum(2,keepdim=True).clamp_min(1e-30)
        exclusive.scatter_(1,nominees[:,group],(posterior*evidence[:,group]).sum(2))
    return exclusive,inclusive


def update(query,blank,patches,dist,scale,words,bank,nominees,fold,priors,unused_order):
    def agreement(tokens):
        s=scale*(tokens[:,None]@patches.transpose(-1,-2)).float()
        return distance_agreement((s.softmax(-1)+s.softmax(-2))/98,dist,dist)
    signal=agreement(query)-agreement(blank)
    evidence=bank[nominees]
    outputs=[]
    for target in range(2):
        eligible=fold[nominees]!=bool(target)
        weights=rank_density(signal,eligible,8)
        prior=priors[1-target]
        observed=weights[:,None].bmm(evidence).squeeze(1)+prior
        normal=eligible.float()[:,None].bmm(evidence).squeeze(1)+prior
        outputs.append(reweight(words,observed/normal.clamp_min(1e-30)))
    ref=(torch.cat(outputs)@bank.T).reshape(2,len(words),-1)
    original=torch.where(fold[None],ref[1],ref[0])
    prior=(priors[0]*(~fold).sum()+priors[1]*fold.sum())/len(fold)
    expected,inclusive=deterministic_update(words,bank,nominees,signal,prior,100)
    all_donors,_=deterministic_update(words,bank,nominees,signal,prior,200)
    return {'original':original,'expected':expected,'inclusive':inclusive,'all_donors':all_donors}


if __name__=='__main__':
    check_identity()
    run(update_fn=update,family='gcwe-expected-donors',start=768,count=1536,controls=False)
