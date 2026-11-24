"""Exact best-of-four semantic donor pairs for GCWE, with matched marginal controls."""
from itertools import product

import torch

from .gcwe_conditioned_donor_test import run
from .geometry_word_flat import distance_agreement, rank_density, reweight


def winner_law(pair,order,draws):
    # The diagonal of the 2D prefix sum is the chance for both indices to be in a given prefix.
    pair=pair.double()
    marginal=pair.sum(2)
    ordered=pair.gather(1,order[:,:,None].expand_as(pair)).gather(2,order[:,None,:].expand_as(pair))
    prefix=ordered.cumsum(1).cumsum(2).diagonal(dim1=1,dim2=2).clamp(0,1).pow(draws)
    increments=torch.diff(prefix,dim=1,prepend=torch.zeros_like(prefix[:,:1]))
    selected=torch.zeros_like(increments).scatter_(1,order,increments)
    if selected.min() < -1e-10 or (selected.sum(1)-1).abs().max()>1e-7:
        raise ValueError('Invalid best-of-pairs distribution')
    return marginal.float(),selected.clamp_min(0).float()


def check_small():
    weights=torch.tensor([[[0.,1.,2.],[1.,0.,3.],[2.,3.,0.]]],dtype=torch.double)
    weights/=weights.sum()
    order=torch.tensor([[1,2,0]])
    marginal,actual=winner_law(weights,order,2)
    truth=torch.zeros(3,dtype=torch.double)
    rank=order[0].argsort()
    for i,j,k,l in product(range(3),repeat=4):
        indices=torch.tensor([i,j,k,l])
        truth[indices[rank[indices].argmax()]]+=weights[0,i,j]*weights[0,k,l]
    torch.testing.assert_close(actual[0].double(),truth,rtol=1e-6,atol=1e-8)
    torch.testing.assert_close(marginal[0].double(),weights[0].sum(1),rtol=1e-6,atol=1e-8)


def update(query,blank,patches,dist,scale,words,bank,nominees,fold,priors,random_order):
    def verify(tokens):
        z=scale*(tokens[:,None]@patches.transpose(-1,-2)).float()
        return distance_agreement((z.softmax(-1)+z.softmax(-2))/98,dist,dist)
    signal=verify(query)-verify(blank)
    evidence=bank[nominees]
    root=evidence.sqrt()
    kernel=root@root.transpose(1,2)
    det=(1.-kernel.double().square()).clamp_min(0)
    det.diagonal(dim1=1,dim2=2).zero_()
    del root,kernel
    outputs={}
    for target in range(2):
        eligible=fold[nominees]!=bool(target)
        mask=eligible[:,:,None]&eligible[:,None,:]
        n=eligible.sum(1,keepdim=True)
        order=signal.masked_fill(~eligible,torch.inf).argsort(dim=1,stable=True)
        weights=det*mask
        distinct=mask.double()
        distinct.diagonal(dim1=1,dim2=2).zero_()
        fallback=weights.sum((1,2),keepdim=True)<=1e-12
        weights=torch.where(fallback,distinct,weights)
        pair=weights/weights.sum((1,2),keepdim=True)
        marg,win=winner_law(pair,order,4)
        # Product proposal retains both endpoint marginals, including independent self draws.
        mg=pair.sum(2)
        independent=mg[:,:,None]*mg[:,None,:]
        # Assign the exact same pair-affinity matrix to different eligible donor IDs.
        identity=torch.arange(order.size(1),device=order.device).expand_as(order)
        packed=identity.gather(1,eligible.int().argsort(dim=1,descending=True,stable=True))
        shuffled=random_order.expand_as(order)
        shuffled=shuffled.gather(1,eligible.gather(1,shuffled).int().argsort(dim=1,descending=True,stable=True))
        permutation=torch.empty_like(order).scatter_(1,packed,shuffled)
        sham=pair.gather(1,permutation[:,:,None].expand_as(pair)).gather(2,permutation[:,None,:].expand_as(pair))
        ordinary=rank_density(signal,eligible,8)
        plans={'original':(eligible.float(),ordinary),'semantic_pairs':(n*marg,n*win)}
        for name,law in [('equal_distinct',distinct/distinct.sum((1,2),keepdim=True)),
                         ('marginals_only',independent),('permuted',sham)]:
            unweighted,winner=winner_law(law,order,4)
            plans[name]=(n*unweighted,n*winner)
        for name,(den,num) in plans.items():
            normal=den[:,None].bmm(evidence).squeeze(1)+priors[1-target]
            observed=num[:,None].bmm(evidence).squeeze(1)+priors[1-target]
            score=reweight(words,observed/normal.clamp_min(1e-30))@bank.T
            if name not in outputs:outputs[name]=torch.empty_like(score)
            dest=fold==bool(target)
            outputs[name][:,dest]=score[:,dest]
    return outputs


if __name__=='__main__':
    check_small()
    run(update_fn=update,family='gcwe-semantic-donor-pairs',start=5376,count=1536,controls=False,
        comparisons=[('semantic_pairs',k) for k in ('original','equal_distinct','marginals_only',
                                                   'permuted','semantic_regional')])
