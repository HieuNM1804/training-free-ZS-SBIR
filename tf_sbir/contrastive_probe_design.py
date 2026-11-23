"""Design low-cost CLIP choice measurements under the current sketch's noun posterior."""
import math
import numpy as np
import torch


BLOCKS=50
SIZE=12


def binary_entropy(p):
    z=p.clamp(1e-7,1-1e-7)
    return -z*z.log()-(1-z)*torch.log1p(-z)


def noun_samples(prior,size,seed):
    # Stratified quadrature of the entire distribution; no shortlist of nouns.
    # Permute vocabulary order to avoid aliasing taxonomic structure.
    rng=torch.Generator().manual_seed(seed)
    order=torch.randperm(prior.size(1),generator=rng).to(prior.device)
    cdf=prior[:,order].double().cumsum(1)
    cdf/=cdf[:,-1:].clone()
    strata=(torch.arange(size,device=prior.device,dtype=torch.double)+
            torch.rand(size,generator=rng,dtype=torch.double).to(prior.device))/size
    ix=torch.searchsorted(cdf.contiguous(),strata.expand(len(prior),-1).contiguous()).clamp_max(len(order)-1)
    return order[ix]


def sample_information(groups,word_ids,prior,common=None):
    # groups: batch x blocks x options (gallery indices). Utility is computed
    # only for sampled noun hypotheses for design, exactly for all nouns at readout.
    bank,idx=groups
    size=idx.size(-1)
    ai,bi=torch.triu_indices(size,size,offset=1,device=idx.device)
    if common is None:
        pieces=[]
        for i in range(len(idx)):
            utility=bank[idx[i][:,:,None],word_ids[i][None,None,:]]+prior[word_ids[i]][None,None,:]
            pieces.append(utility.clamp_min(1e-30).log())
        common=torch.stack(pieces)
    info=[]
    entropy=[]
    for g in common.split(5,dim=1):
        prob=(g[:,:,ai,:]-g[:,:,bi,:]).sigmoid()
        h=binary_entropy(prob.mean(3))
        info.append((h-binary_entropy(prob).mean(3)).clamp_min(0))
        entropy.append(h)
    return torch.cat(info,1),torch.cat(entropy,1)


def design(bank,group_ids,qprior,prior,random_pairs,check=False):
    b,h,n=group_ids.shape
    ai,bi=torch.triu_indices(n,n,offset=1,device=bank.device)
    qids=noun_samples(qprior,512,83920)
    bgids=noun_samples(prior[None],512,69511).expand(b,-1)
    info,entropy=sample_information((bank,group_ids),qids,prior)
    common,_=sample_information((bank,group_ids),bgids,prior)
    choices={'information':info.argmax(2),'uncertainty':entropy.argmax(2),
             'background_information':common.argmax(2),'random':random_pairs.expand(b,-1)}
    # The original global cosine rank before the within-block permutation
    # is supplied to the caller for an equally sized closest-pair control.
    if check:
        v,_=sample_information((bank,group_ids),noun_samples(qprior,2048,11229),prior)
        choice=choices['information']
        fresh=(v.argmax(2)==choice).float().mean(1)
        regret=(v.max(2).values-v.gather(2,choice[:,:,None]).squeeze(2)).mean(1)
    else:
        fresh=regret=None
    return choices,info,ai,bi,fresh,regret


def smooth_utilities(bank,ids,prior):
    # Both sides of every pair are read through the same public noun bank.
    left,right=ids.unbind(-1)
    u=bank[left]+prior
    v=bank[right]+prior
    return u/(u+v).clamp_min(1e-30)


def orientation_posterior(prior,prob,choice):
    """Integrate a query-specific reliability between zero and one.

    The product is a polynomial of degree tests, so Gauss-Legendre integration
    with ceil((tests+1)/2) nodes is exact up to numeric precision.
    """
    count=prob.size(1)
    support=(2*prob-1)*choice[:,:,None]
    nodes,weights=np.polynomial.legendre.leggauss((count+2)//2)
    nodes=torch.as_tensor((nodes+1)/2,device=prob.device,dtype=prob.dtype)
    weights=torch.as_tensor(weights/2,device=prob.device,dtype=prob.dtype)
    terms=[]
    for t,weight in zip(nodes,weights,strict=True):
        terms.append(torch.log1p(t*support).sum(1)+weight.log())
    grid=torch.stack(terms,1)+prior.clamp_min(1e-30).log()[:,None]
    evidence=grid.flatten(1).logsumexp(1)
    joint=(grid-evidence[:,None,None]).exp()
    post=joint.sum(1)
    weighted_words=(joint*nodes[None,:,None]).sum(1)
    return post,weighted_words,evidence


def test_likelihood():
    # Two choices favouring the same hypothesis give integral 1+a+a²/3;
    # changing every direction reverses a. A tied choice contributes one.
    p=torch.tensor([[[.85,.15],[.85,.15],[.6,.4]]],device='cuda')
    q=torch.tensor([[.4,.6]],device='cuda')
    y=torch.tensor([[1.,1.,0.]],device='cuda')
    a=2*p[:,0]-1
    expected=q*(1+a+a.square()/3)
    expected/=expected.sum(1,keepdim=True)
    actual,_,_=orientation_posterior(q,p,y)
    if (actual-expected).abs().max()>2e-6:raise ValueError('Reliability integral differs')
    return float((actual-expected).abs().max())
