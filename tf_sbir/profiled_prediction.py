"""Conditional minimax prediction of the public name of a partially observed visual subject."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Bank:
    rows: torch.Tensor
    visual: torch.Tensor
    observed: torch.Tensor
    lexical_mle: torch.Tensor
    sizes: torch.Tensor
    last_gain: float


def prototypes(photos, components, steps=20):
    centers = photos[:components]
    for _ in range(steps):
        cell = (photos @ centers.T).argmax(1)
        members = F.one_hot(cell,num_classes=len(centers)).T.float()
        sizes = members.sum(1)
        if (sizes==0).any():
            members = members[sizes>0]
        centers = F.normalize(members @ photos,dim=1)
    cell = (photos @ centers.T).argmax(1)
    return cell


def estimate_laws(photos, native, cell):
    groups = F.one_hot(cell).T.float()
    counts = groups.sum(1)
    if (counts==0).any():
        keep = counts>0
        groups = groups[keep]
        counts = counts[keep]
        cell = groups.argmax(0)
    groups /= counts[:,None]
    means = groups @ photos
    measured = groups @ native
    # Maximize the population's soft-evidence mixture likelihood. A final likelihood change
    # reports numerical behavior of this estimator, independent of retrieval labels.
    pi = measured
    last = None
    change = None
    for iteration in range(96):
        joint = native * pi[cell]
        evidence = joint.sum(1,keepdim=True).clamp_min(1e-30)
        if iteration%8==7:
            ell = evidence.log().mean()
            if last is not None:
                change = (ell-last).item()
                if change < -1e-5:
                    raise ValueError("A mixture EM step reduced the training likelihood")
                if change < 2e-6:
                    break
            last = ell
        pi = groups @ (joint / evidence)
    return means, measured, pi, counts, change


def bank_models(photos, lexical, components, seed):
    permutation = torch.randperm(len(photos),generator=torch.Generator().manual_seed(seed)).to(photos.device)
    halves = permutation.chunk(2)
    result=[]
    for train,test in (halves, halves[::-1]):
        x=photos[train]
        cell=prototypes(x,components)
        centers,measured,mle,n,gain=estimate_laws(x,lexical[train],cell)
        y=photos[test]
        visual=y @ centers.T - .5*centers.square().sum(1)
        result.append(Bank(test,visual,measured,mle,n,gain))
    return result


def forecast(banks, scale, law, arms=("profile","bayes","map","unnormalized")):
    size=sum(len(b.rows) for b in banks)
    template=banks[0].observed
    output={arm:template.new_empty(size,template.size(1)) for arm in arms}
    diagnostics=[]
    for b in banks:
        theta=b.observed if law=="reports" else b.lexical_mle
        ll=scale*b.visual
        ll=ll-ll.amax(1,keepdim=True)
        like=ll.exp()
        if "bayes" in arms:
            output["bayes"][b.rows]=F.normalize(like,p=1,dim=1) @ theta
        if "empirical_bayes" in arms:
            output["empirical_bayes"][b.rows]=F.normalize(like*b.sizes,p=1,dim=1) @ theta
        if "map" in arms:
            output["map"][b.rows]=theta[like.argmax(1)]
        if not ({"profile","unnormalized","half_profile"}&set(arms)):
            continue
        norms=[]
        for i in range(0,len(b.rows),128):
            sl=slice(i,i+128)
            largest=(like[sl,:,None]*theta).amax(1)
            z=largest.sum(1,keepdim=True)
            if z.min()<1-3e-5 or not torch.isfinite(z).all():
                raise ValueError("Invalid Shtarkov sum or profile density")
            norms.append(z.flatten())
            if "profile" in arms:
                output["profile"][b.rows[sl]]=largest/z
            if "unnormalized" in arms:
                output["unnormalized"][b.rows[sl]]=largest
            if "half_profile" in arms:
                # Exact power mean control between the bayes average and the max envelope.
                mixture=F.normalize(like[sl].square() @ theta.square(),p=1,dim=1).sqrt()
                output["half_profile"][b.rows[sl]]=F.normalize(mixture,p=1,dim=1)
        log_z=torch.cat(norms).log()
        diagnostics.append({"mean_minimax_regret":log_z.mean().item(),"max_regret":log_z.max().item(),
                            "states":len(theta),"em_gain_last_8":b.last_gain})
    return output,diagnostics


def fixture():
    # Both experiments have the same Bayesian forecast, but admit a different worst oracle.
    first=torch.tensor([[1.,0.,0.],[0.,.5,.5]])
    second=torch.tensor([[.5,.5,0.],[.5,0.,.5]])
    def predict(theta):
        raw=theta.amax(0)
        return raw/raw.sum()
    assert torch.equal(first.mean(0),second.mean(0))
    a,b=predict(first),predict(second)
    assert not torch.allclose(a,b)
    for theta,q in ((first,a),(second,b)):
        bound=theta.amax(0).sum()
        assert torch.allclose((theta/q).amax(0),bound.expand(3))
    return {"same_mixture":first.mean(0).tolist(),"profile_A":a.tolist(),"profile_B":b.tolist()}
