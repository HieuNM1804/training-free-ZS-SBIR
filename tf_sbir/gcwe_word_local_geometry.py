"""Disintegrate GCWE geometry over photo patches and test word-specific spatial credit."""
import torch

from .gcwe_conditioned_donor_test import run
from .geometry_word_flat import distance_agreement, rank_density, reweight


def local_geometry(query, patches, distance, scale):
    sim=scale*(query[:,None]@patches.transpose(-1,-2)).float()
    joint=(sim.softmax(-1)+sim.softmax(-2))/98
    qa,gb=joint.sum(-1),joint.sum(-2)
    def moments(marginal):
        mean=((marginal@distance)*marginal).sum(-1)
        var=(((marginal@distance.square())*marginal).sum(-1)-mean.square()).clamp_min(0)
        return mean,var
    mu,var=moments(qa)
    nu,other=moments(gb)
    centered=(distance-mu[...,None,None]) @ joint @ (distance-nu[...,None,None]).transpose(-1,-2)
    local=(joint*centered).sum(-2)/gb.clamp_min(1e-30)
    local=local/(var*other).sqrt().clamp_min(1e-7)[...,None]
    original=distance_agreement(joint,distance,distance)
    if ((gb*local).sum(-1)-original).abs().max()>3e-5:
        raise ValueError('Patch contributions failed to reconstruct GCWE geometry')
    return local,gb,original


def projected(local,marginal,logits):
    responsibility=(marginal.clamp_min(1e-30).log()[...,None]+logits).softmax(2)
    return (responsibility*local[...,None]).sum(2)


def make_update(source,target,gallery):
    location_perm=torch.randperm(49,generator=torch.Generator().manual_seed(42)).cuda()
    def update(query,blank,patches,dist,scale,words,bank,nominees,fold,priors,unused):
        real,rm,score=local_geometry(query,patches,dist,scale)
        null,nm,floor=local_geometry(blank,patches,dist,scale)
        signal=score-floor
        head=words.topk(16).indices
        logits=scale*torch.einsum('bsnd,bkd->bsnk',patches,target[head].to(patches.dtype)).float()
        def credit(z):
            return projected(real,rm,z)-projected(null,nm,z)
        local=credit(logits)
        shuffled=credit(logits[:,:,location_perm])
        signals={'specific':local,'wrong_word':local.roll(1,-1),
                 'common_map':local.mean(-1,keepdim=True).expand_as(local),'wrong_patch':shuffled}
        evidence=bank[nominees]
        selected=evidence.gather(2,head[:,None].expand(-1,nominees.size(1),-1))
        outputs={}
        for t in range(2):
            eligible=fold[nominees]!=bool(t)
            mass=rank_density(signal,eligible,8)
            normal=eligible.float()[:,None].bmm(evidence).squeeze(1)+priors[1-t]
            observed=mass[:,None].bmm(evidence).squeeze(1)+priors[1-t]
            counts={'original':observed}
            b,n,k=local.shape
            mask=eligible[:,None].expand(-1,k,-1).reshape(b*k,n)
            for name,val in signals.items():
                ranks=rank_density(val.transpose(1,2).reshape(b*k,n).contiguous(),mask,8).reshape(b,k,n)
                adjusted=(ranks*selected.transpose(1,2)).sum(2)+priors[1-t][head]
                counts[name]=observed.scatter(1,head,adjusted)
            include=fold==bool(t)
            for name,qty in counts.items():
                value=reweight(words,qty/normal.clamp_min(1e-30))@bank.T
                if name not in outputs:outputs[name]=torch.empty_like(value)
                outputs[name][:,include]=value[:,include]
        return outputs
    return update


if __name__=='__main__':
    run(family='gcwe-word-local-geometry',update_factory=make_update,start=3840,count=1536,controls=False,
        comparisons=[('specific','original'),('specific','wrong_word'),('specific','common_map'),
                     ('specific','wrong_patch'),('specific','semantic_regional'),('original','semantic_regional')])
