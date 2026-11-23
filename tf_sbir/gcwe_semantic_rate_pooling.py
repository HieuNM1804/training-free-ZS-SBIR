"""Pool sparse GCWE rate evidence while preserving the original CLIP query distribution."""
import torch
from torch.nn import functional as F

from .gcwe_conditioned_donor_test import run
from .geometry_word_flat import distance_agreement, rank_density, reweight


def paired_words(source,target):
    mean=F.normalize(source+target)
    ids=torch.arange(len(mean),device=mean.device)
    neighbors=[]
    for start in range(0,len(ids),256):
        block=ids[start:start+256]
        sim=mean[block] @ mean.T
        sim[torch.arange(len(block),device=block.device),block]=-torch.inf
        neighbors.append(sim.argmax(1))
    neighbors=torch.cat(neighbors)
    left=ids[(neighbors[neighbors]==ids)&(ids<neighbors)]
    return left,neighbors[left]


def rate_factory(src,tgt,bank):
    left,right=paired_words(src,tgt)
    support=torch.cat((left,right))
    frequency=support[bank.mean(0)[support].argsort(stable=True)]
    freq_l,freq_r=frequency[::2],frequency[1::2]

    def update(query,blank,patches,dist,scale,words,gallery,nominees,fold,priors,unused):
        def agreement(tokens):
            sim=scale*(tokens[:,None]@patches.transpose(-1,-2)).float()
            return distance_agreement((sim.softmax(-1)+sim.softmax(-2))/98,dist,dist)
        signal=agreement(query)-agreement(blank)
        evidence=gallery[nominees]
        outputs={}
        for target in range(2):
            eligible=fold[nominees]!=bool(target)
            mass=rank_density(signal,eligible,8)
            observed=mass[:,None].bmm(evidence).squeeze(1)+priors[1-target]
            normal=eligible.float()[:,None].bmm(evidence).squeeze(1)+priors[1-target]
            rates=observed/normal.clamp_min(1e-30)
            variants={'original':rates}
            for name,a,b in [('near_pool',left,right),('freq_pool',freq_l,freq_r)]:
                pooled=(observed[:,a]+observed[:,b])/(normal[:,a]+normal[:,b]).clamp_min(1e-30)
                v=rates.clone()
                v[:,a],v[:,b]=pooled,pooled
                variants[name]=v
            identity=rates.clone()
            identity[:,support]=1.
            variants['neutral_on_pairs']=identity
            target_ids=fold==bool(target)
            for name,rate in variants.items():
                score=reweight(words,rate)@gallery.T
                if name not in outputs:outputs[name]=torch.empty_like(score)
                outputs[name][:,target_ids]=score[:,target_ids]
        return outputs
    return update


if __name__=='__main__':
    run(family='gcwe-semantic-rate-pooling',start=2304,count=1536,controls=False,
        update_factory=rate_factory,comparisons=[('near_pool','original'),('near_pool','freq_pool'),
            ('near_pool','neutral_on_pairs'),('original','semantic_regional'),('near_pool','semantic_regional')])
