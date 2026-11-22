"""Select only among complete rankings shared by all decision rules; no category labels enter this module."""

import torch

from .intent_utility import co_relevance, normalized_membership, utility

ALPHAS=(.2,.55,1.)
RULES=("joint","collapsed","first","incumbent")


def model_for(words, probabilities, library, temp, strength, base_f):
    profiles=library["memberships"][temp][words]
    joint,_,mean=co_relevance(probabilities,profiles)
    base_joint=torch.minimum(base_f[:,:,None],base_f[:,None,:])
    mean=(1-strength)*mean+strength*base_f
    joint=(1-strength)*joint+strength*base_joint
    return {"joint":joint,"collapsed":torch.minimum(mean[:,:,None],mean[:,None,:]),"first":mean,"incumbent":base_joint}


def plan(base, library, group, query_words, settings):
    groups=len(group.bincount())
    size=group.bincount()
    relative=(base-base.amax(1,keepdim=True)).exp()
    soft=relative.new_zeros((len(base),groups)).index_add_(1,group,relative)/size
    base_f=normalized_membership(soft/soft.amax(1,keepdim=True),size)
    models={}
    for count,t,temp,mix in settings:
        logits,words=(x[:,:count] for x in query_words)
        prob=(logits*t).softmax(1)
        measures=model_for(words,prob,library,temp,mix,base_f)
        pairs,first=library["pair"][words],library["first"][words]
        nominees=[words[:,0]]
        for rule,measure in measures.items():
            calculated=utility(pairs,first,rule,measure)
            nominees.append(words.gather(1,calculated.argmax(1,keepdim=True))[:,0])
        models[count,t,temp,mix]=(measures,torch.stack(nominees,1))
    return models


def action_values(model, which, complete, exposure_base, caps, alphas=ALPHAS):
    """All rules see the same source and rule-nominated words, blend options, and incumbent."""
    pair,first=complete
    reference_pair,reference_first=exposure_base
    ids=which.flatten()
    shape=(*which.shape,len(alphas))
    p=pair[ids].view(*shape,*pair.shape[-2:]).flatten(1,2)
    linear=first[ids].view(*shape,first.size(-1)).flatten(1,2)
    count=which.size(1)
    mask=torch.tensor(alphas,device=which.device).repeat(count)
    decisions={}
    for rule,measure in model.items():
        candidate=utility(p,linear,rule,measure)
        stay=utility(reference_pair[:,None],reference_first[:,None],rule,measure)[:,0]
        for cap in caps:
            safe=candidate.masked_fill(mask[None]>cap+1e-6,-torch.inf)
            best,index=safe.max(1)
            adopt=best>stay+1e-8
            decisions[rule,cap]=(torch.where(adopt,index+1,0), (best-stay).clamp_min(0))
    return decisions
