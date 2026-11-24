"""Does the frozen additive readout change how a fixed sketch ranks the same two photos in two galleries?"""
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .data import UNSEEN_CLASSES
from .gallery_context_reversals import coordinates
from .geometry_word_flat import (CHECKPOINT, CONFIG, GLOBAL_WEIGHT, REGIONAL_WEIGHT,
    Index, CandidateRegions, atomic_json, digest, index, log_score, make_word_bank, revision, semantic_inputs)
from .regional_scale_position_contrast import complete_windows, factorial_scores


def boundaries(fixed,truth):
    order=fixed.argsort(dim=1,descending=True,stable=True)[:,:200]
    hits=truth.gather(1,order)
    mask=hits[:,:-1]^hits[:,1:]
    q,place=mask.nonzero(as_tuple=True)
    a,b=order[q,place],order[q,place+1]
    positive=torch.where(hits[q,place],a,b)
    negative=torch.where(hits[q,place],b,a)
    return q,positive,negative


def diagnose(scores,truth,ids,pos,neg):
    by_method={}
    opposite={}
    correctness={}
    for k in scores[0]:
        margins=torch.stack([d[k][ids,pos]-d[k][ids,neg] for d in scores])
        signs=margins.sign()
        reversal=signs.prod(0)<0
        strict=signs>0
        opposite[k]=reversal
        correctness[k]=strict
        top=[x[k].argmax(1) for x in scores]
        tq=torch.arange(len(truth))
        by_method[k]={'pair_reversals':int(reversal.sum()),
            'ties_on_either':int((signs==0).any(0).sum()),
            'fixed_pair_both_correct':int(strict.all(0).sum()),
            'fixed_pair_both_wrong':int((signs<0).all(0).sum()),
            'correct_fixed_pairs_per_gallery':strict.sum(1).tolist(),
            'correct_top1_among_shared':[int(truth[tq,t].sum()) for t in top]}
    comparisons={}
    for original,proposed in (('region','region_additive'),('combined','combined_additive')):
        was,now=opposite[original],opposite[proposed]
        stable_right=correctness[proposed].all(0);stable_wrong=~correctness[proposed].any(0)
        comparisons[proposed]={'reversals_prevented':int((was&~now).sum()),
            'reversals_created':int((~was&now).sum()),
            'previously_reversing_now_both_correct':int((was&stable_right).sum()),
            'previously_reversing_now_neither_correct':int((was&stable_wrong).sum())}
    return by_method,comparisons


@torch.inference_mode()
def run():
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    w=Path('/content/sbir')
    root=w/'results/regional-additive-full'
    inputs=semantic_inputs(w,'sketchy')
    _,qp,qs,pp,gs,q,p,src,tgt,scale=inputs
    views=complete_windows(w,'sketchy',inputs)
    names=('sketchy_1','sketchy_2')
    def common(groups):
        return torch.tensor(sorted(set(groups[names[0]][0].tolist()) & set(groups[names[1]][0].tolist())))
    cq,cp=common(qs),common(gs)
    assert (len(cq),len(cp))==(2231,2205)
    # Define the probe once from direct two-view word agreement restricted to identical image IDs.
    directbank=make_word_bank(p[cp].cuda(),tgt,scale,256)
    direct=[]
    for ids in cq.split(24):
        poses=torch.stack((q[ids].cuda(),views['flipped'][ids].cuda().float()),1)
        direct.append(((scale*(poses@src.T)).softmax(2).mean(1)@directbank.T).cpu())
    direct=torch.cat(direct)
    del directbank
    classes=[Path(qp[j]).parent.name for j in cq.tolist()]
    photo_classes=[Path(pp[j]).parent.name for j in cp.tolist()]
    cat=sorted(set(classes))
    truth=torch.tensor([cat.index(c) for c in classes])[:,None]==torch.tensor([cat.index(c) for c in photo_classes])[None]
    ids,pos,neg=boundaries(direct,truth)
    scores=[]
    for split in names:
        pi,gl=gs[split]
        ci=coordinates(pi,cp)
        assert [UNSEEN_CLASSES[split][j] for j in gl[ci].tolist()]==photo_classes
        gallery=p[pi].cuda()
        stamp=digest({'gallery':[str(pp[j]) for j in pi.tolist()],'model':CHECKPOINT,'source':'regional-64/RGB16'})
        head,bank=index(w,split,gallery,tgt,scale,stamp[:16])
        globalbank=make_word_bank(gallery,tgt,scale,256)
        graph=Index(SimpleNamespace(p=gallery),rgb_only=True)
        buffers={}
        for block in cq.split(24):
            poses=torch.stack((q[block].cuda(),views['flipped'][block].cuda().float()),1)
            crops=views['regions'][block].cuda()
            raw=(scale*(poses@src.T)).softmax(2).mean(1)@globalbank.T
            if (raw[:,ci].cpu()-direct[torch.searchsorted(cq,block)]).abs().max()>2e-5:
                raise ValueError('Direct probe changed between gallery contexts')
            lex=log_score(graph.score(CONFIG,raw))
            reader=CandidateRegions(poses[:,0],poses[:,1],crops[:,:,0],crops[:,:,1],src,head,bank,src.new_empty((0,len(src))),raw)
            reader.prepare(scale)
            native,shared,scale_only,position_only=factorial_scores(reader)
            add=scale_only+position_only-shared
            for key,value in [('region',native),('region_additive',add),('combined',GLOBAL_WEIGHT*lex+REGIONAL_WEIGHT*native),
                              ('combined_additive',GLOBAL_WEIGHT*lex+REGIONAL_WEIGHT*add)]:
                buffers.setdefault(key,[]).append(value[:,ci].cpu())
        scores.append({k:torch.cat(v) for k,v in buffers.items()})
        del globalbank,graph,buffers
    methods,comparisons=diagnose(scores,truth,ids,pos,neg)
    out={'commit':revision(),'queries':len(cq),'photos':len(cp),'distinct_queries_with_boundaries':len(ids.unique()),
         'fixed_relevance_pairs':len(ids),'classes':cat,'methods':methods,'transitions':comparisons}
    atomic_json(root/'shared-gallery-decisions.json',out)
    print(json.dumps(comparisons))


if __name__=='__main__':
    run()
