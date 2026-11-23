"""Test whether a readout word can be independently localized by the other 63 regional words."""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from .geometry_word_flat import (CONFIG, CHECKPOINT, GLOBAL_WEIGHT, REGIONAL_WEIGHT, KD_FUNCTION_SHA,
    Index, PrefixReplay, CandidateRegions, class_interval, digest, index, load_kd, make_word_bank,
    semantic_inputs, source_windows, save_tensor, score_rows, atomic_json, revision)
import hashlib
import inspect
import random
import numpy as np
import torch


def except_word(contributions):
    # Summing both sides is stable even when one word accounts for almost all of the likelihood.
    left = contributions.cumsum(-1)
    right = contributions.flip((-1,)).cumsum(-1).flip((-1,))
    zero = torch.zeros_like(left[..., :1])
    return torch.cat((zero, left[..., :-1]), -1) + torch.cat((right[..., 1:], zero), -1)


def held_read(reader, paired):
    z = reader.region * reader.scale
    word = (z-z.amax((1,2,3),keepdim=True)).exp()
    bank = reader.bank
    avg = except_word(word * bank.mean(0))
    out = {k: [] for k in ('held','sham','self')}
    for block in bank.split(256):
        c = word.unsqueeze(-2)*block
        # [query, pose, region, candidate, word]
        mass = except_word(c)
        norm = avg.unsqueeze(-2).clamp_min(1e-30)
        prior = reader.prior.unsqueeze(-1)
        gate = mass/norm * prior
        gate = gate / gate.sum(2,keepdim=True).clamp_min(1e-30)
        # With exactly zero evidence for a whole (pose, candidate, held-word) tuple, use the region prior.
        gate = torch.where(gate.sum(2,keepdim=True)>0,gate,prior/prior.sum(2,keepdim=True))
        shared = c.sum(-1)
        scale = (word*bank.mean(0)).sum(-1)
        original = reader.prior*shared/scale[:,:,:,None].clamp_min(1e-30)
        original = original / original.sum(2,keepdim=True).clamp_min(1e-30)
        out['self'].append((original*shared).sum(2).mean(1))
        out['held'].append((gate*c).sum(-1).sum(2).mean(1))
        out['sham'].append((gate.index_select(-1,paired)*c).sum(-1).sum(2).mean(1))
    return {k:torch.cat(v,1).clamp_min(1e-30).log() for k,v in out.items()}


@torch.inference_mode()
def run():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    w=Path('/content/sbir')
    root=w/'results/regional-held-word-attention'
    root.mkdir(exist_ok=True,parents=True)
    inputs=semantic_inputs(w,'sketchy')
    cfg,qp,qs,pp,gs,q,p,src,tgt,scale=inputs
    views=source_windows(w,'sketchy',inputs)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    for split,(qi,labels) in qs.items():
        pi,gl=gs[split]
        select=torch.randperm(len(qi),generator=torch.Generator().manual_seed(42))[768:2304].sort().values
        gallery=p[pi].cuda()
        graph=Index(SimpleNamespace(p=gallery),rgb_only=True)
        paths=[str(pp[i]) for i in pi.tolist()]
        stamp=digest({'gallery':paths,'model':CHECKPOINT,'source':'regional-64/RGB16'})
        heads,bank=index(w,split,gallery,tgt,scale,stamp[:16])
        by_frequency=bank.mean(0).argsort(stable=True)
        peer=torch.empty_like(by_frequency).scatter_(0,by_frequency,by_frequency[torch.arange(len(heads),device='cuda')^1])
        fullbank=make_word_bank(gallery,tgt,scale,256)
        ref=torch.load(w/'results/geometry-word-flat-first'/(split+'.pt'),weights_only=True,mmap=True)
        assert torch.equal(qi,ref['qi']) and torch.equal(pi,ref['pi'])
        replay=PrefixReplay(kd,gl,split)
        rows={}
        check=0.
        with ThreadPoolExecutor(max_workers=min(3,len(os.sched_getaffinity(0)))) as pool:
            for batch in select.split(8):
                ids=qi[batch]
                a,b=q[ids].cuda(),views['flipped'][ids].cuda().float()
                crops=views['regions'][ids].cuda()
                poses=torch.stack((a,b),1)
                direct=(scale*(poses@src.T)).softmax(2).mean(1) @ fullbank.T
                lex=graph.score(CONFIG,direct).clamp_min(1e-30).log()
                reader=CandidateRegions(a,b,crops[:,:,0],crops[:,:,1],src,heads,bank,src.new_empty((0,len(src))),direct)
                reader.prepare(scale)
                reader.scale=scale
                existing=reader.read('self',1.,'gallery')
                tested=held_read(reader,peer)
                check=max(check,float((tested['self']-existing).abs().max()))
                for key,r in {'self':existing,'held':tested['held'],'sham':tested['sham']}.items():
                    for name,value in [('regional_'+key,r),('global_regional_'+key,GLOBAL_WEIGHT*lex+REGIONAL_WEIGHT*r)]:
                        rows.setdefault(name,[]).append(score_rows(value,labels[batch],gl.cuda(),split,replay,pool)[:,:2])
        rows={k:torch.cat(v) for k,v in rows.items()}
        mismatch=(100*(rows['global_regional_self']-ref['rows']['without_geometry'][select,:2]).double().abs().mean(0))
        if check>.01 or mismatch.max()>.04:raise ValueError((check,mismatch))
        result={'commit':revision(),'seed':42,'queries':len(select),'gallery':len(pi),'algebra_max':check,
                'reference_mae_points':mismatch.tolist(),'metrics':{k:(100*v.double().mean(0)).tolist() for k,v in rows.items()},'contrasts':{}}
        for prefix in ('regional_','global_regional_'):
            for other in ('self','sham'):
                delta=100*(rows[prefix+'held'].double()-rows[prefix+other].double())
                result['contrasts'][prefix+'held_minus_'+other]={'points':delta.mean(0).tolist(),'macro':class_interval(delta[:,0],labels[select],seed=42)}
        save_tensor({'qi':qi[select],'pi':pi,'labels':labels[select],'rows':rows},root/(split+'.pt'))
        atomic_json(root/(split+'.json'),result)
        print(split,result['metrics'],flush=True)
        del fullbank,graph,bank


if __name__=='__main__':
    run()
