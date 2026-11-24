"""Restore direct word mass in the query-conditioned gallery distribution after RGB diffusion."""
import hashlib
import inspect
import os
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import torch

from .geometry_word_flat import (CONFIG, KD_FUNCTION_SHA, Index, PrefixReplay, atomic_json, class_interval,
    load_kd, make_word_bank, revision, save_tensor, score_rows, semantic_inputs, source_windows)


def normalize(value):
    return value/value.sum(1,keepdim=True).clamp_min(1e-30)


def project(direct,diffused,words):
    a,d=normalize(direct),normalize(diffused)
    ma,md=a@words,d@words
    projected=d*((ma/md.clamp_min(1e-30))@words.T)
    if (projected.sum(1)-1).abs().max()>2e-5 or (projected<0).any():
        raise ValueError('Lost mass in word projection')
    projected=normalize(projected)
    def tv(p,q):
        return (p-q).abs().sum(1,keepdim=True)/2
    motion=tv(projected,d)
    lexical=tv(ma,md)
    original=tv(a,d)
    if (motion-lexical).max()>2e-5 or (lexical-original).max()>2e-5:
        raise ValueError('Stochastic projection increased total variation')
    step=(motion/original.clamp_min(1e-30)).clamp(0,1)
    linear=torch.lerp(d,a,step)
    if (tv(linear,d)-motion).abs().max()>2e-5:
        raise ValueError('Control does not match the amount of mass moved')
    return {'raw':a,'diffused':d,'projected':projected,'linear':linear},torch.cat((original,lexical,motion,step),1)


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    w=Path('/content/sbir')
    root=w/'results/semantic-diffusion-word-mass'
    root.mkdir(exist_ok=True,parents=True)
    inputs=semantic_inputs(w,'sketchy')
    cfg,qp,qs,pp,gs,q,p,src,tgt,scale=inputs
    views=source_windows(w,'sketchy',inputs)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    for split,(qi,labels) in qs.items():
        pi,gl=gs[split]
        ids=torch.randperm(len(qi),generator=torch.Generator().manual_seed(42))[2304:5376].sort().values
        gallery=p[pi].cuda()
        graph=Index(SimpleNamespace(p=gallery),rgb_only=True)
        words=make_word_bank(gallery,tgt,scale,256)
        # The frequency-matched sham retains exactly the same photo-word bank at readout.
        order=words.mean(0).argsort(stable=True)
        peer=(torch.arange(len(order),device='cuda')^1).clamp_max(len(order)-1)
        paired=torch.empty_like(order).scatter_(0,order,order[peer])
        replay=PrefixReplay(kd,gl,split)
        rows={};stats=[]
        ref=torch.load(w/'results/geometry-word-flat-terms'/(split+'.pt'),weights_only=True,mmap=True)
        assert torch.equal(ref['qi'],qi) and torch.equal(ref['pi'],pi)
        classes=gl.cuda()
        with ThreadPoolExecutor(max_workers=min(3,len(os.sched_getaffinity(0)))) as pool:
            for block in ids.split(48):
                poses=torch.stack((q[qi[block]].cuda(),views['flipped'][qi[block]].cuda().float()),1)
                post=(scale*(poses@src.T)).softmax(2).mean(1)
                direct=post@words.T
                diff=graph.score(CONFIG,direct)
                panels,diagnostic=project(direct,diff,words)
                stats.append(diagnostic.cpu())
                d=panels['diffused']
                target=panels['raw']@words;source=d@words
                ratios=target/source.clamp_min(1e-30)
                sham=normalize(d*(ratios[:,paired]@words.T))
                panels['shuffled']=sham
                # Score original positive signals directly to reproduce the historical tie convention.
                panels['diffused']=diff
                panels['raw']=direct
                for name,val in panels.items():
                    if not val.isfinite().all():raise ValueError(name)
                    rows.setdefault(name,[]).append(score_rows(val,labels[block],classes,split,replay,pool))
        rows={k:torch.cat(v) for k,v in rows.items()}
        mae=(100*(rows['diffused'][:,:2]-ref['rows']['semantic_only'][ids,:2]).double().abs().mean(0))
        if mae.max()>.04:raise ValueError(f'Baseline mismatch: {mae}')
        stats=torch.cat(stats)
        result={'commit':revision(),'seed':42,'split':split,'queries':len(ids),'gallery':len(pi),
                'baseline_mae':mae.tolist(),'mean_tv_raw_word_projected_and_step':stats.double().mean(0).tolist(),
                'metrics':{k:(100*v[:,:2].double().mean(0)).tolist() for k,v in rows.items()},'contrasts':{}}
        for control in ('diffused','linear','shuffled','raw'):
            delta=rows['projected'].double()-rows[control].double()
            result['contrasts']['projected_minus_'+control]={
                'points':(100*delta[:,:2].mean(0)).tolist(),'macro_ap':class_interval(100*delta[:,0],labels[ids],seed=42),
                'top200_gained_lost':[int((delta[:,2]>0).sum()),int((delta[:,2]<0).sum())],
                'top1_gained_lost':[int((delta[:,3]>0).sum()),int((delta[:,3]<0).sum())]}
        save_tensor({'qi':qi[ids],'pi':pi,'labels':labels[ids],'diagnostics':stats,'rows':rows},root/(split+'.pt'))
        atomic_json(root/(split+'.json'),result)
        print(split,result['metrics'],flush=True)
        del graph,words


if __name__=='__main__':
    run()
