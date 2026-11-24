"""Use held-apart regional observations to choose how much lexical mass to restore after diffusion."""
import hashlib
import inspect
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .geometry_word_flat import (CHECKPOINT, CONFIG, GLOBAL_WEIGHT, REGIONAL_WEIGHT, CandidateRegions,
    KD_FUNCTION_SHA, Index, PrefixReplay, atomic_json, class_interval, digest, index, load_kd, log_score,
    make_word_bank, revision, save_tensor, score_rows, semantic_inputs, source_windows)
from .semantic_diffusion_word_mass import project


def regional_view(poses,crops,source,heads,compact,scale,raw):
    reader=CandidateRegions(poses[:,0],poses[:,1],crops[:,:,0],crops[:,:,1],source,heads,compact,
                            source.new_empty((0,len(source))),raw)
    reader.prepare(scale)
    l=reader.l
    attention=(reader.prior.log()+reader.logl-log_score(l.mean(-1,keepdim=True))).softmax(2)
    mean=attention.mean(-1,keepdim=True)
    separate=[]
    for sl in (slice(0,9),slice(9,13)):
        m=mean[:,:,sl]
        separate.append(attention[:,:,sl].sum(2,keepdim=True)*m/m.sum(2,keepdim=True).clamp_min(1e-30))
    def read(weights):
        return log_score((weights*l).sum(2).mean(1))
    return read(attention),read(torch.cat(separate,2))


def restore_fraction(witness,ma,md):
    diff=(ma-md).double()
    energy=diff.square().sum(1,keepdim=True)
    step=((witness.double()-md)*diff).sum(1,keepdim=True)/energy.clamp_min(1e-60)
    return torch.where(energy>0,step.clamp(0,1),0.).float()


@torch.inference_mode()
def run():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    w=Path('/content/sbir')
    root=w/'results/semantic-regional-mass-witness'
    root.mkdir(exist_ok=True,parents=True)
    inputs=semantic_inputs(w,'sketchy')
    cfg,qp,qs,pp,gs,q,p,src,tgt,scale=inputs
    views=source_windows(w,'sketchy',inputs)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    for split,(qi,labels) in qs.items():
        pi,gl=gs[split]
        ids=torch.randperm(len(qi),generator=torch.Generator().manual_seed(42))[5376:8448].sort().values
        gallery=p[pi].cuda()
        paths=[str(pp[i]) for i in pi.tolist()]
        stamp=digest({'gallery':paths,'model':CHECKPOINT,'source':'regional-64/RGB16'})
        heads,compact=index(w,split,gallery,tgt,scale,stamp[:16])
        words=make_word_bank(gallery,tgt,scale,256)
        graph=Index(SimpleNamespace(p=gallery),rgb_only=True)
        gallery_perm=torch.randperm(len(pi),generator=torch.Generator().manual_seed(42)).cuda()
        ref=torch.load(w/'results/geometry-word-flat-terms'/(split+'.pt'),mmap=True,weights_only=True)
        assert torch.equal(ref['qi'],qi) and torch.equal(ref['pi'],pi)
        replay=PrefixReplay(kd,gl,split);classes=gl.cuda()
        rows={};steps=[]
        for_pool=min(3,len(os.sched_getaffinity(0)))
        with ThreadPoolExecutor(max_workers=for_pool) as pool:
            for block in ids.split(24):
                ix=qi[block]
                poses=torch.stack((q[ix].cuda(),views['flipped'][ix].cuda().float()),1)
                post=(scale*(poses@src.T)).softmax(2).mean(1)
                raw=post@words.T
                diffuse=graph.score(CONFIG,raw)
                native,witness=regional_view(poses,views['regions'][ix].cuda(),src,heads,compact,scale,raw)
                panels,tv,(ma,md)=project(raw,diffuse,words)
                region_dist=witness.softmax(1)
                frac=restore_fraction(region_dist@words,ma,md)
                sham=restore_fraction(region_dist[:,gallery_perm]@words,ma,md)
                steps.append(torch.cat((frac,sham,tv),1).cpu())
                d,h=panels['diffused'],panels['projected']
                mixtures={'native':d,'adaptive':torch.lerp(d,h,frac),'sham':torch.lerp(d,h,sham),
                          'matched_raw':torch.lerp(d,panels['linear'],frac)}
                for beta in (.25,.5,.75,1.):
                    mixtures[f'fixed_{beta:g}']=torch.lerp(d,h,beta)
                for name,score in mixtures.items():
                    val=GLOBAL_WEIGHT*log_score(score)+REGIONAL_WEIGHT*native
                    rows.setdefault(name,[]).append(score_rows(val,labels[block],classes,split,replay,pool))
        rows={k:torch.cat(v) for k,v in rows.items()}
        stats=torch.cat(steps)
        mismatch=100*(rows['native'][:,:2]-ref['rows']['without_geometry'][ids,:2]).double().abs().mean(0)
        if mismatch.max()>.04:raise ValueError(f'Native replay changed: {mismatch}')
        contrasts={}
        for name in rows:
            delta=rows['adaptive'].double()-rows[name].double()
            contrasts[name]={'points':(100*delta[:,:2].mean(0)).tolist(),
                'macro':class_interval(100*delta[:,0],labels[ids],seed=42),
                'top200_gained_lost':[int((delta[:,2]>0).sum()),int((delta[:,2]<0).sum())]}
        record={'commit':revision(),'seed':42,'queries':len(ids),'gallery':len(pi),
                'mean_fraction':stats[:,:2].mean(0).tolist(),
                'endpoint_fractions':[(stats[:,0]==x).float().mean().item() for x in (0,1)],
                'baseline_mae_points':mismatch.tolist(),'metrics':{k:(100*v[:,:2].double().mean(0)).tolist() for k,v in rows.items()},
                'adaptive_minus_other':contrasts}
        save_tensor({'qi':qi[ids],'pi':pi,'labels':labels[ids],'steps':stats,'rows':rows},root/(split+'.pt'))
        atomic_json(root/(split+'.json'),record)
        print(split,record['metrics']['adaptive'],flush=True)
        del graph,words


if __name__=='__main__':
    run()
