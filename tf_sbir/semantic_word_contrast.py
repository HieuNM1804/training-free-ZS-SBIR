"""Measure whether a public word distinction survives a sketch-to-photo prompt change."""
import hashlib
import inspect
import os
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from .geometry_word_flat import (CHECKPOINT, CONFIG, KD_FUNCTION_SHA, Index, PrefixReplay, atomic_json,
    class_interval, load_kd, make_word_bank, save_tensor, score_rows, semantic_inputs, source_windows, revision)


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


def reliability(gallery,source,target,left,right):
    s=(source[left]-source[right]).double()
    t=(target[left]-target[right]).double()
    centered=gallery.double()-gallery.double().mean(0)
    cov=centered.T@centered/len(centered)
    numerator=(s*(t@cov)).sum(1)
    denom=((s*(s@cov)).sum(1)*(t*(t@cov)).sum(1)).clamp_min(0).sqrt()
    # Undefined contrast is unidentifiable; equal mass is the neutral setting.
    corr=torch.where(denom>0,numerator/denom.clamp_min(1e-30),0.).clamp(-1,1)
    return corr.float(),F.cosine_similarity(s,t).float()


def contract(post,left,right,rho):
    delta=(post[:,left]-post[:,right])*(1-rho)/2
    moved=post.clone()
    moved[:,left]=post[:,left]-delta
    moved[:,right]=post[:,right]+delta
    if moved.min() < -1e-7 or (moved.sum(1)-post.sum(1)).abs().max()>1e-5:
        raise ValueError('Pair contraction did not preserve probabilities')
    return moved


@torch.inference_mode()
def run():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    w=Path('/content/sbir')
    root=w/'results/semantic-word-contrast'
    root.mkdir(parents=True,exist_ok=True)
    inputs=semantic_inputs(w,'sketchy')
    cfg,qp,qs,pp,gs,q,p,src,tgt,scale=inputs
    views=source_windows(w,'sketchy',inputs)
    left,right=paired_words(src,tgt)
    if len(left)==0:raise ValueError('No reciprocal neighbors')
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    for split,(qi,labels) in qs.items():
        pi,gl=gs[split]
        ids=torch.randperm(len(qi),generator=torch.Generator().manual_seed(42))[768:2304].sort().values
        gallery=p[pi].cuda()
        bank=make_word_bank(gallery,tgt,scale,256)
        graph=Index(SimpleNamespace(p=gallery),rgb_only=True)
        corr,text_corr=reliability(gallery,src,tgt,left,right)
        perm=torch.randperm(len(corr),generator=torch.Generator().manual_seed(42)).cuda()
        modes={'original':1.,'gallery_corr':corr,'text_corr':text_corr,'mean_corr':corr.mean(),
               'shuffled_corr':corr[perm],'coarse':0.,'swapped':-1.}
        rows={}
        mass=[]
        replay=PrefixReplay(kd,gl,split)
        with ThreadPoolExecutor(max_workers=min(3,len(os.sched_getaffinity(0)))) as pool:
            for block in ids.split(64):
                poses=torch.stack((q[qi[block]].cuda(),views['flipped'][qi[block]].cuda().float()),1)
                post=(scale*(poses@src.T)).softmax(2).mean(1)
                mass.append((post[:,left]+post[:,right]).sum(1).cpu())
                for key,rho in modes.items():
                    raw=contract(post,left,right,rho) @ bank.T
                    for label,value in [('direct',raw),('diffused',graph.score(CONFIG,raw))]:
                        k=label+'_'+key
                        rows.setdefault(k,[]).append(score_rows(value,labels[block],gl.cuda(),split,replay,pool)[:,:2])
        rows={k:torch.cat(v) for k,v in rows.items()}
        contrasts={}
        for form in ('direct','diffused'):
            for control in ('original','text_corr','mean_corr','shuffled_corr','coarse'):
                a=form+'_gallery_corr';b=form+'_'+control
                delta=100*(rows[a].double()-rows[b].double())
                contrasts[a+' minus '+b]={'points':delta.mean(0).tolist(),'macro':class_interval(delta[:,0],labels[ids],seed=42)}
        record={'commit':revision(),'seed':42,'queries':len(ids),'gallery':len(pi),'pair_count':len(left),
                'mean_paired_query_mass':float(torch.cat(mass).mean()),
                'gallery_rho_quantiles':corr.quantile(torch.tensor([0,.1,.5,.9,1],device='cuda')).tolist(),
                'metrics':{k:(100*v.double().mean(0)).tolist() for k,v in rows.items()},'contrasts':contrasts}
        save_tensor({'qi':qi[ids],'pi':pi,'labels':labels[ids],'rows':rows,'left':left.cpu(),'right':right.cpu(),
                     'rho':corr.cpu(),'text_rho':text_corr.cpu(),'query_mass':torch.cat(mass)},root/(split+'.pt'))
        atomic_json(root/(split+'.json'),record)
        print(split,record['pair_count'],record['metrics']['diffused_gallery_corr'],flush=True)
        del bank,graph


if __name__=='__main__':
    run()
