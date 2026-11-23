"""Measure whether a public word distinction survives a sketch-to-photo prompt change."""
import hashlib
import argparse
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
def run(mass_control=False):
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    w=Path('/content/sbir')
    root=w/'results'/('semantic-word-mass-control' if mass_control else 'semantic-word-contrast')
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
        paired_ids=torch.cat((left,right))
        word_freq=bank.mean(0)
        freq_ids=paired_ids[word_freq[paired_ids].argsort(stable=True)]
        freq_a,freq_b=freq_ids[::2],freq_ids[1::2]
        movement=[]
        replay=PrefixReplay(kd,gl,split)
        with ThreadPoolExecutor(max_workers=min(3,len(os.sched_getaffinity(0)))) as pool:
            for block in ids.split(64):
                poses=torch.stack((q[qi[block]].cuda(),views['flipped'][qi[block]].cuda().float()),1)
                post=(scale*(poses@src.T)).softmax(2).mean(1)
                mass.append((post[:,left]+post[:,right]).sum(1).cpu())
                if mass_control:
                    near=contract(post,left,right,-1.)
                    freq=contract(post,freq_a,freq_b,-1.)
                    near_tv=.5*(near-post).abs().sum(1,keepdim=True)
                    freq_tv=.5*(freq-post).abs().sum(1,keepdim=True)
                    common=torch.minimum(near_tv,freq_tv)
                    panels={'original':post, 'near':post+(near-post)*(common/near_tv.clamp_min(1e-30)),
                            'frequency':post+(freq-post)*(common/freq_tv.clamp_min(1e-30))}
                    for candidate in (panels['near'],panels['frequency']):
                        if (.5*(candidate-post).abs().sum(1,keepdim=True)-common).abs().max()>1e-5:
                            raise ValueError('Different query probability moved')
                    movement.append(common.cpu())
                else:
                    panels={key:contract(post,left,right,rho) for key,rho in modes.items()}
                for key,candidate in panels.items():
                    raw=candidate @ bank.T
                    for label,value in [('direct',raw),('diffused',graph.score(CONFIG,raw))]:
                        k=label+'_'+key
                        rows.setdefault(k,[]).append(score_rows(value,labels[block],gl.cuda(),split,replay,pool)[:,:2])
        rows={k:torch.cat(v) for k,v in rows.items()}
        contrasts={}
        for form in ('direct','diffused'):
            for control in (('original','frequency') if mass_control else ('original','text_corr','mean_corr','shuffled_corr','coarse')):
                a=form+('_near' if mass_control else '_gallery_corr');b=form+'_'+control
                delta=100*(rows[a].double()-rows[b].double())
                contrasts[a+' minus '+b]={'points':delta.mean(0).tolist(),'macro':class_interval(delta[:,0],labels[ids],seed=42)}
        record={'commit':revision(),'seed':42,'queries':len(ids),'gallery':len(pi),'pair_count':len(left),
                'mean_paired_query_mass':float(torch.cat(mass).mean()),
                'gallery_rho_quantiles':corr.quantile(torch.tensor([0,.1,.5,.9,1],device='cuda')).tolist(),
                'metrics':{k:(100*v.double().mean(0)).tolist() for k,v in rows.items()},'contrasts':contrasts}
        if mass_control:
            record['mean_equal_tv']=float(torch.cat(movement).mean())
            record['mean_text_pair_cosine']={k:float((F.normalize(src[a]+tgt[a])*F.normalize(src[b]+tgt[b])).sum(1).mean())
                                            for k,a,b in [('near',left,right),('frequency',freq_a,freq_b)]}
        save_tensor({'qi':qi[ids],'pi':pi,'labels':labels[ids],'rows':rows,'left':left.cpu(),'right':right.cpu(),
                     'rho':corr.cpu(),'text_rho':text_corr.cpu(),'query_mass':torch.cat(mass)},root/(split+'.pt'))
        atomic_json(root/(split+'.json'),record)
        print(split,record['pair_count'],record['metrics']['diffused_original'],flush=True)
        del bank,graph


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mass-control',action='store_true')
    run(parser.parse_args().mass_control)
