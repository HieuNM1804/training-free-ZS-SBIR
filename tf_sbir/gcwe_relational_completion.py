"""Infer a missing semantic role by joining two spatial observations from distinct gallery photos."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from itertools import combinations, permutations
import os
from pathlib import Path
from types import SimpleNamespace

from .gallery_corner_observations import read_corners
from .gcwe_semantic_prediction import log_score
from .gcwe_semantic_reveal import source_refs
from .geometry_word_flat import (CONFIG, COUNTS, Index, KD_FUNCTION_SHA, PrefixReplay, atomic_json, class_interval,
    load_frozen_clip, load_kd, make_word_bank, revision, save_tensor, score_rows, semantic_inputs)
from .regional_predictive_observation import scene_bank
from .regional_scale_position_contrast import complete_windows
import numpy as np
import torch
from torch.nn import functional as F


NAMES=('distinct','tree','same_donor','literal','no_relation','prior')
READOUTS=(*NAMES,'literal_direct')
CORNERS=(0,2,6,8)
AXES=[(a,b,c) for c in range(4) for a,b in combinations([j for j in range(4) if j!=c],2)]


def frames(device):
    grid=np.arange(4).reshape(2,2)
    square=sorted({tuple(np.rot90(v,k).flatten().tolist()) for v in (grid,np.fliplr(grid)) for k in range(4)})
    return torch.tensor(square,device=device),torch.tensor(list(permutations(range(4))),device=device)


def moments(photo,ma,mb,positions,g):
    """One common frame for the literal model, independently marginalized frames for the two edges."""
    a,b,c=positions
    # Compile the four small conditional sums without repeating the word vectors across all frames.
    weight=photo.new_full((*ma.shape[:2],4),1.)
    sums=[]
    for z in range(4):
        mask=g[:,c]==z
        a_lik=ma[:,:,g[mask,a]]
        b_lik=mb[:,:,g[mask,b]]
        sums.append(torch.stack((a_lik.mean(2),b_lik.mean(2),(a_lik*b_lik).mean(2)),-1))
    factors=torch.cat([weight[:,:,:,None],torch.stack(sums,2)],-1)
    return ((factors.transpose(-1,-2) @ photo)/4).double().unbind(2)


def norm(x):
    return x/x.sum(-1,keepdim=True).clamp_min(1e-300)


def compose(stats,valid):
    r,u,v,joint=(x*valid[:,:,None] for x in stats)
    rs,us,vs=(x.sum(1) for x in (r,u,v))
    tree=us*vs/rs.clamp_min(1e-300)
    numerator=(us*vs-(u*v).sum(1)).clamp_min(0)
    denom=(rs.square()-r.square().sum(1)).clamp_min(0)
    degenerate=denom<=1e-12*rs.square()
    distinct=torch.where(degenerate,tree,rs*numerator/denom.clamp_min(1e-300))
    same=(u*v/r.clamp_min(1e-300)).sum(1)
    local=torch.stack([norm(z) for z in (distinct,tree,same,joint.sum(1),rs)],1)
    if not torch.isfinite(local).all() or (local.sum(2)-1).abs().max()>1e-8:
        raise ValueError('Invalid inferred role distribution')
    donor_given_c=r/rs[:,None].clamp_min(1e-300)
    weights=local @ donor_given_c.transpose(1,2)
    direct=norm(joint.sum(2))
    # Literal minus factorized unnormalized mass equals prior times conditional covariance.
    covariance=(joint.sum(1)/rs.clamp_min(1e-300)-
                (us/rs.clamp_min(1e-300))*(vs/rs.clamp_min(1e-300)))
    torch.testing.assert_close(joint.sum(1)-tree,rs*covariance,atol=1e-11,rtol=1e-7)
    return local,weights,direct,degenerate


def check():
    gen=torch.Generator().manual_seed(42)
    photo=torch.rand((2,5,4,7),generator=gen,dtype=torch.double).softmax(-1)
    ma=torch.rand((2,5,4),generator=gen,dtype=torch.double)
    mb=torch.rand((2,5,4),generator=gen,dtype=torch.double)
    g=frames('cpu')[0]
    loc=(0,3,1)
    stats=moments(photo,ma,mb,loc,g)
    r,u,v,j=stats
    obs=compose(stats,torch.ones((2,5),dtype=torch.bool))
    triple=(photo[:,:,g[:,1]]*ma[:,:,g[:,0],None]*mb[:,:,g[:,3],None]).mean(2)
    torch.testing.assert_close(triple,j)
    aa,bb=(~torch.eye(5,dtype=torch.bool)).nonzero().T
    exact=(u[:,aa]*v[:,bb]).sum(1)/(r[:,aa]*r[:,bb]).sum(1)*r.sum(1)
    torch.testing.assert_close(norm(exact),obs[0][:,0])
    single=compose(moments(photo,ma,torch.ones_like(mb),loc,g),torch.ones((2,5),dtype=torch.bool))[0]
    torch.testing.assert_close(single[:,1],single[:,3])


def observations(local,crop_visual,bank_local,keys,part,groups,reference_side):
    n,k=local.shape[0],local.shape[-1]
    near=part[(crop_visual @ keys).topk(32,dim=-1).indices]
    combined=near.flatten(1)
    donor_weights=local.new_zeros(n,len(READOUTS),combined.shape[1])
    log_forecasts=local.new_zeros(n,2,len(NAMES),dtype=torch.double)
    cnt=local.new_zeros(2,dtype=torch.long)
    for a,b,c in AXES:
        first,second=near[:,a],near[:,b]
        valid=torch.cat((torch.ones_like(first,dtype=torch.bool),
                  ~(second[:,:,None]==first[:,None]).any(2)),1)
        ix=torch.cat((first,second),1)
        photo=bank_local[ix]
        ma=(photo*local[:,None,a,None,:]).sum(3)
        mb=(photo*local[:,None,b,None,:]).sum(3)
        stats=moments(photo,ma,mb,(a,b,c),groups[0])
        shuffled=moments(photo,ma,mb,(a,b,c),groups[1])
        joint=stats[3].sum(2).masked_fill(~valid,-1.)
        order=joint.argsort(dim=1,descending=True,stable=True)
        rank=order.argsort(dim=1)
        depleted=valid&(rank>=valid.sum(1,keepdim=True)//4)
        for stress,mask in enumerate((valid,depleted)):
            role,w,direct,deg=compose(stats,mask)
            sham,shw,_,_=compose(shuffled,mask)
            models=torch.cat((role[:,:4],sham[:,:1],role[:,4:5]),1)
            log_forecasts[:,stress]+=log_score(models,local[:,None,c,:])/len(AXES)
            cnt[stress]+=deg.sum()
            if not stress:
                all_w=torch.cat((w[:,:4],shw[:,:1],w[:,4:5],direct[:,None]),1).float()/len(AXES)
                donor_weights[:,:,a*32:(a+1)*32]+=all_w[:,:,:32]
                donor_weights[:,:,b*32:(b+1)*32]+=all_w[:,:,32:]
    if (donor_weights.sum(2)-1).abs().max()>2e-5:
        raise ValueError('Averaged donor law is invalid')
    return combined,donor_weights,log_forecasts*reference_side[:,None,None],cnt


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    check()
    w=Path('/content/sbir')
    root=w/'results/gcwe-relational-completion'
    root.mkdir(parents=True,exist_ok=True)
    inputs=semantic_inputs(w,'sketchy')
    cfg,qp,qs,pp,gs,q,p,src,tgt,scale=inputs
    views=complete_windows(w,'sketchy',inputs)
    gallery_crops=read_corners(w,pp)
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    previous=torch.cat([torch.load(w/'results'/fam/(s+'.pt'),weights_only=True)['qi'] for s in qs
        for fam in ('gcwe-semantic-prediction','gcwe-semantic-reveal')])
    for split,(qi,labels) in qs.items():
        pi,gl=gs[split]
        assert (len(qi),len(pi),len(gl.unique()))==COUNTS[split]
        refs,sources=source_refs(qp,pp,qi,pi)
        order=torch.randperm(len(qi),generator=torch.Generator().manual_seed(42))[6912:]
        eligible=~torch.isin(qi,previous)&torch.tensor([len(sources[int(v)])>1 for v in refs])
        ix=order[eligible[order]][:768].sort().values
        assert len(ix)==768
        parts=torch.randperm(len(pi),generator=torch.Generator().manual_seed(42)).cuda().tensor_split(2)
        folds=torch.zeros(len(pi),device='cuda',dtype=torch.long)
        folds[parts[1]]=1
        gallery=p[pi].cuda()
        keys=[gallery[part].T.contiguous() for part in parts]
        heads,_=scene_bank(w,split,pp,pi,p,tgt,scale,vision)
        bank_local=torch.cat([(scale*(gallery_crops[ids].cuda().float() @ tgt[heads].T)).softmax(-1).mean(2)
                              for ids in pi.split(192)])
        bank=make_word_bank(gallery,tgt,scale,192)
        endpoint=[bank[part].T.contiguous() for part in parts]
        graph=Index(SimpleNamespace(p=gallery),rgb_only=True)
        replay=PrefixReplay(kd,gl,split)
        classes=gl.cuda()
        groups=frames('cuda')
        all_metrics,local_rows,global_rows={},[],[]
        degenerate=torch.zeros(2,dtype=torch.long,device='cuda')
        with ThreadPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0)))) as pool:
            for batch in ix.split(12):
                ids=qi[batch]
                crops=views['regions'][ids][:,CORNERS].cuda().float()
                local=(scale*(crops @ src[heads].T)).softmax(-1).mean(2)
                visual=F.normalize(crops.mean(2),dim=-1)
                ref=refs[batch]
                scores=gallery.new_empty(len(batch),len(READOUTS),len(pi))
                predicted=gallery.new_zeros(len(batch),len(READOUTS),len(src))
                local_measure=gallery.new_zeros(len(batch),2,len(NAMES),dtype=torch.double)
                for t in (0,1):
                    side=folds[ref]==t
                    ns,weights,forecasts,deg=observations(local,visual,bank_local,keys[1-t],parts[1-t],groups,side)
                    words=weights @ bank[ns]
                    predicted+=words*side[:,None,None]
                    scores[:,:,parts[t]]=words @ endpoint[t]
                    local_measure+=forecasts
                    degenerate+=deg
                local_rows.append(local_measure.cpu())
                native=(scale*(torch.stack((q[ids],views['flipped'][ids].float()),1).cuda() @ src.T)).softmax(-1).mean(1)
                # Forecasts and retrieval scores are formed before reading labels or other drawings.
                sibling=[]
                for row,j in enumerate(batch.tolist()):
                    sib=qi[[s for s in sources[int(ref[row])] if s!=j]]
                    views_s=torch.stack((q[sib],views['flipped'][sib].float()),1).cuda()
                    sibling.append((scale*(views_s @ src.T)).softmax(-1).mean((0,1)))
                truth=torch.stack((torch.stack(sibling),bank[ref]),1)
                pred=torch.cat((predicted,native[:,None]),1)
                global_rows.append(log_score(pred[:,:,None],truth[:,None]).cpu())
                for j,key in enumerate((*READOUTS,'native')):
                    raw=native @ bank.T if key=='native' else scores[:,j].contiguous()
                    official=score_rows(graph.score(CONFIG,raw),labels[batch],classes,split,replay,pool)
                    all_metrics.setdefault(key,[]).append(official[:,:2])
        rows=torch.cat(local_rows)
        global_vals=torch.cat(global_rows).double()
        metrics={k:torch.cat(v).double() for k,v in all_metrics.items()}
        contrasts={}
        for setting,j in [('ordinary',0),('depleted',1)]:
            contrasts[setting]={}
            for i,k in enumerate(NAMES):
                delta=rows[:,j,0]-rows[:,j,i]
                contrasts[setting][k]=dict(mean=float(delta.mean()),classes=class_interval(delta,labels[ix],seed=42))
        save_tensor(dict(qi=qi[ix],pi=pi,labels=labels[ix],reference_ids=refs[ix],rows=metrics,
             local=rows,global_forecasts=global_vals),root/(split+'.pt'))
        atomic_json(root/(split+'.json'),dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),
            evaluator=KD_FUNCTION_SHA,queries=len(ix),gallery=len(pi),role_names=list(NAMES),readouts=[*READOUTS,'native'],
            global_targets=['separate_redraw','reference_photo'],degenerate=degenerate.tolist(),contrasts=contrasts,
            local_log_scores=rows.mean(0).tolist(),global_log_scores=global_vals.mean(0).tolist(),
            retrieval_percent={k:(100*v.mean(0)).tolist() for k,v in metrics.items()}))
        print(split,[(k,v['mean']) for k,v in contrasts['ordinary'].items()],flush=True)
        del graph,bank,endpoint


if __name__=='__main__':
    run()
