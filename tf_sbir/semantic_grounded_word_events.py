"""Marginalize word identity and local role jointly for full-gallery semantic retrieval."""
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, product
import hashlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

from .gallery_corner_observations import read_corners
from .gcwe_relational_completion import CORNERS
from .geometry_word_flat import (CONFIG, COUNTS, Index, KD_FUNCTION_SHA, PrefixReplay, atomic_json,
    class_interval, load_kd, make_word_bank, revision, save_tensor, score_rows, semantic_inputs)
from .regional_scale_position_contrast import complete_windows
import torch


PAIRS=list(combinations(range(4),2))
ARMS=('joint','native','candidate_words','turned','permuted_words')
ROTATIONS=((2,0,3,1),(1,3,0,2))


def odds_contrast(log_words,i,j):
    d=log_words[...,i,:]-log_words[...,j,:]
    return (.5*(d.unsqueeze(-1)-d.unsqueeze(-2))).tanh()


def check():
    rng=torch.Generator().manual_seed(42)
    rand=lambda shape:torch.randn(shape,generator=rng,dtype=torch.double)
    a,b=rand((2,5)).softmax(1)
    q,d=rand((2,4,5))
    h=a*b/(a@b)
    joint=0.
    for i,j in PAIRS:
        x,y=odds_contrast(q,i,j),odds_contrast(d,i,j)
        w=h[:,None]*h[None,:]
        assert abs((x*w).sum())<1e-12 and abs((y*w).sum())<1e-12
        torch.testing.assert_close(x,odds_contrast(q+rand((1,5))+rand((4,1)),i,j))
        torch.testing.assert_close(y,odds_contrast(d+rand((1,5))+rand((4,1)),i,j))
        orientation_prob=0.
        for u,v in product(range(5),repeat=2):
            qs=torch.stack((q[i,u]+q[j,v],q[i,v]+q[j,u])).softmax(0)
            ds=torch.stack((d[i,u]+d[j,v],d[i,v]+d[j,u])).softmax(0)
            orientation_prob+=a[u]*b[u]*a[v]*b[v]*(qs @ ds)
        joint+=orientation_prob/len(PAIRS)
        torch.testing.assert_close(orientation_prob,(a@b).square()*(1+(w*x*y).sum())/2)
    assert 0 < joint < (a@b).square()


def local_posterior(poses,text,scale):
    return (scale*(poses.float() @ text.T)).softmax(-1).mean(-2)


def score_block(q_global,q_local,bank,desc,head,prob,scale,tgt):
    """One gallery block against all queries; keep the vocabulary overlap fully marginalized."""
    ng,k=head.shape
    native=q_global @ bank.T
    d=local_posterior(desc,tgt,scale).gather(2,head[:,None].expand(-1,4,-1)).clamp_min(1e-30).log()
    perm=torch.arange(k,device=head.device).view(-1,4).roll(1,1).flatten()
    kernels={}
    for i,j in PAIRS:
        real=odds_contrast(d,i,j)
        rotated=sum(odds_contrast(d,g[i],g[j]) for g in ROTATIONS)/2
        wrong_word=real[:,perm][:,:,perm]
        kernels[i,j]=torch.stack((real,rotated,wrong_word),-1).reshape(ng,k*k,3).contiguous()
    results={k:[] for k in ARMS}
    masses=[]
    for first in range(0,len(q_global),48):
        sl=slice(first,first+48)
        g=native[sl].T.contiguous()
        h=q_global[sl][:,head].permute(1,0,2)*prob[:,None]/g[:,:,None].clamp_min(1e-30)
        h2=(h[:,:,:,None]*h[:,:,None,:]).flatten(2)
        p2=(prob[:,:,None]*prob[:,None,:]).flatten(1)
        # Gather once per block, preserving the candidate-specific ordering of exposed heads.
        query=q_local[sl][:,:,head].permute(2,0,1,3).contiguous()
        corr=g.new_zeros((*g.shape,3))
        prior_corr=torch.zeros_like(g)
        for i,j in PAIRS:
            x=odds_contrast(query,i,j).flatten(2)
            kernel=kernels[i,j]
            corr+=(x*h2).bmm(kernel)/len(PAIRS)
            prior_corr+=(x*p2[:,None,:]).bmm(kernel[:,:,:1]).squeeze(2)/len(PAIRS)
        mass=h.sum(2)
        if (mass>1.0001).any() or (corr.abs()>mass[:,:,None].square()+3e-5).any():
            raise ValueError('Matched-word correlation escaped its possible support')
        lookup={'joint':corr[:,:,0],'turned':corr[:,:,1],'permuted_words':corr[:,:,2],
                'candidate_words':prior_corr}
        results['native'].append(g.T)
        for name,v in lookup.items():
            results[name].append((g*(1+v).clamp_min(0).sqrt()).T)
        masses.append(mass.square().T)
    return {k:torch.cat(v) for k,v in results.items()},torch.cat(masses)


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    check()
    w=Path('/content/sbir')
    out=w/'results/semantic-grounded-word-events'
    out.mkdir(parents=True,exist_ok=True)
    inputs=semantic_inputs(w,'sketchy')
    _,_,qs,pp,gs,q,p,src,tgt,scale=inputs
    views=complete_windows(w,'sketchy',inputs)
    photo_corners=read_corners(w,pp)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    previous=torch.cat([torch.load(w/'results'/fam/(s+'.pt'),weights_only=True)['qi'] for s in qs for fam in
        ('gcwe-semantic-prediction','gcwe-semantic-reveal','gcwe-relational-completion')]).unique()
    for split,(qi,labels) in qs.items():
        pi,gl=gs[split]
        assert (len(qi),len(pi),len(gl.unique()))==COUNTS[split]
        order=torch.randperm(len(qi),generator=torch.Generator().manual_seed(42))[8192:]
        ix=order[~torch.isin(qi[order],previous)][:384].sort().values
        assert len(ix)==384
        ids=qi[ix]
        poses=torch.stack((q[ids],views['flipped'][ids].float()),1).cuda()
        q_global=local_posterior(poses,src,scale)
        q_local=torch.cat([local_posterior(views['regions'][b][:,CORNERS].cuda(),src,scale)
                           for b in ids.split(48)]).clamp_min(1e-30).log()
        gallery=p[pi].cuda()
        bank=make_word_bank(gallery,tgt,scale,128)
        graph=Index(SimpleNamespace(p=gallery),rgb_only=True)
        class_ids=gl.cuda()
        replay=PrefixReplay(kd,gl,split)
        scores={k:gallery.new_empty(len(ix),len(pi)) for k in ARMS}
        coverage=gallery.new_empty(len(ix),len(pi))
        for i in range(0,len(pi),96):
            sl=slice(i,i+96)
            photo=bank[sl]
            probs,head=photo.topk(32)
            by_arm,mass=score_block(q_global,q_local,photo,photo_corners[pi[sl]].cuda(),head,probs,scale,tgt)
            for k,v in by_arm.items():
                scores[k][:,sl]=v
            coverage[:,sl]=mass
        rows={}
        with ThreadPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0)))) as pool:
            for k,raw in scores.items():
                for mode,v in [('direct',raw),('diffused',graph.score(CONFIG,raw))]:
                    chunks=[]
                    for b in torch.arange(len(ix)).split(48):
                        chunks.append(score_rows(v[b],labels[ix[b]],class_ids,split,replay,pool)[:,:2])
                    rows[k+'/'+mode]=torch.cat(chunks).double()
        hit=labels[ix,None].cuda()==class_ids[None]
        top=scores['native'].topk(200).indices
        uncertainty={'mean_exposed_probability_all_pairs':float(coverage.mean()),
            'relevant':float(coverage[hit].mean()),'native_top200':float(coverage.gather(1,top).mean())}
        for mode in ('direct','diffused'):
            delta=rows['joint/'+mode]-rows['native/'+mode]
            uncertainty[mode+'_class_interval_map']=class_interval(100*delta[:,0],labels[ix],seed=42)
        checks={}
        for name in ARMS:
            dl=100*(rows['joint/diffused']-rows[name+'/diffused'])
            checks[name]=dict(change=dl.mean(0).tolist(),ap_class=class_interval(dl[:,0],labels[ix],seed=42))
        save_tensor(dict(qi=ids,pi=pi,labels=labels[ix],rows=rows),out/(split+'.pt'))
        atomic_json(out/(split+'.json'),dict(commit=revision(),seed=42,device=torch.cuda.get_device_name(),
            queries=len(ix),gallery=len(pi),evaluator=KD_FUNCTION_SHA,coverage=uncertainty,
            joint_minus=checks,metrics={k:(100*v.mean(0)).tolist() for k,v in rows.items()}))
        print(split,checks['native']['change'],flush=True)
        del bank,graph,scores,coverage


if __name__=='__main__':
    run()
