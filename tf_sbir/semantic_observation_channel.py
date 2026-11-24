"""Test the cross-region predictions of a content-dependent semantic reporting channel."""
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from .gallery_corner_observations import read_corners
from .gcwe_relational_completion import CORNERS,frames
from .geometry_word_flat import atomic_json,load_frozen_clip,revision,save_tensor,semantic_inputs
from .regional_predictive_observation import cluster_effect,scene_bank
from .regional_scale_position_contrast import complete_windows
from .semantic_identification import greedy_pairs
import torch
from torch.nn import functional as F
import numpy as np


PAIRS=list(combinations(range(4),2))
NAMES=('inferred','unseen_context','reverse_context','any_bijection','registered',
       'visibility','uncorrected','mean_sketch','swapped_visibility')


def trials_for_split(qp,pp,qi,pi,gl,p,excluded):
    lookup={(Path(pp[i]).parent.name,Path(pp[i]).stem):j for j,i in enumerate(pi.tolist())}
    groups=defaultdict(list)
    for i,qid in enumerate(qi.tolist()):
        path=Path(qp[qid])
        groups[lookup[(path.parent.name,path.stem.rsplit('-',1)[0])]].append(i)
    pool=[j for j,items in groups.items() if len(items)>1 and str(pp[int(pi[j])]) not in excluded]
    rng=np.random.default_rng(42)
    within=[]
    for c in gl.unique().tolist():
        pairs=greedy_pairs([j for j in pool if gl[j].item()==c],p)
        within.extend(pairs[j] for j in rng.choice(len(pairs),size=min(6,len(pairs)),replace=False))
    across=greedy_pairs([k for v in within for k in v],p,gl.numpy())
    trial=[[s,j,k,m,pid] for m,pairs in enumerate((within,across)) for pid,(j1,j2) in enumerate(pairs)
           for j,k in ((j1,j2),(j2,j1)) for s in groups[j]]
    return torch.tensor(trial)


def channel_forecasts(q,b):
    lq=q.clamp_min(1e-30).log()
    lb=b.clamp_min(1e-30).log()
    shifts=lq[:,None]-lb
    shared=(shifts.sum(2,keepdim=True)-shifts)/3
    other_q=((lq.sum(1,keepdim=True)-lq)/3).softmax(-1)
    return torch.stack(((lb+shared).softmax(-1),b,other_q[:,None].expand_as(b),
                        (lb+shared.flip(1)).softmax(-1)),dim=2)


def assignment_kernel(logb,i,j):
    d=logb[:,:,i]-logb[:,:,j]
    return (d.unsqueeze(-1)-d.unsqueeze(-2)).sigmoid()


def assignment_scores(q,b,group,full=False):
    n=len(q)
    lb=b.clamp_min(1e-30).log()
    result=[]
    for i,j in PAIRS:
        k,l=(z for z in range(4) if z not in (i,j))
        ctx=assignment_kernel(lb,group[:,i],group[:,j])
        pair=q[:,i,:,None]*q[:,j,None,:]
        weight=2*(ctx*pair[:,None,None]).sum((-1,-2))/len(group)
        if not torch.allclose(weight.sum(2),torch.ones_like(weight[:,:,0]),atol=3e-6):
            raise ValueError('Symmetric orientation design failed normalization')
        kernel=assignment_kernel(lb,group[:,k],group[:,l])
        def predict(w):
            return (w[:,:,:,None,None]*kernel).sum(2)
        inferred=predict(weight)
        # For the full group, exchanging only the two unseen locations leaves every
        # observed likelihood unchanged and complements the forecast exactly.
        if len(group)==24:
            torch.testing.assert_close(inferred,torch.full_like(inferred,.5),atol=4e-6,rtol=0.)
            inferred=torch.full_like(inferred,.5)
        tested=q[:,k,:,None]*q[:,l,None,:]
        def evaluate(prob):
            return (tested[:,None]*prob.clamp_min(1e-30).log()).sum((-1,-2))
        vals=[evaluate(inferred)]
        if full:
            neutral=kernel.mean(2)
            torch.testing.assert_close(neutral,torch.full_like(neutral,.5),atol=4e-6,rtol=0.)
            neutral=torch.full_like(neutral,.5)
            # A reversed observed assignment complements each orientation's likelihood.
            reverse=predict(2./len(group)-weight)
            fixed=assignment_kernel(lb,k,l)
            vals.extend([evaluate(neutral),evaluate(reverse),evaluate(fixed)])
        result.append(torch.stack(vals,1))
    return torch.stack(result).mean(0)


def measure(q,b,groups):
    square=assignment_scores(q,b,groups[0],True)
    arbitrary=assignment_scores(q,b,groups[1])
    assign=torch.cat((square[:,:3],arbitrary,square[:,3:4]),1)
    forecasts=channel_forecasts(q,b)
    local=(q[:,None,None]*forecasts.clamp_min(1e-30).log()).sum(-1).mean(-1).transpose(1,2)
    return torch.cat((assign,local),1)


def check():
    gen=torch.Generator().manual_seed(42)
    b=torch.randn(3,2,4,6,generator=gen,dtype=torch.double).softmax(-1)
    eta=torch.rand(3,2,1,6,generator=gen,dtype=torch.double)
    modified=b*eta
    modified/=modified.sum(-1,keepdim=True)
    q=modified[:,0]
    groups=frames('cpu')
    forecasts=channel_forecasts(q,b)
    torch.testing.assert_close(forecasts[:,0,0],q)
    for g in groups:
        base=assignment_scores(q,b,g,True)
        torch.testing.assert_close(base,assignment_scores(q,modified,g,True))
    equal=b[:,:,:1].expand_as(b)
    torch.testing.assert_close(assignment_scores(q,equal,groups[0]),torch.full((3,1,2),-np.log(2.),dtype=torch.double))


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    check()
    w=Path('/content/sbir')
    root=w/'results/semantic-observation-channel'
    root.mkdir(parents=True,exist_ok=True)
    inputs=semantic_inputs(w,'sketchy')
    cfg,qp,qs,pp,gs,_,p,src,tgt,scale=inputs
    views=complete_windows(w,'sketchy',inputs)
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    crops=read_corners(w,pp)
    exclude=set()
    for s in qs:
        for fam in ('semantic-identification','regional-layout-forecast'):
            rec=torch.load(w/'results'/fam/(s+'.pt'),weights_only=True,mmap=True)
            assert torch.equal(gs[s][0],rec['pi'])
            exclude.update(str(pp[i]) for i in rec['pi'][rec['trials'][:,1:3].unique()].tolist())
    groups=frames('cuda')
    for split,(qi,labels) in qs.items():
        pi,gl=gs[split]
        trials=trials_for_split(qp,pp,qi,pi,gl,F.normalize(p[pi].float()),exclude)
        used=trials[:,1:3].unique(sorted=True)
        assert not any(str(pp[i]) in exclude for i in pi[used].tolist())
        heads,_=scene_bank(w,split,pp,pi,p,tgt,scale,vision)
        gallery=(scale*(crops[pi[used]].cuda().float() @ tgt[heads].T)).softmax(-1).mean(2)
        data=[]
        for batch in trials.split(24):
            obs=(scale*(views['regions'][qi[batch[:,0]]][:,CORNERS].cuda().float() @ src[heads].T)).softmax(-1).mean(2)
            locations=torch.searchsorted(used,batch[:,1:3].contiguous()).cuda()
            data.append(measure(obs,gallery[locations],groups).cpu())
        rows=torch.cat(data).double()
        report={}
        for mode,name in enumerate(('same_category','different_category')):
            mask=trials[:,3]==mode
            tr=trials[mask].numpy()
            v=rows[mask].numpy()
            margins=v[:,:,0]-v[:,:,1]
            # Unconditioned assignment is analytically 1/2 for every word pair and candidate.
            margins[:,1]=0
            margins[:,3]=0
            margins[:,7]=0
            pairs=np.unique(tr[:,4])
            association=[]
            for pid in pairs:
                sub=tr[:,4]==pid
                a,b=np.unique(tr[sub,1])
                m=margins[sub&(tr[:,1]==a),None]+margins[sub&(tr[:,1]==b)][None]
                association.append(((m>0)+.5*(m==0)).mean((0,1)))
            contrasts={}
            for primary,control in [(0,1),(0,2),(0,3),(0,4),(5,6),(5,7),(5,8)]:
                delta=v[:,primary]-v[:,control]
                contrasts[NAMES[primary]+'-'+NAMES[control]]=dict(true=cluster_effect(delta[:,0],tr[:,4]),
                    foil=cluster_effect(delta[:,1],tr[:,4]),contrast=cluster_effect(margins[:,primary]-margins[:,control],tr[:,4]))
            report[name]=dict(queries=len(tr),pairs=len(pairs),single_query=((margins>0)+.5*(margins==0)).mean(0).tolist(),
                assignment=np.mean(association,axis=0).tolist(),mean_log_score=v.mean(0).tolist(),contrasts=contrasts)
        save_tensor(dict(qi=qi,pi=pi,trials=trials,labels=labels,scores=rows),root/(split+'.pt'))
        atomic_json(root/(split+'.json'),dict(commit=revision(),gpu=torch.cuda.get_device_name(),seed=42,
            excluded_sources=len(exclude),methods=list(NAMES),assessment=report))
        print(split,report['same_category']['assignment'],flush=True)


if __name__=='__main__':
    run()
