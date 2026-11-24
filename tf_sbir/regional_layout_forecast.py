"""Test whether an inferred common arrangement transfers observed semantics to a hidden region."""
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
from .geometry_word_flat import atomic_json,load_frozen_clip,revision,save_tensor,semantic_inputs
from .regional_observation_control import crop_features
from .regional_predictive_observation import cluster_effect,scene_bank
from .regional_scale_position_contrast import complete_windows
from .regional_source_assignment import half_correct
from .semantic_identification import greedy_pairs
from torch.nn import functional as F


METHODS=('square','unconditioned','one_anchor','any_bijection','registered')
CORNERS=(0,2,6,8)


def maps(device):
    grid=np.arange(4).reshape(2,2)
    group={tuple(np.rot90(x,k).reshape(-1).tolist()) for x in (grid,np.fliplr(grid)) for k in range(4)}
    assert len(group)==8
    return [torch.tensor(list(g),device=device) for g in (sorted(group),list(permutations(range(4))))]


def predict(sketch,photos,target,groups):
    # Use the other positions explicitly: the target sketch row is never an input of the forecasting equation.
    observed=[i for i in range(4) if i!=target]
    predictions={}
    for name,g in zip(('square','any_bijection'),groups):
        reference=photos[:,:,g[:,observed],:]
        overlap=(sketch[:,None,None,observed,:]*reference).sum(-1).clamp_min(1e-30)
        weights=overlap.log().sum(-1).softmax(2)
        outcomes=photos[:,:,g[:,target],:]
        predictions[name]=(weights[:,:,:,None]*outcomes).sum(2)
        if name=='square':
            single=(overlap/overlap.sum(2,keepdim=True)).mean(-1)
            predictions['one_anchor']=(single[:,:,:,None]*outcomes).sum(2)
    predictions['unconditioned']=photos.mean(2)
    predictions['registered']=photos[:,:,target]
    return predictions


def validate():
    torch.manual_seed(42)
    sketch=torch.randn(2,4,6).softmax(-1)
    photos=torch.randn(2,2,4,6).softmax(-1)
    groups=maps('cpu')
    for target in range(4):
        pred=predict(sketch,photos,target,groups)
        altered=sketch.clone()
        altered[:,target]=altered[:,target].flip(-1)
        transformed=photos[:,:,groups[0][2]]
        equiv=predict(sketch,transformed,target,groups)
        held=predict(altered,photos,target,groups)
        neutral=predict(torch.full_like(sketch,1/6),photos,target,groups)
        for k,value in pred.items():
            torch.testing.assert_close(value.sum(-1),torch.ones_like(value[...,0]))
            torch.testing.assert_close(value,held[k])
            if k!='registered':
                torch.testing.assert_close(value,equiv[k])
                torch.testing.assert_close(neutral[k],photos.mean(2))


def make_trials(qp,pp,qi,pi,gl,p,exclude):
    photos={(Path(pp[i]).parent.name,Path(pp[i]).stem):j for j,i in enumerate(pi.tolist())}
    groups=defaultdict(list)
    for i,qid in enumerate(qi.tolist()):
        path=Path(qp[qid])
        groups[photos[(path.parent.name,path.stem.rsplit('-',1)[0])]].append(i)
    refs=[r for r,draws in groups.items() if len(draws)>=2 and str(pp[int(pi[r])]) not in exclude]
    rng=np.random.default_rng(42)
    within=[]
    for cat in gl.unique().tolist():
        pairs=greedy_pairs([r for r in refs if int(gl[r])==cat],p)
        within.extend(pairs[i] for i in rng.choice(len(pairs),8,replace=False))
    across=greedy_pairs([i for pair in within for i in pair],p,gl.numpy())
    result=[]
    for mode,pairs in enumerate((within,across)):
        for pair,(a,b) in enumerate(pairs):
            for s,t in ((a,b),(b,a)):
                result.extend([i,s,t,mode,pair] for i in groups[s])
    return torch.tensor(result)


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    validate()
    w=Path('/content/sbir')
    root=w/'results/regional-layout-forecast'
    root.mkdir(parents=True,exist_ok=True)
    inputs=semantic_inputs(w,'sketchy')
    cfg,qp,qs,pp,gs,_,p,src,tgt,scale=inputs
    views=complete_windows(w,'sketchy',inputs)
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    excluded=set()
    for s in qs:
        r=torch.load(w/'results/semantic-identification'/(s+'.pt'),weights_only=True)
        assert torch.equal(gs[s][0],r['pi'])
        excluded.update(str(pp[i]) for i in r['pi'][r['trials'][:,1:3].unique()].tolist())
    groups=maps('cuda')
    for split,(qi,labels) in qs.items():
        pi,gl=gs[split]
        trials=make_trials(qp,pp,qi,pi,gl,F.normalize(p[pi].float()),excluded)
        used=trials[:,1:3].unique(sorted=True)
        paths=[pp[i] for i in pi[used].tolist()]
        assert not excluded.intersection(map(str,paths))
        heads,_=scene_bank(w,split,pp,pi,p,tgt,scale,vision)
        crops=crop_features(vision,paths)[:,CORNERS]
        bank=(scale*(crops @ tgt[heads].T)).softmax(-1).mean(2)
        rows=[]
        for batch in trials.split(64):
            ix=batch[:,0]
            poses=views['regions'][qi[ix]][:,CORNERS].float().cuda()
            obs=(scale*(poses @ src[heads].T)).softmax(-1).mean(2)
            locations=torch.searchsorted(used,batch[:,1:3].contiguous()).cuda()
            forecasts=[]
            for target in range(4):
                pred=predict(obs,bank[locations],target,groups)
                forecasts.append(torch.stack([(obs[:,None,target]*pred[k].clamp_min(1e-30).log()).sum(-1)
                                               for k in METHODS],1))
            rows.append(torch.stack(forecasts).mean(0).cpu())
        scores=torch.cat(rows).double()
        report={}
        for mode,c in enumerate(('same_category','different_category')):
            mask=trials[:,3]==mode
            tr=trials[mask].numpy()
            v=scores[mask].numpy()
            margins=v[:,:,0]-v[:,:,1]
            pairwise=[]
            for pid in np.unique(tr[:,4]):
                sub=tr[:,4]==pid
                a,b=np.unique(tr[sub,1])
                pairwise.append(half_correct(margins[sub&(tr[:,1]==a),None]+margins[sub&(tr[:,1]==b)][None]).mean((0,1)))
            pairwise=np.stack(pairwise)
            effects={}
            for j,k in enumerate(METHODS[1:],1):
                d=v[:,0]-v[:,j]
                effects[k]=dict(true=cluster_effect(d[:,0],tr[:,4]),foil=cluster_effect(d[:,1],tr[:,4]),
                    contrast=cluster_effect(d[:,0]-d[:,1],tr[:,4]),assignment_delta=cluster_effect(
                        pairwise[:,0]-pairwise[:,j],np.arange(len(pairwise))))
            report[c]=dict(queries=len(tr),pairs=len(pairwise),pair_accuracy=(margins>0).mean(0).tolist(),
                source_assignment=pairwise.mean(0).tolist(),primary_minus=effects)
        save_tensor(dict(qi=qi,pi=pi,trials=trials,labels=labels,scores=scores),root/(split+'.pt'))
        atomic_json(root/(split+'.json'),dict(commit=revision(),seed=42,device=torch.cuda.get_device_name(),
            excluded_photos=len(excluded),methods=list(METHODS),assessment=report))
        print(split,report['same_category']['source_assignment'],flush=True)


if __name__=='__main__':
    run()
