"""Hold frame uncertainty fixed while comparing the semantic consequences implied by two photographs."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from .benchmark import merge_inputs
from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .crossdomain_inputs import runtime
from .data import SKETCHY_SPLITS
from .encoding import load_frozen_clip,save_tensor
from .foundation import load_words
from .gcwe_transported_intervention import TEST,assign,encode,outcomes,posterior,structure
from .regional_predictive_observation import cluster_effect
from .semantic import read_words


def involutions(perm,actions):
    unit=torch.arange(4,device=perm.device)
    result=[]
    for a,b in TEST:
        h=next(h for h,row in zip(perm,actions) if row[a]==b and row[b]==a and torch.equal(h[h],unit))
        mapped=(perm[:,h,None]==perm.T[None]).all(1)
        assert (mapped.sum(1)==1).all()
        ix=mapped.long().argmax(1)
        assert torch.equal(ix[ix],torch.arange(len(ix),device=perm.device))
        result.append(ix)
    return result


def assess(weights,forecasts,partners):
    losses=[]
    qualities=[]
    for j,w in enumerate((*weights,(weights[0]+weights[1])*.5)):
        columns=[]
        radius=torch.stack([.25*(w-w[:,ix]).abs().sum(1) for ix in partners],1)
        for i,r in enumerate(forecasts):
            predicted=(r*w[:,None,:,None]).sum(2)
            z=predicted-.5
            assert (z.abs()<=radius[:,:,None]+2e-6).all()
            loss=(1-predicted).square().mean((1,2))
            signed=z.mean((1,2))
            cost=z.square().mean((1,2))
            torch.testing.assert_close(.25-loss,signed-cost,atol=2e-7,rtol=1e-5)
            columns.append(loss)
            if j==i:
                qualities.append(torch.stack((radius.mean(1),z.abs().mean((1,2)),signed,cost),1))
        losses.append(torch.stack(columns,1))
    return torch.stack(losses,1),torch.stack(qualities,1)


def summarize(meta,loss,quality):
    native=loss[:,1,1]-loss[:,0,0]
    shared=loss[:,2,1]-loss[:,2,0]
    semantic=.5*((loss[:,0,1]-loss[:,0,0])+(loss[:,1,1]-loss[:,1,0]))
    geo=.5*((loss[:,1,0]-loss[:,0,0])+(loss[:,1,1]-loss[:,0,1]))
    torch.testing.assert_close(native,semantic+geo,atol=1e-12,rtol=1e-10)
    interaction=(loss[:,0,1]-loss[:,0,0])-(loss[:,1,1]-loss[:,1,0])
    crossed=loss[:,0,1]-loss[:,1,0]
    def hit(x):
        return (x>1e-7).double()+.5*(x.abs()<=1e-7)
    output={}
    for m,group in enumerate(('same','different')):
        mask=meta[:,4]==m
        ids=meta[mask,5].numpy()
        measures={'native':native,'shared':shared,'crossed':crossed,'semantic_main':semantic,
            'geometry_main':geo,'interaction':interaction,'shared_advantage_change':shared-native,
            'native_accuracy':hit(native),'shared_accuracy':hit(shared),'shared_accuracy_change':hit(shared)-hit(native)}
        output[group]=dict(qualities=quality[mask].mean(0).tolist(),
            effects={k:cluster_effect(x[mask].numpy(),ids) for k,x in measures.items()})
    return output


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    w=Path('/content/sbir')
    root=w/'results/gcwe-shared-intervention-frame'
    root.mkdir(parents=True,exist_ok=True)
    cfg=runtime(w,'sketchy')
    paths,qs=merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    ph,gs=merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    refs={s:torch.load(w/'results/gcwe-transported-intervention'/(s+'.pt'),weights_only=True) for s in qs}
    qids=torch.cat([r['meta'][:,0] for r in refs.values()]).unique()
    pids=torch.cat([r['meta'][:,2:4].flatten() for r in refs.values()]).unique()
    src,tgt=[v.cuda() for v in load_words(cfg,read_words(cfg.vocabulary))]
    model=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda'))
    scale=model.logit_scale.exp().float()
    parts=structure(src.device)
    _,_,perm,square,action=parts
    partners=involutions(perm[square],action[square])
    photos=encode([ph[i] for i in pids.tolist()],False,model,parts)
    sketches=[encode([paths[i] for i in qids.tolist()],m,model,parts) for m in (False,True)]
    for s,rec in refs.items():
        assert torch.equal(rec['qi'],qs[s][0]) and torch.equal(rec['pi'],gs[s][0])
        meta=rec['meta']
        stored,results={},{}
        for key,(qe,qd,_) in zip(('native','flipped'),sketches):
            losses,qualities=[],[]
            for b in meta.split(10):
                qpos=torch.searchsorted(qids,b[:,0].contiguous())
                _,draws=outcomes(qe[qpos].cuda(),src,scale,b[:,0])
                weights,forecast=[],[]
                for ids in (b[:,2],b[:,3]):
                    pos=torch.searchsorted(pids,ids.contiguous())
                    logp=scale*(F.normalize(photos[0][pos,1:].cuda().float(),dim=-1)@tgt.T)
                    post,_=posterior(qd[qpos].cuda(),photos[1][pos].cuda(),scale,perm,square)
                    r=assign(logp,draws,action[square])
                    for j,ix in enumerate(partners):
                        torch.testing.assert_close(r[:,j,ix]+r[:,j],torch.ones_like(r[:,j]),atol=2e-6,rtol=0)
                    weights.append(post[:,square])
                    forecast.append(r)
                value,quality=assess(weights,forecast,partners)
                losses.append(value.cpu())
                qualities.append(quality.cpu())
            loss,quality=torch.cat(losses).double(),torch.cat(qualities).double()
            original=rec['rows'][key]['scores'][:,0,:,0]
            torch.testing.assert_close(torch.stack((loss[:,0,0],loss[:,1,1]),1),original,atol=2e-6,rtol=1e-6)
            stored[key]=dict(loss=loss,quality=quality)
            results[key]=summarize(meta,loss,quality)
        save_tensor(dict(meta=meta,rows=stored),root/(s+'.pt'))
        atomic_json(root/(s+'.json'),dict(commit=revision(),gpu=torch.cuda.get_device_name(),seed=42,
            quality_columns=['geometric_upper_bound','absolute_probability_margin','signed_probability_margin','squared_margin'],
            assessment=results))
        print(s,results['native']['same']['effects']['semantic_main'],flush=True)


if __name__=='__main__':
    run()
