"""Separate portable semantic direction from the normalization cost of transferring an interaction."""
from pathlib import Path

import numpy as np
import torch
from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .crossdomain_inputs import runtime
from .encoding import load_frozen_clip,save_tensor
from .foundation import load_words
from .semantic import read_words
from .benchmark import merge_inputs
from .data import SKETCHY_SPLITS
from .regional_predictive_observation import cluster_effect
from .regional_semantic_interactions import OPPOSITE,interactions


ARMS=('photo','wrong_parts','source_redraw','wrong_redraw_parts')


def decompose(base,target,change):
    # Subtracting the base mean of the increment makes the KL directly evaluable as log E_b exp(d-E_b d).
    logb=base.log_softmax(-1)
    b=logb.exp()
    centered=change-(b*change).sum(-1,keepdim=True)
    alignment=((target-b)*change).sum(-1)
    cost=(logb+centered).logsumexp(-1)
    gain=(target*((base+change).log_softmax(-1)-logb)).sum(-1)
    torch.testing.assert_close(alignment-cost,gain,atol=3e-6,rtol=2e-5)
    assert cost.min()>-2e-6
    return torch.stack((alignment,cost,gain),-1).mean(-2)


def analyze(raw,meta):
    report={}
    for group,name in enumerate(('same_category','different_category')):
        ok=meta[:,4]==group
        ids=meta[ok,5].numpy()
        x=raw[ok].numpy()
        arms={}
        for i,k in enumerate(ARMS):
            v=x[:,i]
            arms[k]={stat:{column:cluster_effect(values,ids) for column,values in
                     [('true',v[:,0,j]),('foil',v[:,1,j]),('contrast',v[:,0,j]-v[:,1,j])]}
                     for j,stat in enumerate(('alignment','cost','gain'))}
        report[name]=dict(sources=int(ok.sum()),arms=arms)
        for i,j,k in [(0,1,'photo_location'),(2,3,'redraw_location'),(2,0,'redraw_minus_photo')]:
            d=x[:,i,:,0]-x[:,j,:,0]
            report[name][k]=cluster_effect(d[:,0]-d[:,1],ids)
    return report


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    w=Path('/content/sbir')
    root=w/'results/regional-interaction-information'
    root.mkdir(parents=True,exist_ok=True)
    cfg=runtime(w,'sketchy')
    sk,qs=merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    ph,gs=merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    caches={}
    for path in (w/'cache/regional-semantic-interactions').glob('*.pt'):
        rec=torch.load(path,weights_only=True)
        kind=Path(rec['paths'][0]).parent.parent.name
        assert kind in ('sketch','photo') and all(Path(p).parent.parent.name==kind for p in rec['paths'])
        mapping=caches.setdefault(kind,{})
        mapping.update(zip(rec['paths'],rec['features']))
    src,tgt=[v.cuda() for v in load_words(cfg,read_words(cfg.vocabulary))]
    scale=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).logit_scale.exp().float()
    def lookup(ids,kind,paths):
        return torch.stack([caches[kind][paths[i]] for i in ids.tolist()]).cuda()
    for split in qs:
        initial=torch.load(w/'results/regional-semantic-interactions'/(split+'.pt'),weights_only=True)
        meta=initial['meta']
        assert torch.equal(initial['qi'],qs[split][0]) and torch.equal(initial['pi'],gs[split][0])
        sibling={int(row[2]):int(row[1]) for row in meta}
        assert all(sibling[int(row[2])]==int(row[1]) for row in meta)
        results,rows={},{}
        for context in ('center','rest'):
            batches=[]
            for batch in meta.split(12):
                qry=batch[:,0]
                base,obs,_,_=interactions(lookup(qry,'sketch',sk),context)
                base=scale*(base@src.T)
                prob=(scale*(obs@src.T)).softmax(-1)
                ends=[]
                for ids in [batch[:,2],batch[:,3]]:
                    pd=scale*(interactions(lookup(ids,'photo',ph),context)[2]@tgt.T)
                    sd=scale*(interactions(lookup(torch.tensor([sibling[i] for i in ids.tolist()]),'sketch',sk),context)[2]@src.T)
                    ends.append(torch.stack([decompose(base,prob,d) for d in (pd,pd[:,OPPOSITE],sd,sd[:,OPPOSITE])],1))
                batches.append(torch.stack(ends,2).cpu())
            raw=torch.cat(batches).double()
            ref=initial['rows'][context]['scores'][:,0]
            torch.testing.assert_close(raw[:,:2,:,2],ref[:,2:4]-ref[:,0:1],atol=1e-5,rtol=2e-5)
            rows[context]=raw
            results[context]=analyze(raw,meta)
        save_tensor(dict(meta=meta,rows=rows),root/(split+'.pt'))
        atomic_json(root/(split+'.json'),dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),
            arms=ARMS,columns=['alignment','cost','gain'],assessment=results))
        print(split,results['center']['same_category']['arms']['photo']['alignment']['true'],flush=True)


if __name__=='__main__':
    run()
