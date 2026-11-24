"""A photo proposes a set of semantic interactions, a subset of sketch words chooses coordinates."""
from pathlib import Path

import numpy as np
import torch
from .benchmark import merge_inputs
from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .crossdomain_inputs import runtime
from .data import SKETCHY_SPLITS
from .encoding import load_frozen_clip,save_tensor
from .foundation import load_words
from .regional_predictive_observation import cluster_effect
from .regional_semantic_interactions import interactions,log_score
from .semantic import read_words


ARMS=('base','single','photo_interactions','photo_additive','query_additive','query_plus_interactions','query_plus_additive')


def coordinates(response,bank):
    """Rows are basis functions, columns are revealed words. Centering eliminates arbitrary logit offsets."""
    basis=bank.double()
    basis=basis-basis.mean(-1,keepdim=True)
    target=response.double()
    target=target-target.mean(-1,keepdim=True)
    gram=basis@basis.transpose(1,2)
    return ((target@basis.transpose(1,2))@torch.linalg.pinv(gram,rtol=1e-10,hermitian=True)).float()


def validate():
    g=torch.Generator().manual_seed(42)
    y=torch.randn(2,4,22,generator=g)
    basis=torch.randn(2,6,22,generator=g)
    test=torch.randn(2,6,13,generator=g)
    direct=torch.linalg.lstsq((basis-basis.mean(-1,keepdim=True)).transpose(1,2).double(),
                             (y-y.mean(-1,keepdim=True)).transpose(1,2).double()).solution.transpose(1,2).float()
    x=coordinates(y,basis)
    torch.testing.assert_close(x,direct)
    idx=[3,1,0,5,2,4]
    torch.testing.assert_close(x@test,coordinates(y+4,basis[:,idx]+2)@test[:,idx],atol=1e-5,rtol=1e-5)


def forecast(base,truth,other,query_basis,photo_basis,photo_main,src,tgt,scale,folds):
    # The target word values from the joint sketch intervention are accessed through the seen split in the solver.
    result=[]
    for seen,held in (folds,folds[::-1]):
        observed=(truth-base)@src[seen].T
        base_test=scale*(base@src[held].T)
        qs=query_basis@src[seen].T
        qt=query_basis@src[held].T
        ps=photo_basis@tgt[seen].T
        pt=photo_basis@tgt[held].T
        ms=photo_main@tgt[seen].T
        mt=photo_main@tgt[held].T
        # Each row has its own corresponding one-dimensional photo intervention.
        local=ps.double()-ps.double().mean(-1,keepdim=True)
        yy=observed.double()-observed.double().mean(-1,keepdim=True)
        coef=(yy*local).sum(-1)/local.square().sum(-1).clamp_min(1e-20)
        estimates=[base_test,base_test+scale*coef.float()[:,:,None]*pt]
        for s,t in [(ps,pt),(ms,mt),(qs,qt),(torch.cat((qs,ps),1),torch.cat((qt,pt),1)),
                    (torch.cat((qs,ms),1),torch.cat((qt,mt),1))]:
            estimates.append(base_test+scale*(coordinates(observed,s)@t))
        t1=(scale*(truth@src[held].T)).softmax(-1)
        t2=(scale*(other@src[held].T)).softmax(-1)
        result.append(torch.stack([torch.stack([log_score(pred,t) for pred in estimates],1) for t in (t1,t2)],1))
    return (result[0]+result[1])*.5


def analyze(meta,scores):
    out={}
    for name,m in [('same',0),('different',1)]:
        mask=meta[:,4]==m
        ids=meta[mask,5].numpy()
        for v,view in enumerate(('own','redraw')):
            x=scores[mask,v].numpy()
            margin=x[:,:,0]-x[:,:,1]
            primary=ARMS.index('query_plus_interactions')
            group=dict(accuracy=((margin>1e-6)+.5*(np.abs(margin)<=1e-6)).mean(0).tolist(),
                       mean_log_score=x.mean(0).tolist(),n=len(x),comparisons={})
            for j,control in enumerate(ARMS):
                if j==primary:continue
                group['comparisons'][control]=dict(true=cluster_effect(x[:,primary,0]-x[:,j,0],ids),
                    discrimination=cluster_effect(margin[:,primary]-margin[:,j],ids))
            out[name+'/'+view]=group
    return out


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    validate()
    w=Path('/content/sbir')
    root=w/'results/regional-conditional-repertoire'
    root.mkdir(parents=True,exist_ok=True)
    cfg=runtime(w,'sketchy')
    paths,qs=merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    photos,gs=merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    cache={}
    for file in (w/'cache/regional-semantic-interactions').glob('*.pt'):
        rec=torch.load(file,weights_only=True)
        cache.update(zip(rec['paths'],rec['features']))
    src,tgt=[v.cuda() for v in load_words(cfg,read_words(cfg.vocabulary))]
    scale=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).logit_scale.exp().float()
    anchors=torch.randperm(len(tgt),generator=torch.Generator().manual_seed(42))[:2]
    axis=tgt[anchors[0]]-tgt[anchors[1]]
    folds=(tgt@axis).argsort(stable=True).tensor_split(2)
    assert sum(len(f) for f in folds)==len(tgt) and len(torch.cat(folds).unique())==len(tgt)
    def get(ids,p):
        return torch.stack([cache[p[i]] for i in ids.tolist()]).cuda()
    for split in qs:
        initial=torch.load(w/'results/regional-semantic-interactions'/(split+'.pt'),weights_only=True)
        meta=initial['meta']
        assert torch.equal(initial['qi'],qs[split][0]) and torch.equal(initial['pi'],gs[split][0])
        values,assessment={},{}
        for context in ('center','rest'):
            rows=[]
            for b in meta.split(12):
                base,truth,_,_=interactions(get(b[:,0],paths),context)
                other=interactions(get(b[:,1],paths),context)[1]
                scores=[]
                for ix in (2,3):
                    main,_,effect,_=interactions(get(b[:,ix],photos),context)
                    scores.append(forecast(base,truth,other,base,effect,main,src,tgt,scale,folds))
                rows.append(torch.stack(scores,-1).cpu())
            raw=torch.cat(rows).double()
            for k in ('query_additive','base'):
                assert torch.equal(raw[:,:,ARMS.index(k),0],raw[:,:,ARMS.index(k),1])
            values[context]=raw
            assessment[context]=analyze(meta,raw)
        save_tensor(dict(meta=meta,qi=initial['qi'],pi=initial['pi'],folds=[f.cpu() for f in folds],scores=values),root/(split+'.pt'))
        atomic_json(root/(split+'.json'),dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),arms=ARMS,
            fold_sizes=[len(f) for f in folds],word_axis_ids=anchors.tolist(),assessment=assessment))
        print(split,assessment['center']['same/own']['accuracy'],flush=True)


if __name__=='__main__':
    run()
