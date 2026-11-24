"""Compare candidate-conditioned one-bit semantic forecasts with the ambient CLIP word geometry."""
import math
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
from .regional_semantic_interactions import interactions
from .semantic import read_words


ARMS=('fair','source_text','photo_text','query_additive','photo_additive','photo_interactions')
DRAWS=1024


def rank_audit(text,folds,sample):
    result=[]
    for seen,held in (folds,folds[::-1]):
        t=text[seen].double()
        t-=t.mean(0)
        gram=t.T@t
        singular=torch.linalg.eigvalsh(gram).clamp_min(0).sqrt()
        # Reconstruct only from the revealed centered word logits.
        residual=sample.double().flatten(0,1)
        measured=residual@t.T
        predicted=torch.linalg.solve(gram,t.T@measured.T).T
        result.append(dict(n=len(t),rank=int((singular>singular.max()*1e-10).sum()),
            smallest=float(singular.min()),largest=float(singular.max()),
            max_embedding_error=float((residual-predicted).abs().max()),
            max_held_logit_error=float(((residual-predicted)@text[held].double().T).abs().max())))
    return result


def subspace_words(raw,text):
    # The six directions can be linearly dependent; discard only numerical nulls.
    _,s,v=torch.linalg.svd(raw.double(),full_matrices=False)
    return (text@v.float().transpose(1,2))*(s>s[:,:1]*1e-6)[:,None]


def agreement(x,y):
    length=x.norm(dim=-1)*y.norm(dim=-1)
    cos=(x*y).sum(-1)/length.clamp_min(1e-20)
    return torch.where(length>1e-16,1-cos.clamp(-1,1).acos()/math.pi,.5)


def experiment_rows(base,actual,redraw,main,effects,src,tgt,scale,ids,mode,context):
    b,n,_=actual.shape
    weights=(scale*(base@src.T)).softmax(-1)
    delta=(actual-base)@src.T
    sibling=redraw@src.T
    draws=[]
    for j,qid in enumerate(ids.tolist()):
        seed=42+10007*qid+int(context=='rest')*811
        g=torch.Generator(device=src.device).manual_seed(seed)
        if mode=='plausible':
            words=torch.multinomial(weights[j],4*DRAWS,replacement=True,generator=g).reshape(n,DRAWS,4)
        else:
            words=torch.stack([torch.randperm(len(src),generator=g,device=src.device)[:4*DRAWS].reshape(DRAWS,4) for _ in range(n)])
        draws.append(words)
    words=torch.stack(draws)
    sorted_ids=words.sort(dim=-1).values
    valid=(sorted_ids[:,:,:,1:]!=sorted_ids[:,:,:,:-1]).all(-1)
    def sign(values,which):
        lookup=values.gather(2,words[:,:,:,which])
        return (lookup-values.gather(2,words[:,:,:,which+1])).sign()
    observed=sign(delta,0)
    endpoints=(sign(delta,2),sign(sibling,2))
    qproj=subspace_words(base,src)
    draws_per_pair=valid.sum(2)
    def losses(table):
        if table is None:
            match=torch.full_like(observed,.5)
        else:
            if table.ndim==2:table=table.expand(b,-1,-1)
            batch=torch.arange(b,device=src.device)[:,None,None]
            left=table[batch,words[:,:,:,0]]-table[batch,words[:,:,:,1]]
            right=table[batch,words[:,:,:,2]]-table[batch,words[:,:,:,3]]
            match=agreement(left,right)
        prob=.5+observed*(match-.5)
        out=[]
        for second in endpoints:
            target=(1+second)/2
            risk=(prob-target).square()+target*(1-target)
            case=(risk*valid).sum(2)/draws_per_pair.clamp_min(1)
            supported=draws_per_pair>0
            out.append((case*supported).sum(1)/supported.sum(1))
        return torch.stack(out,1)
    common=[losses(v) for v in (None,src,tgt,qproj)]
    arms=[]
    for m,e in zip(main,effects):
        arms.append(torch.stack([*common,losses(subspace_words(m,tgt)),losses(subspace_words(e,tgt))],-1))
    return torch.stack(arms,-1),draws_per_pair


def report(meta,raw):
    out={}
    for m,name in enumerate(('same','different')):
        ok=meta[:,4]==m
        ids=meta[ok,5].numpy()
        for i,view in enumerate(('own','redraw')):
            risk=raw[ok,i].numpy()
            assert np.isfinite(risk).all()
            margin=risk[:,:,1]-risk[:,:,0]
            contrasts={}
            for j,k in enumerate(ARMS[:-1]):
                contrasts[k]=dict(true=cluster_effect(risk[:,j,0]-risk[:,-1,0],ids),
                    discrimination=cluster_effect(margin[:,-1]-margin[:,j],ids))
            out[name+'/'+view]=dict(n=len(risk),true_risk=risk[:,:,0].mean(0).tolist(),
                source_accuracy=((margin>1e-7)+.5*(np.abs(margin)<=1e-7)).mean(0).tolist(),contrasts=contrasts)
    return out


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    unit=torch.tensor([[1.,0.],[1.,0.],[1.,0.]])
    torch.testing.assert_close(agreement(unit,torch.tensor([[1.,0.],[0.,1.],[-1.,0.]])),torch.tensor([1.,.5,0.]))
    w=Path('/content/sbir')
    root=w/'results/regional-one-bit-inference'
    root.mkdir(parents=True,exist_ok=True)
    cfg=runtime(w,'sketchy')
    paths,qs=merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    ph,gs=merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    cache={}
    for f in (w/'cache/regional-semantic-interactions').glob('*.pt'):
        z=torch.load(f,weights_only=True)
        cache.update(zip(z['paths'],z['features']))
    src,tgt=[v.cuda() for v in load_words(cfg,read_words(cfg.vocabulary))]
    scale=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).logit_scale.exp().float()
    def get(ids,items):
        return torch.stack([cache[items[k]] for k in ids.tolist()]).cuda()
    for split in qs:
        ref=torch.load(w/'results/regional-conditional-repertoire'/(split+'.pt'),weights_only=True)
        meta=ref['meta']
        assert torch.equal(ref['qi'],qs[split][0]) and torch.equal(ref['pi'],gs[split][0])
        folds=tuple(f.cuda() for f in ref['folds'])
        audit=rank_audit(src,folds,scale*interactions(get(meta[:16,0],paths),'center')[2])
        results,stored={},{}
        for context in ('center','rest'):
            for mode in ('plausible','uniform'):
                rows,counts=[],[]
                for b in meta.split(4):
                    base,actual,_,_=interactions(get(b[:,0],paths),context)
                    sibling_delta=interactions(get(b[:,1],paths),context)[2]
                    candidates=[interactions(get(b[:,i],ph),context) for i in (2,3)]
                    raw,count=experiment_rows(base,actual,sibling_delta,[t[0] for t in candidates],
                        [t[2] for t in candidates],src,tgt,scale,b[:,0],mode,context)
                    rows.append(raw.cpu())
                    counts.append(count.cpu())
                raw=torch.cat(rows).double()
                count=torch.cat(counts)
                tag=context+'/'+mode
                stored[tag]=dict(risk=raw,eligible=count)
                results[tag]=dict(coverage=float(count.double().mean()/DRAWS),
                                  empty_pairs=int((count==0).sum()),assessment=report(meta,raw))
        save_tensor(dict(meta=meta,rows=stored),root/(split+'.pt'))
        atomic_json(root/(split+'.json'),dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),
            arms=ARMS,audit=audit,results=results))
        print(split,results['center/plausible']['assessment']['same/own']['source_accuracy'],flush=True)


if __name__=='__main__':
    run()
