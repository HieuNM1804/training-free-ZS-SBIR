"""Test whether a candidate predicts semantic information that emerges when two sketch regions coexist."""
import hashlib
import io
from itertools import combinations
import os
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .benchmark import merge_inputs
from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .crossdomain_inputs import runtime
from .data import Images, SKETCHY_SPLITS, configure_image_worker, normal_transform
from .encoding import load_frozen_clip, save_tensor
from .foundation import load_words
from .regional_predictive_observation import cluster_effect
from .semantic import read_words


PAIRS=list(combinations(range(4),2))
OPPOSITE=[PAIRS.index(tuple(i for i in range(4) if i not in pair)) for pair in PAIRS]
ARMS=('raw_additive','unit_additive','candidate','wrong_parts','generic')


def coalition_tokens(device):
    y,x=torch.meshgrid(torch.arange(7),torch.arange(7),indexing='ij')
    group=(2*(y>3).long()+(x>3).long()).flatten()
    cross=((y==3)|(x==3)).flatten()
    return [torch.cat((torch.zeros(1,dtype=torch.long),torch.where(cross|(((mask>>group)&1).bool()))[0]+1)).to(device)
            for mask in range(16)]


def encode(paths,vision,device):
    masks=coalition_tokens(device)
    counts=sorted(set(len(m) for m in masks))
    loader=DataLoader(Images(paths,normal_transform()),batch_size=40,num_workers=min(2,len(os.sched_getaffinity(0))),
                      pin_memory=True,worker_init_fn=configure_image_worker)
    collected=[]
    maximum=0.
    for pixels in loader:
        pixels=pixels.to(device=device,dtype=vision.conv1.weight.dtype,non_blocking=True)
        tokens=vision._embeds(pixels)
        raw=torch.empty((len(tokens),16,512),device=device)
        for size in counts:
            ids=[i for i,m in enumerate(masks) if len(m)==size]
            view=tokens[:,torch.stack([masks[i] for i in ids])].flatten(0,1)
            out=vision.transformer(view)
            raw[:,ids]=(vision.ln_post(out[:,0])@vision.proj).float().reshape(len(tokens),len(ids),-1)
        if not collected:
            native=vision(pixels).float()
            maximum=float((F.normalize(native)-F.normalize(raw[:,15])).abs().max())
            if maximum>.0005:
                raise ValueError(f'Full-token replay error {maximum}')
        collected.append(raw.half().cpu())
    return torch.cat(collected),maximum


def read_trials(bundle,split,qi,pi,paths,photos):
    with zipfile.ZipFile(bundle) as z:
        record=torch.load(io.BytesIO(z.read(split+'.pt')),weights_only=True)
    assert torch.equal(qi,record['qi']) and torch.equal(pi,record['pi'])
    trials=record['trials'].numpy()
    rng=np.random.default_rng(42)
    choices={}
    for src in np.unique(trials[:,1]):
        members=np.unique(trials[trials[:,1]==src,0])
        own,redraw=rng.choice(members,size=2,replace=False)
        for row in (own,redraw):
            sketch=Path(paths[int(qi[row])])
            photo=Path(photos[int(pi[src])])
            assert sketch.parent.name==photo.parent.name and sketch.stem.rsplit('-',1)[0]==photo.stem
        choices[src]=(int(qi[own]),int(qi[redraw]))
    meta=[]
    for src,foil,mode,pair in np.unique(trials[:,1:],axis=0):
        meta.append((*choices[src],int(pi[src]),int(pi[foil]),mode,pair))
    return torch.tensor(np.asarray(meta))


def interactions(raw,context):
    raw=raw.float()
    c=torch.tensor([0 if context=='center' else 15^(1<<a)^(1<<b) for a,b in PAIRS],device=raw.device)
    a=c|torch.tensor([1<<a for a,b in PAIRS],device=raw.device)
    b=c|torch.tensor([1<<b for a,b in PAIRS],device=raw.device)
    ab=a|b
    norm=lambda v:F.normalize(v,dim=-1)
    base=norm(raw[:,a]+raw[:,b]-raw[:,c])
    observed=norm(raw[:,ab])
    unit=norm(raw)[:,a]+norm(raw)[:,b]-norm(raw)[:,c]
    return base,observed,observed-base,unit


def log_score(logits,target):
    return (target*logits.log_softmax(-1)).sum(-1).mean(-1)


def inspect_context(meta,sketch_ids,sketches,photo_ids,photos,generic,src,tgt,scale,context):
    library=interactions(photos.cuda(),context)[2]
    mean=library[torch.searchsorted(photo_ids,generic).cuda()].mean(0)
    records=[]
    for batch in meta.split(12):
        own,redraw,photo,foil,_,_=batch.T
        q=sketches[torch.searchsorted(sketch_ids,own.contiguous())].cuda()
        s=sketches[torch.searchsorted(sketch_ids,redraw.contiguous())].cuda()
        base,observed,_,unit=interactions(q,context)
        targets=[(scale*(h@src.T)).softmax(-1) for h in (observed,interactions(s,context)[1])]
        common=scale*(base@src.T)
        lower=scale*(unit@src.T)
        scores=[]
        for photo_batch in (photo,foil):
            increment=library[torch.searchsorted(photo_ids,photo_batch.contiguous()).cuda()]
            correction=scale*(increment@tgt.T)
            candidates=[common,lower,common+correction,common+correction[:,OPPOSITE],common+scale*(mean@tgt.T)]
            scores.append(torch.stack([torch.stack([log_score(l,t) for l in candidates],-1)
                                       for t in targets],1))
        oracle=torch.stack([(t*t.clamp_min(1e-30).log()).sum(-1).mean(-1) for t in targets],1)
        records.append((torch.stack(scores,-1).cpu(),oracle.cpu()))
    return tuple(torch.cat(x).double() for x in zip(*records))


def summary(meta,scores,oracle):
    output={}
    for mode,group in enumerate(('same_category','different_category')):
        keep=meta[:,4]==mode
        pair_ids=meta[keep,5].numpy()
        for i,target in enumerate(('own','redraw')):
            m=scores[keep,i].numpy()
            margin=m[:,:,0]-m[:,:,1]
            hit=(margin>1e-6)+.5*(np.abs(margin)<=1e-6)
            ids=np.unique(pair_ids)
            if not all((pair_ids==pid).sum()==2 for pid in ids):
                raise ValueError('Paired design must include one drawing in each source direction')
            pair_margin=np.stack([margin[pair_ids==pid].mean(0) for pid in ids])
            comparisons={}
            for j,name in enumerate(ARMS):
                if name=='candidate':continue
                comparisons[name]=dict(true=cluster_effect(m[:,2,0]-m[:,j,0],pair_ids),
                    discrimination=cluster_effect(margin[:,2]-margin[:,j],pair_ids))
            output[group+'/'+target]=dict(sources=len(m),pairs=len(ids),accuracy=hit.mean(0).tolist(),
                paired_accuracy=((pair_margin>1e-6)+.5*(np.abs(pair_margin)<=1e-6)).mean(0).tolist(),
                oracle_gain_over_additive=float((oracle[keep,i].numpy()-m[:,0,0]).mean()),
                contrasts=comparisons)
    return output


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    w=Path('/content/sbir')
    root=w/'results/regional-semantic-interactions'
    root.mkdir(parents=True,exist_ok=True)
    cfg=runtime(w,'sketchy')
    sk_paths,qs=merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    ph_paths,gs=merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    bundle=w/'repo/docs/results/semantic-observation-channel/paired-T4.zip'
    meta={s:read_trials(bundle,s,qs[s][0],gs[s][0],sk_paths,ph_paths) for s in qs}
    used=torch.cat([v[:,2:4].flatten() for v in meta.values()]).unique()
    generics={}
    for s,(pi,_) in gs.items():
        pool=pi[~torch.isin(pi,used)]
        generics[s]=pool[torch.randperm(len(pool),generator=torch.Generator().manual_seed(42))[:128]]
    sketch_ids=torch.cat([m[:,:2].flatten() for m in meta.values()]).unique()
    photo_ids=torch.cat([used,*generics.values()]).unique()
    words=read_words(cfg.vocabulary)
    assert len(words)==20931
    src,tgt=[z.cuda() for z in load_words(cfg,words)]
    model=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda'))
    scale=model.logit_scale.exp().float()
    def features(paths,ids):
        selected=[paths[i] for i in ids.tolist()]
        stamp=hashlib.sha256(('raw-coalitions-v1/'+str(cfg.clip_checkpoint)+'\n'+'\n'.join(selected)).encode()).hexdigest()
        file=w/'cache/regional-semantic-interactions'/(stamp+'.pt')
        if file.exists():
            saved=torch.load(file,weights_only=True)
            assert saved['paths']==selected
            return saved['features'],saved['error']
        result,error=encode(selected,model.visual,torch.device('cuda'))
        save_tensor(dict(paths=selected,features=result,error=error),file)
        return result,error
    sketches,sk_err=features(sk_paths,sketch_ids)
    photos,ph_err=features(ph_paths,photo_ids)
    for s,trials in meta.items():
        outcomes,raws={},{}
        for context in ('center','rest'):
            score,oracle=inspect_context(trials,sketch_ids,sketches,photo_ids,photos,generics[s],src,tgt,scale,context)
            outcomes[context]=summary(trials,score,oracle)
            raws[context]=dict(scores=score,oracle=oracle)
        save_tensor(dict(meta=trials,generic=generics[s],qi=qs[s][0],pi=gs[s][0],rows=raws),root/(s+'.pt'))
        atomic_json(root/(s+'.json'),dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),arms=ARMS,
            full_token_replay_max=[sk_err,ph_err],assessment=outcomes))
        print(s,len(trials),outcomes['center']['same_category/own']['accuracy'],flush=True)


if __name__=='__main__':
    run()
