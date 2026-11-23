"""Exact finite differences of the frozen global CLIP descriptor after deleting one patch."""
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .benchmark import merge_inputs
from .crossdomain_inputs import runtime
from .data import Images,SKETCHY_SPLITS,configure_image_worker,normal_transform
from .encoding import load_frozen_clip,save_tensor
from .gallery_context_reversals import coordinates
from .standardized_context_audit import audit_inputs


def global_readout(vision,tokens):
    pooled,_=vision._pool(vision.transformer(tokens))
    return F.normalize((pooled@vision.proj).float(),dim=-1)


def cache_dir(w,paths):
    h=hashlib.sha256(('openai-b32/input-mean/single-native-tile/cls-v1\n'+'\n'.join(map(str,paths))).encode()).hexdigest()
    return w/'cache/patch-removal-correspondence'/h


def intervention(vision,px,check):
    # LayerNorm and the patch projection are token-wise at this boundary.
    # Replacing a single input embedding is identical to the corresponding RGB edit.
    orig=vision._embeds(px)
    mean=px.float().mean((-1,-2),keepdim=True).to(px.dtype)
    canvas=vision._embeds(mean.expand_as(px).contiguous())
    n,bands,width=orig.shape
    assert bands==50 and width==768
    variants=orig[:,None].expand(n,50,50,width).clone()
    edits=torch.arange(1,50,device=px.device)
    variants[:,edits,edits]=canvas[:,1:]
    cls=global_readout(vision,variants.flatten(0,1)).view(n,50,-1)
    raw=cls[:,:1]-cls[:,1:]
    same=(orig[:,1:]==canvas[:,1:]).all(-1)
    raw=raw.masked_fill(same[:,:,None],0.)
    err=0.
    if check:
        ids=torch.tensor([0,8,18,31,48],device=px.device)
        explicit=px[:1].expand(len(ids)+1,-1,-1,-1).clone()
        for j,k in enumerate(ids.tolist(),1):
            y,x=divmod(k,7)
            explicit[j,:,32*y:32*(y+1),32*x:32*(x+1)]=mean[0]
        check_cls=F.normalize(vision(explicit).float())
        err=float((1-(check_cls*cls[0,torch.cat([ids[:1],ids+1])]).sum(1)).abs().max())
        if err>.0001:raise ValueError(f'Pixel-level reference differs {err}')
    return cls[:,0].half().cpu(),raw.half().cpu(),err


@torch.inference_mode()
def encode(w,paths,vision,tag):
    cache=cache_dir(w,paths)
    cache.mkdir(parents=True,exist_ok=True)
    count=240
    for start in range(0,len(paths),count):
        part=paths[start:start+count]
        file=cache/f'{start:06d}.pt'
        if file.is_file():
            old=torch.load(file,weights_only=True,mmap=True)
            assert old['delta'].shape==(len(part),49,512)
            continue
        dl=DataLoader(Images(part,normal_transform(),'rgb'),batch_size=30,
                num_workers=min(2,len(os.sched_getaffinity(0))),pin_memory=True,worker_init_fn=configure_image_worker)
        cls,delta=[],[]
        err=0.
        for batch in dl:
            for px in batch.cuda(non_blocking=True).to(vision.conv1.weight.dtype).split(6):
                x,v,d=intervention(vision,px,not cls)
                cls.append(x);delta.append(v);err=max(err,d)
        save_tensor({'cls':torch.cat(cls),'delta':torch.cat(delta),'rgb_check':err},file)
        if start%(count*4)==0 or start+count>=len(paths):print(tag,min(start+count,len(paths)),'/',len(paths),flush=True)
    if not (cache/'done.json').exists():
        (cache/'done.json').write_text(json.dumps({'images':len(paths),'shard':count,'hash':cache.name}))
    return cache


def load_response(w,paths):
    cache=cache_dir(w,paths)
    cfg=json.loads((cache/'done.json').read_text())
    assert cfg['images']==len(paths) and cfg['hash']==cache.name
    pieces=[torch.load(cache/f'{i:06d}.pt',mmap=True,weights_only=True) for i in range(0,len(paths),cfg['shard'])]
    return torch.cat([a['cls'] for a in pieces]),torch.cat([a['delta'] for a in pieces]),max(a['rgb_check'] for a in pieces)


def make_plan(w,qs,gs,enc,q,p,flip):
    cfg=runtime(w,'sketchy')
    qp,_=merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    pp,_=merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    panels={}
    union_q=[]
    union_g=[]
    for s,(qi,_) in qs.items():
        rows=torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values
        ix=coordinates(enc,qi[rows])
        pi,_=gs[s]
        candidates=(F.normalize(q[ix].cuda()+flip[ix].float().cuda())@F.normalize(p[pi].cuda()).T).topk(200).indices.cpu()
        panels[s]=(rows,ix,candidates)
        union_q.append(qi[rows]);union_g.append(pi[candidates.flatten()].unique())
    uq=torch.cat(union_q).unique(sorted=True)
    ug=torch.cat(union_g).unique(sorted=True)
    return cfg,uq,ug,[qp[i] for i in uq.tolist()],[pp[i] for i in ug.tolist()],panels


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    qs,gs,enc,_,q,p,flip,*_=audit_inputs(w,768)
    cfg,_,_,qpaths,gpaths,_=make_plan(w,qs,gs,enc,q,p,flip)
    print('unique:',len(qpaths),len(gpaths),flush=True)
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    encode(w,qpaths,vision,'queries')
    encode(w,gpaths,vision,'gallery')


if __name__=='__main__':
    run()
