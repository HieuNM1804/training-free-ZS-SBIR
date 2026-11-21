"""Democratically aggregate overlapping visual regions independently within each image."""

from dataclasses import dataclass, replace
from pathlib import Path
import argparse

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import Images, configure_image_worker, normal_transform
from .encoding import global_cache_name, load_frozen_clip, save_tensor
from .image_retrieval_round import DECISION
from .retrieval_diagnosis import global_inputs
from .shared_concept_likelihood import emission, target_inputs

FULL_FINALISTS = 2


@dataclass(frozen=True)
class Config:
    scales: str = "5"
    grouping: str = "together"
    side: str = "both"
    pooling: str = "democratic"
    mode: str = "semantic"


def canonical():
    return Config()


def candidates():
    return [Config(s,g,d) for s in ("3","5","3+5") for g in ("together","by_scale")
            for d in ("both","photo")]


def controls(c):
    return {"primary":c,"canonical":canonical(),"same_regions_equal_weights":replace(c,pooling="mean"),
            "region_only":replace(c,grouping="region_only"),"full_frame":replace(c,mode="frame"),
            "rgb_without_words":replace(c,mode="rgb"),"photo_only":replace(c,side="photo"),
            "sketch_only":replace(c,side="sketch"),"shuffled_photos":replace(c,mode="shuffle")}


def cache_key(c):
    return c.scales,c.pooling,c.grouping,c.side,c.mode


def window_ids(side,device):
    grid=torch.arange(1,50,device=device).reshape(7,7)
    ids=grid.unfold(0,side,1).unfold(1,side,1).reshape(-1,side*side)
    return F.pad(ids,(1,0))


@torch.inference_mode()
def region_inputs(a,paths,mode):
    if mode=="photo":
        previous=a.cache/global_cache_name(a.root,paths,mode,"winclip-2-3-5-v1","fp16",a.clip_checkpoint)
        if previous.exists():
            obj=torch.load(previous,weights_only=True,mmap=True)
            return {s:obj["masked"+s] for s in ("3","5")}
    file=a.cache/global_cache_name(a.root,paths,mode,"masked-3-5-democratic-v1","fp16",a.clip_checkpoint)
    if file.exists():
        return torch.load(file,weights_only=True,mmap=True)
    visual=load_frozen_clip(a.clip_checkpoint,torch.device("cuda")).visual
    indices={str(s):window_ids(s,"cuda") for s in (3,5)}
    output={s:[] for s in indices}
    loader=DataLoader(Images(paths,normal_transform()),batch_size=128,num_workers=a.workers,pin_memory=True,
                      worker_init_fn=configure_image_worker,prefetch_factor=2)
    for x in tqdm(loader,desc=mode+" masked CLIP windows",mininterval=6.):
        tokens=visual._embeds(x.to("cuda",dtype=visual.conv1.weight.dtype,non_blocking=True))
        for s,ix in indices.items():
            batch=tokens[:,ix].flatten(0,1)
            local=[]
            for view in batch.split(1024):
                pooled,_=visual._pool(visual.transformer(view))
                local.append(F.normalize((pooled@visual.proj).float(),dim=-1).half().cpu())
            output[s].append(torch.cat(local).reshape(len(x),len(ix),-1))
    save_tensor({s:torch.cat(v) for s,v in output.items()},file)
    return torch.load(file,weights_only=True,mmap=True)


def prepare_ir_data(data,args,qpaths,ppaths,qi,pi):
    data.query_regions=region_inputs(args,qpaths,"sketch"),qi
    data.photo_regions=region_inputs(args,ppaths,"photo"),pi


def aggregate(x,pooling):
    if pooling=="mean":
        return F.normalize(x.sum(1),dim=1)
    K=(x@x.transpose(1,2)).clamp_min_(0.)
    weights=x.new_ones(len(x),x.size(1),1)
    for _ in range(18):
        # Symmetric positive Sinkhorn update with damping 1/2.
        weights=weights/(weights*(K@weights)).clamp_min(1e-20).sqrt()
    return F.normalize((weights*x).sum(1),dim=1)


def pooled(global_x,regions,c):
    source,ids=regions
    batches=[]
    for start in range(0,len(ids),384):
        selected=ids[start:start+384]
        g=global_x[start:start+len(selected),None,:]
        groups=[F.normalize(source[s][selected].to(g.device).float(),dim=2) for s in c.scales.split("+")]
        if c.grouping=="by_scale":
            groups=[aggregate(x,c.pooling)[:,None,:] for x in groups]
        if c.grouping!="region_only":
            groups.append(g)
        batches.append(aggregate(torch.cat(groups,dim=1),c.pooling))
    return torch.cat(batches)


class Index:
    def __init__(self,data):
        self.d=data
        self.cache={}
        self.info={"regions":"3x3 and 5x5 masked-window CLS, 7x7 patch grid",
                   "gram":"positive cosine kernel per individual image","iterations":18}

    def descriptors(self,c,side):
        d=self.d
        x,regions=(d.q,d.query_regions) if side=="sketch" else (d.p,d.photo_regions)
        if c.side not in (side,"both"):
            return x
        key=(side,c.scales,c.grouping,c.pooling)
        if key not in self.cache:
            self.cache[key]=pooled(x,regions,c)
        return self.cache[key]

    def score(self,c):
        if c.mode=="frame":
            return self.d.base
        q=self.descriptors(c,"sketch")
        p=self.descriptors(c,"photo")
        if c.mode=="rgb":
            return q@p.T
        if c.mode=="shuffle":
            ids=torch.randperm(len(p),generator=torch.Generator().manual_seed(14591)).to(p.device)
            p=p[ids]
        probs=target_inputs(p,self.d.target,self.d.scale)
        probs/=probs.sum(1,keepdim=True)
        return emission(q,self.d.source,self.d.scale)@probs.T


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Precompute CLIP window features for this study")
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    opts=parser.parse_args()
    args,qpaths,_,ppaths,_,_,_=global_inputs(opts.workspace,"sketchy")
    for mode,paths in (("sketch",qpaths),("photo",ppaths)):
        print(mode,{s:list(t.shape) for s,t in region_inputs(args,paths,mode).items()},flush=True)
