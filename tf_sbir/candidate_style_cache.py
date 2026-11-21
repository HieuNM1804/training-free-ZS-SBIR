"""Counterfactual rendering of one query under gallery patch-style distributions."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json,read
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .data import Images,configure_image_worker,normal_transform
from .encoding import load_frozen_clip,save_tensor
from .retrieval_diagnosis import global_inputs


FAMILY="candidate-style-counterfactual"
K=6
FRACTIONS=(.35,.85)


def patch_features(w,dataset,role,cfg,paths,vision):
    folder=w/"cache"/FAMILY/dataset
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/(role+"-conv.npy")
    info=folder/(role+"-conv.json")
    sig={"count":len(paths),"paths_sha256":digest([str(p) for p in paths]),"checkpoint":CHECKPOINT}
    if path.exists() and info.exists() and read(info)==sig:
        return np.load(path,mmap_mode="r")
    temp=folder/(role+"-conv-tmp.npy")
    arr=np.lib.format.open_memmap(temp,mode="w+",dtype=np.float16,shape=(len(paths),49,768))
    loader=DataLoader(Images(paths,normal_transform(),"rgb"),batch_size=224,num_workers=5,pin_memory=True,
                      worker_init_fn=configure_image_worker,prefetch_factor=2)
    start=0
    for px in loader:
        v=vision.conv1(px.cuda(non_blocking=True).to(vision.conv1.weight.dtype)).flatten(2).transpose(1,2)
        arr[start:start+len(v)]=v.cpu().numpy()
        start+=len(v)
    if start!=len(paths):raise ValueError("Incomplete patch-token bank")
    arr.flush()
    temp.replace(path)
    atomic_json(info,sig)
    return np.load(path,mmap_mode="r")


def stats(x):
    v,m=torch.var_mean(x.float(),dim=1,correction=0)
    return m,v.sqrt()


def clustering(stats,k):
    gen=torch.Generator(device=stats.device).manual_seed(42717)
    means=stats[torch.randperm(len(stats),generator=gen,device=stats.device)[:k]]
    for _ in range(35):
        cl=torch.cdist(stats,means).argmin(1)
        total=torch.zeros_like(means).index_add_(0,cl,stats)
        n=cl.bincount(minlength=k)
        means=torch.where(n[:,None]>0,total/n[:,None].clamp_min(1),means)
    cl=torch.cdist(stats,means).argmin(1)
    return cl,means


def native(v,patches):
    cls=v.class_embedding.to(patches).expand(len(patches),1,-1)
    original=v.ln_pre(torch.cat((cls,patches),1)+v.positional_embedding.to(patches))
    tokens=v.transformer(original)
    pooled,_=v._pool(tokens)
    return F.normalize((pooled@v.proj).float())


def render(v,patches,gaussian,fraction):
    m,sd=stats(patches)
    dest_m,dest_sd=gaussian.chunk(2)
    # The diagonal Gaussian Wasserstein geodesic linearly interpolates means and
    # standard deviations. Degenerate channels retain zero spatial residual.
    shift=m+fraction*(dest_m-m)
    size=sd+fraction*(dest_sd-sd)
    t=shift[:,None]+(patches-m[:,None])*((size/sd.clamp_min(1e-7))[:,None])
    return native(v,t.to(v.conv1.weight.dtype))


def cache_views(v,raw,ids,features,centers,alpha):
    out=torch.empty((len(ids),len(centers),512),dtype=torch.float16)
    max_error=0.
    for start in range(0,len(ids),144):
        selected=ids[start:start+144]
        x=torch.from_numpy(raw[selected.numpy()]).cuda()
        if start%1152==0:
            max_error=max(max_error,float((1-(native(v,x)*features[selected].cuda()).sum(1)).abs().max()))
            if max_error>.002:raise ValueError("Style intervention lost correspondence with native RGB features")
        result=torch.stack([render(v,x,c,alpha) for c in centers],1)
        out[start:start+len(x)]=result.half().cpu()
    return out,max_error


@torch.inference_mode()
def run(w):
    for dataset in ("sketchy","tuberlin"):
        cfg,qp,qs,pp,gs,qraw,praw=global_inputs(w,dataset)
        vision=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
        photos=patch_features(w,dataset,"gallery",cfg,pp,vision)
        sketches=patch_features(w,dataset,"query",cfg,qp,vision)
        for split,(qi,_) in qs.items():
            pi,_=gs[split]
            folder=w/"cache"/FAMILY/split
            moments=torch.cat([torch.cat(stats(torch.from_numpy(photos[chunk.numpy()]).cuda()),1)
                               for chunk in pi.split(160)])
            cluster,centers=clustering(moments,K)
            centers=torch.cat((centers,moments.mean(0,keepdim=True)))
            ref=torch.randperm(len(pi),generator=torch.Generator().manual_seed(85571))[:128]
            for alpha in FRACTIONS:
                file=folder/f"views-{alpha:g}.pt"
                if file.exists():continue
                query,error=cache_views(vision,sketches,qi,qraw,centers,alpha)
                panel,perr=cache_views(vision,photos,pi[ref],praw,centers,alpha)
                save_tensor({"q":query,"reference":panel,"reference_ids":ref,"styles":cluster.cpu(),
                             "centers":centers.cpu(),"alpha":alpha,"native_input_max_error":max(error,perr),
                             "checkpoint":CHECKPOINT,"style_count":K,"queries":len(qi),"gallery":len(pi)},file)
                print(split,"fraction",alpha,"complete",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
