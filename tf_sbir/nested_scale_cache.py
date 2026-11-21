"""Encode the two nested zoom scales in each unlabelled gallery photograph."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json,read
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .data import Images,normal_transform,configure_image_worker
from .encoding import save_tensor,load_frozen_clip
from .regional_interventions import windows,SIDES
from .retrieval_diagnosis import global_inputs

FAMILY="nested-scale-word-evidence"


@torch.inference_mode()
def run(w,dataset):
    cfg,_,_,paths,_,_,raw=global_inputs(w,dataset)
    folder=w/"cache"/FAMILY/dataset
    sig=digest({"photos":[str(p) for p in paths],"checkpoint":CHECKPOINT,"sizes":list(SIDES),"resize":"224-bicubic-antialiased"})
    final=folder/"photo.pt"
    info=folder/"photo.json"
    if final.is_file() and info.is_file() and read(info)["signature"]==sig:return
    v=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    out=torch.empty((len(paths),13,512),dtype=torch.float16)
    batch=96
    dl=DataLoader(Images(paths,normal_transform(),"rgb"),batch_size=batch,num_workers=5,pin_memory=True,
                  prefetch_factor=2,worker_init_fn=configure_image_worker)
    offset=0
    checked=0.
    for step,pixels in enumerate(dl):
        p=pixels.cuda(non_blocking=True)
        if step%48==0:
            native=F.normalize(v(p.to(v.conv1.weight.dtype)).float())
            checked=max(checked,float((1-(native*raw[offset:offset+len(p)].cuda()).sum(1)).abs().max()))
            if checked>.002:raise ValueError("Gallery photo ordering differs from native CLIP")
        features=[]
        for side in SIDES:
            positions=windows(side)
            for start in range(0,len(positions),4):
                cells=positions[start:start+4]
                tile=torch.stack([p[:,:,y*32:(y+side)*32,x*32:(x+side)*32] for y,x in cells],1)
                resized=F.interpolate(tile.flatten(0,1),size=(224,224),mode="bicubic",align_corners=False,antialias=True)
                values=F.normalize(v(resized.to(v.conv1.weight.dtype)).float()).view(len(p),len(cells),-1)
                features.append(values)
        out[offset:offset+len(p)]=torch.cat(features,1).half().cpu()
        offset+=len(p)
        if step%50==0:print(dataset,offset,len(paths),flush=True)
    if offset!=len(paths):raise ValueError("Incomplete photo windows")
    save_tensor(out,final)
    atomic_json(info,{"signature":sig,"photos":len(paths),"native_check":checked})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--dataset",choices=("sketchy","tuberlin"),required=True)
    a=parser.parse_args();run(a.workspace,a.dataset)
