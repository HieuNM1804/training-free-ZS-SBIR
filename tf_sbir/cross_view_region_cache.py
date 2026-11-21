"""Read the horizontal reflection of each original sketch window with the same CLIP."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json,read
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .data import Images,normal_transform,configure_image_worker
from .encoding import load_frozen_clip,save_tensor
from .regional_interventions import windows,SIDES
from .retrieval_diagnosis import global_inputs

FAMILY="cross-view-region-verification"


@torch.inference_mode()
def run(w,ds):
    cfg,paths,_,_,_,raw,_=global_inputs(w,ds)
    dest=w/"cache"/FAMILY/ds
    file=dest/"views.pt"
    info=dest/"views.json"
    sig=digest({"checkpoint":CHECKPOINT,"paths":[str(p) for p in paths],"sizes":list(SIDES),"pose":"mirror each corresponding crop"})
    if file.exists() and info.exists() and read(info)["signature"]==sig:return
    v=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    dl=DataLoader(Images(paths,normal_transform(),"rgb"),batch_size=120,num_workers=5,pin_memory=True,
                  prefetch_factor=2,worker_init_fn=configure_image_worker)
    data=torch.empty((len(paths),13,512),dtype=torch.float16)
    full=torch.empty((len(paths),512),dtype=torch.float16)
    offset=0
    err=0.
    for step,px in enumerate(dl):
        p=px.cuda(non_blocking=True)
        if step%40==0:
            direct=F.normalize(v(p.to(v.conv1.weight.dtype)).float())
            err=max(err,float((1-(direct*raw[offset:offset+len(p)].cuda()).sum(1)).abs().max()))
            if err>.002:raise ValueError("Native sketch mismatch")
        full[offset:offset+len(p)]=F.normalize(v(p.flip((-1,)).to(v.conv1.weight.dtype)).float()).half().cpu()
        outs=[]
        for side in SIDES:
            loc=windows(side)
            for start in range(0,len(loc),3):
                use=loc[start:start+3]
                patches=torch.stack([p[:,:,32*y:32*(y+side),32*x:32*(x+side)] for y,x in use],1)
                zoom=F.interpolate(patches.flatten(0,1),size=(224,224),mode="bicubic",align_corners=False,antialias=True).flip((-1,))
                z=F.normalize(v(zoom.to(v.conv1.weight.dtype)).float()).reshape(len(p),len(use),-1)
                outs.append(z)
        data[offset:offset+len(p)]=torch.cat(outs,1).half().cpu()
        offset+=len(p)
        if step%48==0:print(ds,offset,len(paths),flush=True)
    if offset!=len(paths):raise ValueError("Missing source windows")
    save_tensor({"regions":data,"global":full},file)
    atomic_json(info,{"signature":sig,"queries":len(paths),"native_check":err})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),required=True)
    a=p.parse_args();run(a.workspace,a.dataset)
