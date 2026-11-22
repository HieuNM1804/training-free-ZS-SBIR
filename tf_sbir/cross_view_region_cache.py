"""Read both poses of each corresponding sketch window using frozen CLIP."""

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


def two_pose_regions(vision, pixels):
    full = F.normalize(vision(pixels.flip((-1,)).to(vision.conv1.weight.dtype)).float())
    outputs = []
    for side in SIDES:
        positions = windows(side)
        for start in range(0, len(positions), 3):
            take = positions[start:start + 3]
            patches = torch.stack([pixels[:, :, 32*y:32*(y+side), 32*x:32*(x+side)] for y, x in take], 1)
            zoom = F.interpolate(patches.flatten(0, 1), size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
            both = torch.stack((zoom, zoom.flip((-1,))), 1).flatten(0, 1)
            projected = F.normalize(vision(both.to(vision.conv1.weight.dtype)).float())
            outputs.append(projected.reshape(len(pixels), len(take), 2, -1))
    return full.half(), torch.cat(outputs, 1).half()


@torch.inference_mode()
def run(w,ds):
    cfg,paths,_,_,_,raw,_=global_inputs(w,ds)
    dest=w/"cache"/FAMILY/ds
    file=dest/"views.pt"
    info=dest/"views.json"
    sig=digest({"checkpoint":CHECKPOINT,"paths":[str(p) for p in paths],"sizes":list(SIDES),"pose":"native and mirrored corresponding crops"})
    if file.exists() and info.exists() and read(info)["signature"]==sig:return
    v=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    dl=DataLoader(Images(paths,normal_transform(),"rgb"),batch_size=120,num_workers=5,pin_memory=True,
                  prefetch_factor=2,worker_init_fn=configure_image_worker)
    data=torch.empty((len(paths),13,512),dtype=torch.float16)
    original=torch.empty_like(data)
    full=torch.empty((len(paths),512),dtype=torch.float16)
    offset=0
    err=0.
    for step,px in enumerate(dl):
        p=px.cuda(non_blocking=True)
        if step%40==0:
            direct=F.normalize(v(p.to(v.conv1.weight.dtype)).float())
            err=max(err,float((1-(direct*raw[offset:offset+len(p)].cuda()).sum(1)).abs().max()))
            if err>.002:raise ValueError("Native sketch mismatch")
        reflected,pair=two_pose_regions(v,p)
        full[offset:offset+len(p)]=reflected.cpu()
        pair=pair.cpu()
        original[offset:offset+len(p)]=pair[:,:,0]
        data[offset:offset+len(p)]=pair[:,:,1]
        offset+=len(p)
        if step%48==0:print(ds,offset,len(paths),flush=True)
    if offset!=len(paths):raise ValueError("Missing source windows")
    save_tensor({"original_regions":original,"regions":data,"global":full},file)
    atomic_json(info,{"signature":sig,"queries":len(paths),"native_check":err})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),required=True)
    a=p.parse_args();run(a.workspace,a.dataset)
