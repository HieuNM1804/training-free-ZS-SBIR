"""Keep a common finite orbit of native patch contents in eight physical image poses."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json,read
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .data import Images,configure_image_worker,normal_transform
from .encoding import load_frozen_clip,save_tensor
from .layout_word_cache import native_pool
from .retrieval_diagnosis import global_inputs


FAMILY="pose-conditional-layout"
SEED=589241
POSES=tuple((r,f) for r in range(4) for f in (False,True))


def orient(image,pose):
    r,f=pose
    x=torch.rot90(image,r,dims=(-2,-1))
    return x.flip((-1,)) if f else x


def patch_orbit(device):
    generator=torch.Generator().manual_seed(SEED)
    order=torch.randperm(49,generator=generator)
    groups=order[:48].view(12,4)
    mats=[]
    for step in range(4):
        move=torch.arange(49)
        move[groups.flatten()]=groups.roll(-step,dims=1).flatten()
        mats.append(move)
    return torch.stack(mats).to(device)


@torch.inference_mode()
def extract(w,dataset,cfg,paths,raw,batch,workers,pose_ids):
    outdir=w/"cache"/FAMILY/dataset
    outdir.mkdir(parents=True,exist_ok=True)
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    orbit=patch_orbit("cuda")
    assert torch.equal(orbit[1][orbit[1]],orbit[2]) and torch.equal(orbit[1][orbit[3]],orbit[0])
    grid=torch.arange(49,device=orbit.device).view(7,7)
    for h in pose_ids:
        pose=POSES[h]
        file=outdir/f"pose-{h}.pt"
        info=outdir/f"pose-{h}.json"
        sig={"paths":digest([str(p) for p in paths]),"checkpoint":CHECKPOINT,"pose":h,"orbit_seed":SEED,"queries":len(paths)}
        if file.exists() and info.exists() and read(info)["identity"]==sig:continue
        maph=orient(grid,pose).flatten()
        index=maph.argsort()[orbit[:,maph]]
        if not torch.equal(index[1][index[1]],index[2]):raise ValueError("Physical pose failed to conjugate the patch group")
        values=torch.empty((len(paths),4,512),dtype=torch.float16)
        loader=DataLoader(Images(paths,normal_transform(),"rgb"),batch_size=batch,num_workers=workers,pin_memory=True,
                          worker_init_fn=configure_image_worker,prefetch_factor=2 if workers else None)
        start=0
        err=0.
        for it,pixels in enumerate(loader):
            pixels=orient(pixels.cuda(non_blocking=True).to(vision.conv1.weight.dtype),pose).contiguous()
            patches=vision.conv1(pixels).flatten(2).transpose(1,2)
            n=len(pixels)
            body=patches[:,index]
            cls=vision.class_embedding.to(body).view(1,1,1,-1).expand(n,4,1,-1)
            seq=vision.ln_pre(torch.cat((cls,body),2)+vision.positional_embedding.to(body)[None,None]).flatten(0,1)
            feat=native_pool(vision,seq).unflatten(0,(n,4))
            if it%80==0:
                same=raw[start:start+n].to(feat) if h==0 else F.normalize(vision(pixels).float())
                err=max(err,float((1-(same*feat[:,0]).sum(1)).abs().max()))
                if err>.002:raise ValueError("Native pose replay changed")
            values[start:start+n]=feat.half().cpu()
            start+=n
        if start!=len(paths):raise ValueError("Missing transformed query features")
        save_tensor(values,file)
        atomic_json(info,{"identity":sig,"raw_pose_max_one_minus_cosine":err})
        print(dataset,"pose",h,start,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--batch",type=int,default=112)
    parser.add_argument("--workers",type=int,default=5)
    parser.add_argument("--poses",type=int,nargs="+",default=list(range(8)))
    args=parser.parse_args()
    for dataset in ("sketchy","tuberlin"):
        cfg,paths,_,_,_,raw,_=global_inputs(args.workspace,dataset)
        extract(args.workspace,dataset,cfg,paths,raw,args.batch,args.workers,args.poses)
