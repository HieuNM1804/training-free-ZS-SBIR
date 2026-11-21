"""Encode the same RGB patch contents with their native image positions reassigned."""

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
from .retrieval_diagnosis import global_inputs


FAMILY="layout-conditioned-word-evidence"
SEED=533412
DRAWS=4


def layouts(dev):
    rng=torch.Generator().manual_seed(SEED)
    perms=torch.stack([torch.randperm(49,generator=rng) for _ in range(DRAWS)]).to(dev)
    return perms


def native_pool(v,tokens):
    hidden=v.transformer(tokens)
    pooled,_=v._pool(hidden)
    return F.normalize((pooled@v.proj).float())


@torch.inference_mode()
def collect(w,dataset,role,cfg,paths,raw,batch,workers,model):
    folder=w/"cache"/FAMILY/dataset
    file=folder/(role+".pt")
    record=folder/(role+".json")
    identity={"paths_sha256":digest([str(p) for p in paths]),"n":len(paths),"checkpoint":CHECKPOINT,"seed":SEED,"draws":DRAWS}
    if file.exists() and record.exists() and read(record)["identity"]==identity:return
    v=model.visual
    perms=layouts("cuda")
    loader=DataLoader(Images(paths,normal_transform(),"rgb"),batch_size=batch,num_workers=workers,pin_memory=True,
                      worker_init_fn=configure_image_worker,prefetch_factor=2 if workers else None)
    out=torch.empty((len(paths),DRAWS,512),dtype=torch.float16)
    max_error=0.
    equivariance=0.
    start=0
    for ids,pixels in enumerate(loader):
        pixels=pixels.cuda(non_blocking=True).to(v.conv1.weight.dtype)
        patches=v.conv1(pixels).flatten(2).transpose(1,2)
        n=len(pixels)
        cls=v.class_embedding.to(patches).expand(n,1,-1)
        pos=v.positional_embedding.to(patches)
        # The content permutation is applied before the positional vectors. Local
        # edges, local appearance and every patch vector remain identical by construction.
        arranged=patches[:,perms]
        content=torch.cat((cls[:,None].expand(n,DRAWS,1,-1),arranged),dim=2)
        tokens=v.ln_pre(content+pos[None,None]).flatten(0,1)
        features=native_pool(v,tokens).reshape(n,DRAWS,512)
        out[start:start+n]=features.half().cpu()
        if ids % 30 == 0:
            original=v.ln_pre(torch.cat((cls,patches),1)+pos)
            actual=native_pool(v,original)
            e=float((1-(actual*raw[start:start+n].to(actual)).sum(1)).abs().max())
            max_error=max(max_error,e)
            ix=torch.cat((torch.zeros(1,device=pixels.device,dtype=torch.long),perms[0]+1))
            jointly=native_pool(v,original[:,ix])
            equivariance=max(equivariance,float((1-(actual*jointly).sum(1)).abs().max()))
            if max(max_error,equivariance)>.002: raise ValueError("Native encoder or joint positional-permutation control mismatch")
        start+=n
        if start%4096<n:print(dataset,role,start,"/",len(paths),flush=True)
    if start!=len(raw): raise ValueError("Incomplete patch permutation bank")
    save_tensor(out,file)
    atomic_json(record,{"identity":identity,"native_encoder_max_one_minus_cosine":max_error,
                        "joint_patch_position_reordering_max_one_minus_cosine":equivariance})
    print("ready",dataset,role,start,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--batch",type=int,default=96)
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args()
    for dataset in ("sketchy","tuberlin"):
        cfg,qp,_,pp,_,q,p=global_inputs(args.workspace,dataset)
        model=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda"))
        for role,paths,data in (("gallery",pp,p),("query",qp,q)):
            collect(args.workspace,dataset,role,cfg,paths,data,args.batch,args.workers,model)
