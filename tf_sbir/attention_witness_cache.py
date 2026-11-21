"""Cache native CLS queries and RGB patch keys/values from the frozen CLIP visual blocks."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json,read
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .data import Images,configure_image_worker,normal_transform
from .encoding import load_frozen_clip
from .retrieval_diagnosis import global_inputs


LAYERS=(7,11)
FAMILY="counterfactual-patch-witness"


def bank_path(w,dataset,role):
    return w/"cache"/FAMILY/dataset/role


def extract(w,dataset,role,paths,features,checkpoint,batch_size,workers):
    folder=bank_path(w,dataset,role)
    folder.mkdir(parents=True,exist_ok=True)
    stamp=folder/"ready.json"
    expected={"paths":digest([str(x) for x in paths]),"checkpoint":CHECKPOINT,"count":len(paths),
              "zero_based_layers":list(LAYERS),"rgb_size":224,"storage":"fp16-v1"}
    if stamp.is_file() and read(stamp)["identity"]==expected:
        if all((folder/f"{field}-{layer}.npy").is_file() for layer in LAYERS for field in ("reader","patch")):
            return
    model=load_frozen_clip(checkpoint,torch.device("cuda"))
    memory={}
    for layer in LAYERS:
        for name,tail in (("reader",(12,64)),("patch",(12,49,2,64))):
            memory[layer,name]=np.lib.format.open_memmap(folder/f"{name}-{layer}.tmp",
                mode="w+",dtype=np.float16,shape=(len(paths),*tail))
    tensors={}
    def hook(layer):
        def save(block,args):
            x=block.ln_1(args[0])
            if x.shape[1:]!=(50,768):
                raise ValueError("Unexpected native token layout")
            q,k,v=F.linear(x,block.attn.in_proj_weight,block.attn.in_proj_bias).chunk(3,dim=-1)
            reader=q[:,0].reshape(-1,12,64)
            key=k[:,1:].reshape(-1,49,12,64).permute(0,2,1,3)
            val=v[:,1:].reshape(-1,49,12,64).permute(0,2,1,3)
            tensors[layer]=(reader,torch.stack((key,val),dim=-2))
        return save
    handles=[model.visual.transformer.resblocks[i].register_forward_pre_hook(hook(i)) for i in LAYERS]
    loader=DataLoader(Images(paths,normal_transform(),"rgb"),batch_size=batch_size,num_workers=workers,
                      pin_memory=True,worker_init_fn=configure_image_worker,prefetch_factor=2 if workers else None)
    start=0
    worst=0.
    try:
        with torch.inference_mode():
            for x in loader:
                y=F.normalize(model.encode_image(x.to("cuda",dtype=model.visual.conv1.weight.dtype,non_blocking=True)).float())
                end=start+len(x)
                err=float((1-(y*features[start:end].to(y)).sum(1)).abs().max())
                worst=max(worst,err)
                if worst>.002:
                    raise ValueError("Token cache image order or frozen encoder does not match original inputs")
                for layer,(r,kv) in tensors.items():
                    memory[layer,"reader"][start:end]=r.half().cpu().numpy()
                    memory[layer,"patch"][start:end]=kv.half().cpu().numpy()
                start=end
                if start%4096<len(x):
                    print(dataset,role,start,"/",len(paths),flush=True)
    finally:
        for h in handles: h.remove()
    if start!=len(paths):
        raise ValueError("Incomplete feature store")
    for (layer,field),arr in memory.items():
        arr.flush()
        os.replace(folder/f"{field}-{layer}.tmp",folder/f"{field}-{layer}.npy")
    atomic_json(stamp,{"identity":expected,"maximum_one_minus_cached_clip_cosine":worst})
    print("ready",dataset,role,start,worst,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--batch-size",type=int,default=160)
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args()
    for dataset in ("sketchy","tuberlin"):
        cfg,qp,_,pp,_,q,p=global_inputs(args.workspace,dataset)
        for role,paths,features in (("gallery",pp,p),("query",qp,q)):
            extract(args.workspace,dataset,role,paths,features,cfg.clip_checkpoint,args.batch_size,args.workers)
