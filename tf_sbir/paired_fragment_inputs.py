"""Index single-part views of every target image using identical native patch interventions in both domains."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json, read
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .drawing_evidence_subsets import PARTITIONS,divide
from .encoding import load_frozen_clip,save_tensor
from .gallery_panel_resume import semantic_inputs
from .recoverable_patch_information import to_tokens
from .semantic_neuron_gallery import batches


FAMILY="cross-domain-fragment-likelihood"
PARTS=("grid","appearance","shuffled")
STEP=512


def encode(v,pixels):
    t=v.conv1(pixels.to(v.conv1.weight.dtype))
    assignment=divide(t)
    b=len(pixels)
    fill=t.float().mean((-1,-2),keepdim=True).to(t.dtype)
    variants=[]
    for index in range(len(PARTS)):
        mask=assignment[:,PARTITIONS.index(PARTS[index])]
        keep=torch.arange(6,device=t.device)[:,None,None]==mask[None]
        groups=keep.reshape(6,b,1,7,7)
        x=torch.where(groups,t[None],fill[None]).flatten(0,1)
        r=torch.cat([to_tokens(v,chunk) for chunk in x.split(384)])
        variants.append(r.reshape(6,b,512).transpose(0,1))
    return to_tokens(v,t),torch.stack(variants,1)


@torch.inference_mode()
def run(w,ds,modality):
    cfg,qp,qs,pp,gs,q,p,*_=semantic_inputs(w,ds)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    paths,baseline=(pp,p) if modality=="photo" else (qp,q)
    folder=w/"cache"/FAMILY/ds/modality
    folder.mkdir(parents=True,exist_ok=True)
    signature=digest({"paths":list(map(str,paths)),"model":CHECKPOINT,"partitions":PARTS,"subset":"retain one of six and fill the rest with mean native patch token"})
    if (folder/"protocol.json").is_file():
        if read(folder/"protocol.json")["signature"]!=signature:raise ValueError("Changed photo bank")
        if (folder/"done").is_file():return
    atomic_json(folder/"protocol.json",{"signature":signature,"count":len(paths),"step":STEP,"parts":PARTS,"model":CHECKPOINT})
    v=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    for first in range(0,len(paths),STEP):
        file=folder/f"{first:06}.pt"
        if file.is_file():continue
        ids=torch.arange(first,min(len(paths),first+STEP))
        out,raw=[],[]
        for px in batches(paths,ids,cfg.workers,64):
            whole,z=encode(v,px.cuda(non_blocking=True))
            raw.append(whole.cpu());out.append(z.half().cpu())
        base=torch.cat(raw)
        check=float((1-(base*baseline[ids]).sum(1)).abs().max())
        if check>.002:raise ValueError("Native whole image differs")
        save_tensor({"ids":ids,"partial":torch.cat(out),"native_gap":torch.tensor(check)},file)
        if first%(STEP*2)==0 or first+STEP>=len(paths):print(ds,modality,first+len(ids),len(paths),flush=True)
    (folder/"done").touch()


def read_features(w,ds,modality,ids):
    folder=w/"cache"/FAMILY/ds/modality
    meta=read(folder/"protocol.json")
    if not (folder/"done").is_file():raise ValueError("Incomplete singleton bank")
    out=torch.empty((len(ids),len(PARTS),6,512),dtype=torch.float16)
    for batch in ids.div(STEP,rounding_mode="floor").unique():
        first=int(batch)*STEP
        take=(ids>=first)&(ids<first+STEP)
        file=torch.load(folder/f"{first:06}.pt",weights_only=True,mmap=True)
        pos=ids[take]-first
        if not torch.equal(file["ids"][pos],ids[take]):raise ValueError("Fragment identity mismatch")
        out[take]=file["partial"][pos]
    return out,meta


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    p.add_argument("--modality",choices=("sketch","photo"),default="photo")
    args=p.parse_args()
    run(args.workspace,args.dataset,args.modality)
