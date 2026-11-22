"""Decompose the observed native ViT-B/32 embedding into additive, directly comparable vectors."""

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

FAMILY="word-attribution-agreement"
BLOCKS=4
HEADS=12


def contributions(vision,pixels):
    hidden=vision._embeds(pixels)
    layers=vision.transformer.resblocks
    if len(layers)!=12 or vision.attn_pool is not None:
        raise ValueError("The additive readout requires the original ViT-B/32 CLS and 12 blocks")
    for block in layers[:-BLOCKS]:
        hidden=block(hidden)
    stem=hidden[:,0].float().clone()
    parts=[]
    ff=[]
    for block in layers[-BLOCKS:]:
        x=block.ln_1(hidden)
        att=block.attn
        if att.num_heads!=HEADS:
            raise ValueError("Wrong number of ViT-B/32 visual heads")
        q,k,v=F.linear(x,att.in_proj_weight,att.in_proj_bias).chunk(3,dim=-1)
        q=q[:,:1]
        q,k,v=(z.unflatten(-1,(HEADS,-1)).transpose(1,2) for z in (q,k,v))
        responses=F.scaled_dot_product_attention(q,k,v).squeeze(2)
        proj=att.out_proj.weight.float().unflatten(1,(HEADS,-1))
        channels=block.ls_1(torch.einsum("bhk,ohk->bho",responses.float(),proj))
        parts.append(channels)
        if att.out_proj.bias is not None:
            stem+=block.ls_1(att.out_proj.bias.float())
        hidden=hidden+block.ls_1(block.attention(q_x=x))
        update=block.ls_2(block.mlp(block.ln_2(hidden)))
        ff.append(update[:,:1].float())
        hidden=hidden+update
    final=hidden[:,0].float()
    raw=torch.cat((*parts,*ff,stem[:,None]),1)
    remainder=final-raw.sum(1)
    raw_error=(remainder.norm(dim=-1)/final.norm(dim=-1).clamp_min(1e-12)).amax()
    raw[:,-1]+=remainder
    norm=vision.ln_post
    std=(final.var(-1,keepdim=True,unbiased=False)+norm.eps).sqrt()
    centered=(raw-raw.mean(-1,keepdim=True))*(norm.weight.float()/std[:,None])
    centered[:,-1]+=norm.bias.float()
    out=centered@vision.proj.float()
    # Reconstruct the actual full output in the same precision as CLIP uses.
    normal=F.normalize((vision.ln_post(hidden[:,0])@vision.proj).float())
    length=out.sum(1).norm(dim=-1).clamp_min(1e-12)
    out/=length[:,None,None]
    direct_err=(out.sum(1)-normal).norm(dim=-1).amax()
    stored=out.half()
    storage_err=(stored.float().sum(1)-normal).norm(dim=-1).amax()
    return stored,normal,torch.stack((raw_error,direct_err,storage_err))


@torch.inference_mode()
def run(w,ds):
    cfg,qp,_,pp,_,qraw,praw=global_inputs(w,ds)
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    folder=w/"cache"/FAMILY/ds
    for name,paths,raw in (("photo",pp,praw),("sketch",qp,qraw)):
        file=folder/(name+".pt")
        desc=file.with_suffix(".json")
        signature=digest({"checkpoint":CHECKPOINT,"paths":digest([str(p) for p in paths]),"blocks":BLOCKS,"heads":HEADS,
                          "decomposition":"direct CLS last four, 48 heads, 4 MLP, complementary stream v1"})
        if file.is_file() and desc.is_file() and read(desc)["signature"]==signature:continue
        out=torch.empty((len(paths),BLOCKS*HEADS+BLOCKS+1,512),dtype=torch.float16)
        loader=DataLoader(Images(paths,normal_transform(),"rgb"),batch_size=184,num_workers=5,pin_memory=True,
                          worker_init_fn=configure_image_worker,prefetch_factor=2)
        pos=0
        worst=torch.zeros(3,device="cuda")
        checked=0.
        for step,data in enumerate(loader):
            x=data.cuda(non_blocking=True).to(vision.conv1.weight.dtype)
            pieces,original,errs=contributions(vision,x)
            worst=torch.maximum(worst,errs)
            if step%28==0:
                direct=F.normalize(vision(x).float())
                checked=max(checked,float((1-(original*direct).sum(1)).abs().max()),
                            float((1-(original*raw[pos:pos+len(x)].cuda()).sum(1)).abs().max()))
                if checked>.002 or worst.amax()>.015:
                    raise ValueError(f"Native decomposition mismatch {checked}, {worst.tolist()}")
                print(name,pos,len(paths),worst.tolist(),flush=True)
            out[pos:pos+len(x)]=pieces.cpu()
            pos+=len(x)
        if pos!=len(paths) or worst.amax()>.015:raise ValueError("Incomplete or inaccurate decomposition")
        save_tensor(out,file)
        atomic_json(desc,{"signature":signature,"rows":pos,"slot_order":"4 blocks × 12 heads, 4 block MLPs, rest",
                          "max_native_1_minus_cos":checked,"max_relative_remainder":float(worst[0]),
                          "max_normalized_reconstruction_error":float(worst[1]),"max_fp16_reconstruction_error":float(worst[2])})
        print("stored",ds,name,pos,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),required=True)
    a=p.parse_args();run(a.workspace,a.dataset)
