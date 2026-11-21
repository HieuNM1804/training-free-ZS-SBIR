"""Compare retention, deletion and reframing of the same region in frozen CLIP."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json, read
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .data import Images, configure_image_worker, normal_transform
from .encoding import load_frozen_clip, save_tensor

FAMILY="regional-evidence-interventions"
SIDES=(3,5)


def windows(side):
    return [(y,x) for y in range(0,8-side,2) for x in range(0,8-side,2)]


def native(vision, tokens):
    hidden=vision.transformer(tokens)
    pooled,_=vision._pool(hidden)
    return F.normalize((pooled@vision.proj).float())


def make_views(vision, pixels):
    pixels=pixels.cuda(non_blocking=True)
    dtype=vision.conv1.weight.dtype
    patches=vision.conv1(pixels.to(dtype)).flatten(2).transpose(1,2)
    cls=vision.class_embedding.to(patches).expand(len(patches),1,-1)
    all_tokens=vision.ln_pre(torch.cat((cls,patches),1)+vision.positional_embedding.to(patches))
    grid=torch.arange(49,device=patches.device).reshape(7,7)
    outputs=[]
    for side in SIDES:
        cells=windows(side)
        keep=torch.stack([grid[y:y+side,x:x+side].flatten()+1 for y,x in cells])
        removed=torch.stack([grid.flatten()[~torch.isin(grid.flatten()+1,row)]+1 for row in keep])
        for start in range(0,len(cells),3):
            a,b=keep[start:start+3],removed[start:start+3]
            col=torch.zeros((len(a),1),device=a.device,dtype=a.dtype)
            keep_seq=all_tokens[:,torch.cat((col,a),1)].flatten(0,1)
            drop_seq=all_tokens[:,torch.cat((col,b),1)].flatten(0,1)
            vk=native(vision,keep_seq).unflatten(0,(len(pixels),len(a)))
            vd=native(vision,drop_seq).unflatten(0,(len(pixels),len(a)))
            tiles=torch.stack([pixels[:,:,32*y:32*(y+side),32*x:32*(x+side)] for y,x in cells[start:start+3]],1)
            zoom=F.interpolate(tiles.flatten(0,1),size=(224,224),mode="bicubic",align_corners=False,antialias=True)
            vz=F.normalize(vision(zoom.to(dtype)).float()).unflatten(0,(len(pixels),len(a)))
            outputs.append(torch.stack((vk,vd,vz),2))
    return torch.cat(outputs,1),native(vision,all_tokens)


def obtain(w, dataset, cfg, paths, raw, ids, batch=56):
    folder=w/"cache"/FAMILY/dataset
    file=folder/"regions.pt"
    info=folder/"regions.json"
    sig={"checkpoint":CHECKPOINT,"paths":digest([str(p) for p in paths]),"sizes":SIDES,"stride":2}
    h=digest(sig)
    old_ids=torch.empty(0,dtype=torch.long)
    old=None
    if file.exists() and info.exists() and read(info)["signature"]==h:
        blob=torch.load(file,weights_only=True,mmap=True)
        old_ids,old=blob["ids"],blob["features"]
    ids=ids.unique(sorted=True)
    missing=ids[~torch.isin(ids,old_ids)]
    if len(missing):
        model=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda"))
        subset=[paths[i] for i in missing.tolist()]
        loader=DataLoader(Images(subset,normal_transform(),"rgb"),batch_size=batch,num_workers=4,
                          pin_memory=True,worker_init_fn=configure_image_worker,prefetch_factor=2)
        chunks=[]
        offset=0
        error=0.
        for batch_number,pixels in enumerate(loader):
            features,original=make_views(model.visual,pixels)
            keys=missing[offset:offset+len(pixels)]
            error=max(error,float((1-(original*raw[keys].cuda()).sum(1)).abs().max()))
            if error>.002:raise ValueError("Full visual forward did not reproduce the source sketch")
            chunks.append(features.half().cpu())
            offset+=len(pixels)
            if batch_number%60==0:
                print("regions",dataset,offset,"/",len(missing),flush=True)
        if offset!=len(missing):raise ValueError("Incomplete intervention")
        all_ids=torch.cat((old_ids,missing))
        order=all_ids.argsort()
        old_ids=all_ids[order]
        old=torch.cat(([old] if old is not None else [])+chunks)[order]
        save_tensor({"ids":old_ids,"features":old},file)
        atomic_json(info,{"signature":h,"native_one_minus_cosine":error,"count":len(old_ids)})
    positions=torch.searchsorted(old_ids,ids)
    if not torch.equal(old_ids[positions],ids):raise ValueError("Cache index mismatch")
    return old[positions]


def regional_logits(query, views, source, scale, kind, rho, power, temperature):
    full=(query@source.T)*scale
    if kind=="native":return full
    z=(views.float()@source.T)*scale
    amplitudes=[]
    agreements=[]
    for indices in (slice(0,9),slice(9,13)):
        keep,drop,zoom=z[:,indices].unbind(2)
        k=(temperature*keep).softmax(1)
        d=(-temperature*drop).softmax(1)
        n=keep.shape[1]
        if kind=="shift": d=d.roll(1,1)
        weight=k*d
        a=n*weight.sum(1)
        if kind=="zoom":weight=torch.ones_like(k)
        elif kind=="keep":weight=k
        elif kind=="drop":weight=d
        elif kind not in ("joint","shift"):raise ValueError(kind)
        weight=weight/weight.sum(1,keepdim=True).clamp_min(1e-30)
        amp=torch.logsumexp(zoom+weight.clamp_min(1e-30).log(),1)
        amplitudes.append(amp)
        agreements.append(a.clamp_min(1e-12).log())
    pooled=torch.stack(amplitudes).logsumexp(0)-full.new_tensor(2.).log()
    concordance=torch.stack(agreements).mean(0) if kind in ("joint","shift") else 0.
    return (1-rho)*full+rho*pooled+power*concordance


def null_check(views,source,scale):
    z=(views[:4].float()@source[:13].T)*scale
    errs=[]
    for ids in (slice(0,9),slice(9,13)):
        k=z[:,ids,0].double().softmax(1)
        d=(-z[:,ids,1].double()).softmax(1)
        avg=sum((k*d.roll(s,1)).sum(1) for s in range(k.size(1)))
        errs.append(float((avg-1).abs().max()))
    if max(errs)>1e-10:raise ValueError("Wrong finite-location null")
    return max(errs)
