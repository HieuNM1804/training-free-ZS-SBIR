"""Replace specified native MLP units with corresponding activations from real unlabeled gallery patches and replay the same frozen visual tail."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json, read
from .crossdomain_benchmark import digest
from .encoding import load_frozen_clip, save_tensor
from .gallery_panel_resume import semantic_inputs
from .patch_response_mediation import project, read_block
from .retrieval_diagnosis import stratified_sample
from .semantic_neuron_gallery import FAMILY, GROUPS, LAYERS, batches, native, read_donors, to_gpu


DONORS=("late4","late32","scene","early","shuffled","mean","zero")
COUNTS=(8,32)
ARMS=[(l,g,k,d) for l in LAYERS for g in GROUPS for k in COUNTS for d in DONORS]


def code(arm):
    return "/".join(map(str,arm))


def original_scores(source,scales,estimates):
    result=[]
    for l,g,n,d in ARMS:
        z=source["values"][l,g][:,:,:n].float()
        predicted=estimates[l,g][d][...,:n]
        scale=scales[l,g][:n]
        loss=((predicted-z)/scale).square().mean((1,2))
        result.append(loss)
    return torch.stack(result,1)


def embeddings(v,source,units,estimates,arms):
    last_outputs={}
    for l in LAYERS:
        base=source["states"][l][1]
        block=v.transformer.resblocks[l-1]
        fc_out=tuple(block.mlp.children())[-1].weight
        layer_arms=[a for a in arms if a[0]==l]
        for start in range(0,len(layer_arms),8):
            take=layer_arms[start:start+8]
            states=[]
            for arm in take:
                _,group,n,method=arm
                ids=units[l][group][:n]
                old=source["values"][l,group][:,:,:n]
                new=estimates[l,group][method][...,:n]
                delta=F.linear((new-old).float(),fc_out[:,ids].float()).to(base.dtype)
                packed=torch.cat((base[:,:1],base[:,1:]+block.ls_2(delta)),1)
                states.append(packed)
            h=torch.cat(states)
            for later in v.transformer.resblocks[l:]:
                h=read_block(v,later,h)
            zz=project(v,h).reshape(len(take),len(base),512)
            for arm,z in zip(take,zz,strict=True):
                last_outputs[arm]=z-source["scene"]
    return torch.stack([last_outputs[a] for a in arms],1)


def original_tail(v,source):
    errors=[]
    for l in LAYERS:
        h=source["states"][l][1]
        for block in v.transformer.resblocks[l:]:
            h=read_block(v,block,h)
        errors.append((1-(project(v,h)*source["scene"]).sum(1)).abs())
    return torch.stack(errors,1)


@torch.inference_mode()
def extract(w,ds,stage):
    cfg,qp,qs,pp,gs,q,p,*_=semantic_inputs(w,ds)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    v=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    for s,(qi,labels) in qs.items():
        index=w/"cache"/FAMILY/(s+"-memory-v2.pt")
        memo=torch.load(index,weights_only=True,mmap=True)
        bank=to_gpu(memo["table"])
        scales={k:u.flatten(0,1).float().std(0).clamp_min(1e-4) for k,u in bank["values"].items()}
        units={l:{g:idx.cuda() for g,idx in row.items()} for l,row in memo["units"].items()}
        perm=torch.randperm(bank["late"].shape[0]*bank["late"].shape[1],generator=torch.Generator().manual_seed(11213)).cuda()
        if stage=="photo-holdout":
            imageids=memo["holdout"];paths=pp;reference=p
        else:
            which=stratified_sample(labels,36,63241) if stage=="pilot-query" else torch.arange(len(qi))
            imageids=qi[which];paths=qp;reference=q
        root=w/"cache"/FAMILY/s/stage
        root.mkdir(parents=True,exist_ok=True)
        if stage=="all-query":
            freeze=read(w/"results"/FAMILY/"freeze.json")
            keep=set(freeze["input_embeddings"])
            arms=[a for a in ARMS if code(a) in keep]
            if len(arms)!=len(keep) or not keep:raise ValueError("Unrecognized final embedding")
        else:arms=ARMS
        spec={"memory":read(index.with_suffix(".json"))["signature"],"stage":stage,
              "images":digest([str(paths[int(i)]) for i in imageids]),"arms":[code(a) for a in arms],"count":len(imageids)}
        sig=digest(spec)
        if (root/"manifest.json").exists():
            if read(root/"manifest.json")["signature"]!=sig:raise ValueError("Changed intervention protocol")
            if (root/"done").is_file():continue
        atomic_json(root/"manifest.json",spec|{"signature":sig})
        save_tensor({"ids":imageids},root/"ids.pt")
        shard=512 if stage=="all-query" else 128
        for first in range(0,len(imageids),shard):
            path=root/f"{first:06}.pt"
            if path.is_file():continue
            chosen=imageids[first:first+shard]
            batch_size={"all-query":64,"photo-holdout":24,"pilot-query":12}[stage]
            loader=batches(paths,chosen,cfg.workers,batch_size)
            zs,delta,err,loss=[],[],[],[]
            for j,px in enumerate(loader):
                src=native(v,px.cuda(non_blocking=True),units)
                est=read_donors(src,bank,perm)
                zs.append(src["scene"].cpu())
                if stage!="all-query":
                    loss.append(original_scores(src,scales,est).cpu())
                if j==0:err.append(original_tail(v,src).amax(0).cpu())
                if stage!="photo-holdout":delta.append(embeddings(v,src,units,est,arms).half().cpu())
            arr=torch.cat(zs)
            numerical=torch.stack(err).amax(0)
            if numerical.max()>.002 or (1-(arr*reference[chosen]).sum(1)).abs().max()>.002:
                raise ValueError("The untouched network and its replay disagree")
            obj={"ids":chosen,"native":arr,"tail_check":numerical}
            if loss:obj["reconstruction_error"]=torch.cat(loss)
            if delta:obj["change"]=torch.cat(delta)
            save_tensor(obj,path)
            print(s,stage,first+len(arr),len(imageids),flush=True)
        (root/"done").touch()
        del bank


def restore(w,s,stage):
    root=w/"cache"/FAMILY/s/stage
    if not (root/"done").is_file():raise ValueError("Incomplete intervention cache")
    desc=read(root/"manifest.json")
    files=[torch.load(f,mmap=True,weights_only=True) for f in sorted(root.glob("[0-9]*.pt"))]
    data={k:(torch.stack([p[k] for p in files]) if k=="tail_check" else torch.cat([p[k] for p in files])) for k in files[0]}
    ids=torch.load(root/"ids.pt",weights_only=True)["ids"]
    size=512 if stage=="all-query" else 128
    if not torch.equal(data["ids"],ids) or len(files)!=((len(ids)+size-1)//size):raise ValueError("Image or shard order changed")
    return data,desc


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    parser.add_argument("--stage",choices=("pilot-query","photo-holdout","all-query"),required=True)
    a=parser.parse_args()
    extract(a.workspace,a.dataset,a.stage)
