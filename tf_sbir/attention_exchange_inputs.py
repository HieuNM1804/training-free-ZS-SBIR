"""Record the exact native CLS queries and candidate patch keys of the frozen CLIP attention heads."""
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json,read
from .crossdomain_benchmark import digest,revision
from .data import Images,configure_image_worker,normal_transform
from .drawing_evidence_subsets import divide
from .encoding import load_frozen_clip,save_tensor
from .query_fragment_reference_inputs import dataset
from .retrieval_diagnosis import stratified_sample


FAMILY="counterfactual-attention-compatibility"
LAYERS=(6,9,12)
SHARD=384


def hidden_start(v,t):
    patches=t.flatten(2).transpose(1,2)
    cls=v.class_embedding.to(patches).expand(len(t),1,-1)
    return v.ln_pre(v.patch_dropout(torch.cat((cls,patches),1)+v.positional_embedding.to(patches)))


def extract(v,t,keys,check=False):
    state=hidden_start(v,t)
    qs=[];ks=[];errs=[]
    for i,block in enumerate(v.transformer.resblocks,1):
        if i in LAYERS:
            a=block.attn;normal=block.ln_1(state)
            if a.num_heads!=12 or normal.size(1)!=50:raise ValueError("Wrong ViT-B/32 attention")
            width=normal.size(-1)
            cq=F.linear(normal[:,:1],a.in_proj_weight[:width],a.in_proj_bias[:width]).reshape(len(t),12,64)
            qs.append(cq)
            if keys:
                pk=F.linear(normal[:,1:],a.in_proj_weight[width:2*width],a.in_proj_bias[width:2*width]).reshape(len(t),49,12,64).transpose(1,2)
                ks.append(pk)
                if check:
                    sample=normal[:2]
                    _,orig=a(sample,sample,sample,need_weights=True,average_attn_weights=False)
                    orig=orig[:,:,0,1:].float();orig/=orig.sum(2,keepdim=True).clamp_min(1e-20)
                    replay=((cq[:2,:,None,:].float()*pk[:2].float()).sum(3)/8).softmax(2)
                    errs.append(float((replay-orig).abs().amax()))
        state=block(state)
    pooled,_=v._pool(state)
    return F.normalize((pooled@v.proj).float()),torch.stack(qs,1),torch.stack(ks,1) if keys else None,max(errs,default=0.)


def partial_queries(v,t):
    n=len(t);dtype=t.dtype
    f=t.float().mean((-1,-2),keepdim=True).to(dtype)
    group=divide(t)[:,1]
    ids=torch.arange(6,device=t.device)[:,None,None]
    masks=(ids==group[None]).reshape(6,n,1,7,7)
    parts=torch.where(masks,t[None],f[None]).flatten(0,1)
    _,q,_,_=extract(v,parts,False)
    _,blank,_,_=extract(v,f.expand_as(t),False)
    return q.reshape(6,n,3,12,64).float().mean(0).half(),blank


@torch.inference_mode()
def run(w,ds,mode):
    if ds=="tuberlin" and not (w/"results"/FAMILY/"freeze.json").exists():raise ValueError("No TU before freeze")
    cfg,qs,views,gs,gviews,q,p=dataset(w,ds)
    photo=mode=="photos"
    paths=gs if photo else qs
    base=p if photo else q
    if mode=="pilot":
        ids=torch.cat([ii[stratified_sample(y,36,63241)] for ii,y in views.values()]).unique(sorted=True)
    else:ids=torch.arange(len(paths))
    root=w/"cache"/FAMILY/ds/mode;root.mkdir(parents=True,exist_ok=True)
    sig=digest({"layers":LAYERS,"paths":[paths[i] for i in ids.tolist()],"version":"frozen CLS projection plus 49 conditional patch keys v1"})
    if (root/"done.json").is_file():
        if read(root/"done.json")["signature"]!=sig:raise ValueError("Different attention cache")
        return
    missing=[j for j in range(0,len(ids),SHARD) if not (root/f"{j:06}.pt").is_file()]
    save_tensor(ids,root/"ids.pt")
    torch.set_float32_matmul_precision("highest")
    v=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    if missing:
        todo=torch.cat([ids[j:j+SHARD] for j in missing])
        loader=DataLoader(Images([paths[i] for i in todo.tolist()],normal_transform(),"rgb"),batch_size=192 if photo else 96,num_workers=cfg.workers,pin_memory=True,persistent_workers=cfg.workers>0,worker_init_fn=configure_image_worker)
        start=0;jj=0;n=0;acc={};checks=[]
        for pix in loader:
            t=v.conv1(pix.cuda(non_blocking=True).to(v.conv1.weight.dtype))
            orig,cq,pk,att_err=extract(v,t,True,start==0)
            ix=todo[start:start+len(pix)];native=float((1-(orig*base[ix].cuda()).sum(1)).abs().amax())
            if native>.002 or att_err>.003:raise ValueError(f"Attention replay failed {native:g}, {att_err:g}")
            result={"q":cq,"k":pk}
            if not photo:result["part"],result["empty"]=partial_queries(v,t)
            for k,val in result.items():acc.setdefault(k,[]).append(val.half().cpu())
            start+=len(pix);n+=len(pix);checks.append((native,att_err))
            pos=missing[jj];length=len(ids[pos:pos+SHARD])
            if n==length:
                save_tensor({"ids":ids[pos:pos+length],**{k:torch.cat(z) for k,z in acc.items()},"checks":torch.tensor(checks).amax(0)},root/f"{pos:06}.pt")
                if jj%10==0 or jj+1==len(missing):print(ds,mode,start,len(todo),flush=True)
                jj+=1;n=0;acc={};checks=[]
            elif n>length:raise ValueError("Cache block mismatch")
        if n or jj!=len(missing):raise ValueError("Partial cache")
    atomic_json(root/"done.json",{"signature":sig,"images":len(ids),"code":revision(),"device":torch.cuda.get_device_name()})


def layer(w,ds,mode,l):
    root=w/"cache"/FAMILY/ds/mode
    manifest=read(root/"done.json");ids=torch.load(root/"ids.pt",weights_only=True)
    if len(ids)!=manifest["images"]:raise ValueError("Missing input")
    files=[torch.load(f,weights_only=True,mmap=True) for f in sorted(root.glob("[0-9]*.pt"))]
    if not torch.equal(ids,torch.cat([d["ids"] for d in files])):raise ValueError("Attention cache order changed")
    j=LAYERS.index(l);names=("q","k") if mode=="photos" else ("q","k","part","empty")
    return ids,{name:torch.cat([d[name][:,j] for d in files]).cuda() for name in names},torch.stack([d["checks"] for d in files]).amax(0)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    ap.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    ap.add_argument("--mode",choices=("photos","pilot","full"),required=True)
    a=ap.parse_args();run(a.workspace,a.dataset,a.mode)
