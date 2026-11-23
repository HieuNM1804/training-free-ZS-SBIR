"""Capture paired coherent and permuted views of the same six source fragments, including image-specific empty controls."""

import argparse
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .benchmark import merge_inputs
from .concept_benchmark import atomic_json,read
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT,runtime
from .data import CLIP_MEAN,CLIP_STD,Images,SKETCHY_SPLITS,configure_image_worker,normal_transform
from .drawing_evidence_subsets import divide
from .encoding import load_frozen_clip,load_global_inputs,save_tensor
from .recoverable_patch_information import to_tokens
from .retrieval_diagnosis import stratified_sample


FAMILY="query-fragment-reference"
MODES=("appearance","area_shuffle","grid","permuted_fragments_A","permuted_fragments_B",
       "permuted_whole_A","permuted_whole_B","empty_token","mean_colour","fixed_white","fixed_neutral")
SEEDS=(40487,60251)


def dataset(w,ds):
    workers=min(8,len(os.sched_getaffinity(0)))
    cfg=runtime(w,ds,batch=640,workers=workers)
    splits=SKETCHY_SPLITS if ds=="sketchy" else ("tuberlin",)
    qp,qs=merge_inputs(cfg.root,splits,"sketch")
    pp,gs=merge_inputs(cfg.root,splits,"photo")
    q,p,_=load_global_inputs(cfg,qp,pp,False)
    return cfg,qp,qs,pp,gs,q,p


def permutation(seed,device):
    return torch.randperm(49,generator=torch.Generator().manual_seed(seed)).to(device)


def shuffle_locations(tokens,idx):
    return tokens.flatten(-2).index_select(-1,idx).reshape_as(tokens)


def encode(v,pixels,modes):
    n=len(pixels)
    dtype=v.conv1.weight.dtype
    x=pixels.to(dtype)
    base=v.conv1(x)
    if base.shape[1:]!=(768,7,7):raise ValueError("Wrong native tokenization")
    fill=base.float().mean((-1,-2),keepdim=True).to(dtype)
    groups=divide(base)
    appearance=groups[:,1]
    device=base.device
    shuffles=[permutation(seed,device) for seed in SEEDS]
    arange=torch.arange(6,device=device)[:,None,None]
    def isolate(ids):
        selector=(arange==ids[None]).reshape(6,n,1,7,7)
        return torch.where(selector,base[None],fill[None])
    true=isolate(appearance)
    def features(t):
        # Bound peak early-transformer activation memory; the cap is identical for controls.
        t=t.reshape(-1,768,7,7)
        z=torch.cat([to_tokens(v,part) for part in t.split(768)])
        return z.reshape(-1,n,512).transpose(0,1)
    obs={}
    for name in modes:
        if name=="appearance":tokens=true
        elif name=="area_shuffle":tokens=isolate(appearance[:,permutation(60013,device)])
        elif name=="grid":tokens=isolate(groups[:,0])
        elif name.startswith("permuted_fragments"):
            j=0 if name.endswith("_A") else 1
            tokens=shuffle_locations(true,shuffles[j])
        elif name.startswith("permuted_whole"):
            j=0 if name.endswith("_A") else 1
            tokens=shuffle_locations(base,shuffles[j])[None]
        elif name=="empty_token":tokens=fill.expand_as(base)[None]
        elif name=="mean_colour":
            uniform=pixels.float().mean((-1,-2),keepdim=True).to(dtype).expand_as(x)
            tokens=v.conv1(uniform.contiguous())[None]
        elif name=="fixed_white":
            white=(1-pixels.new_tensor(CLIP_MEAN))/pixels.new_tensor(CLIP_STD)
            z=v.conv1(white.to(dtype)[None,:,None,None].expand(1,3,224,224).contiguous())
            obs[name]=to_tokens(v,z).expand(n,1,512);continue
        elif name=="fixed_neutral":
            z=torch.zeros_like(base[:1])
            obs[name]=to_tokens(v,z).expand(n,1,512);continue
        else:raise ValueError(name)
        obs[name]=features(tokens)
    full=to_tokens(v,base)
    # Merging the six complementary token selections reconstructs the original before the transformer.
    error=(true.float().sum(0)-5*fill.float()-base.float()).abs().amax()
    if error>4e-4:raise ValueError(f"The partial views do not reassemble: {float(error):g}")
    return full,obs,float(error)


def selected_modes(w,ds,stage):
    if stage=="pilot":return list(MODES)
    name="shortlist.json" if ds=="sketchy" else "freeze.json"
    frozen=read(w/"results"/FAMILY/name)
    if digest(frozen["config"])!=frozen["sha256"]:raise ValueError("Invalid input configuration")
    return frozen["config"]["input_modes"]


@torch.inference_mode()
def run(w,ds,stage,*,requested_modes=None,folder_family=FAMILY):
    cfg,paths,qs,_,_,baseline,_=dataset(w,ds)
    if stage=="pilot" and ds!="sketchy":raise ValueError("No TU pilot or tuning")
    ids=torch.cat([qi[stratified_sample(y,36,63241)] for qi,y in qs.values()]).unique(sorted=True) if stage=="pilot" else torch.arange(len(paths))
    modes=selected_modes(w,ds,stage) if requested_modes is None else sorted(requested_modes)
    out=w/"cache"/folder_family/ds/stage
    out.mkdir(parents=True,exist_ok=True)
    sig=digest({"model":CHECKPOINT,"paths":[str(paths[int(i)]) for i in ids],"operation":"mean native patch fill, exact area control, paired common spatial permutations v1",
                "modes":modes,"seeds":SEEDS})
    if (out/"manifest.json").is_file():
        if read(out/"manifest.json")["signature"]!=sig:raise ValueError("Changed query reference bank")
        if (out/"done").exists():return
    atomic_json(out/"manifest.json",{"signature":sig,"modes":modes,"images":len(ids),"shard":512,"gpu":torch.cuda.get_device_name()})
    save_tensor({"ids":ids},out/"ids.pt")
    visual=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    missing=[i for i in range(0,len(ids),512) if not (out/f"{i:06}.pt").is_file()]
    if missing:
        seq=torch.cat([ids[i:i+512] for i in missing])
        loader=DataLoader(Images([paths[int(i)] for i in seq],normal_transform(),"rgb"),batch_size=128,num_workers=cfg.workers,
                          pin_memory=True,worker_init_fn=configure_image_worker,persistent_workers=cfg.workers>0)
        start=0;sh=0;used=0
        acc={k:[] for k in modes};ori=[];errs=[]
        for pix in loader:
            global_ids=seq[start:start+len(pix)];start+=len(pix)
            z,options,algebra=encode(visual,pix.cuda(non_blocking=True),modes)
            error=float((1-(z*baseline[global_ids].cuda()).sum(1)).abs().max())
            if error>.002:raise ValueError("Native embedding disagrees with the original cached CLIP")
            for name,arr in options.items():acc[name].append(arr.half().cpu())
            ori.append(z.cpu());errs.append((error,algebra));used+=len(pix)
            first=missing[sh]
            needed=len(ids[first:first+512])
            if used==needed:
                save_tensor({"ids":ids[first:first+512],"native":torch.cat(ori),"modes":{k:torch.cat(v) for k,v in acc.items()},
                             "checks":torch.tensor(errs).amax(0)},out/f"{first:06}.pt")
                if sh%4==0 or sh+1==len(missing):print(ds,stage,first+used,len(ids),flush=True)
                sh+=1;used=0;acc={k:[] for k in modes};ori=[];errs=[]
            elif used>needed:raise ValueError("Shard boundary mismatch")
        if sh!=len(missing) or used:raise ValueError("Incomplete query responses")
    (out/"done").touch()


def load(w,ds,stage,folder_family=FAMILY):
    root=w/"cache"/folder_family/ds/stage
    if not (root/"done").is_file():raise ValueError("Reference bank incomplete")
    meta=read(root/"manifest.json")
    files=[torch.load(f,mmap=True,weights_only=True) for f in sorted(root.glob("[0-9]*.pt"))]
    ids=torch.load(root/"ids.pt",weights_only=True)["ids"]
    if not torch.equal(ids,torch.cat([f["ids"] for f in files])):raise ValueError("Reference bank image order changed")
    return {"ids":ids,"native":torch.cat([f["native"] for f in files]),
            "modes":{k:torch.cat([f["modes"][k] for f in files]) for k in meta["modes"]},
            "checks":torch.stack([f["checks"] for f in files]).amax(0)}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    p.add_argument("--stage",choices=("pilot","full"),required=True)
    a=p.parse_args()
    run(a.workspace,a.dataset,a.stage)
