"""Enumerate every way of retaining six disjoint sets of native CLIP patches, leaving retained pixels and positions untouched."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json, read
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .encoding import load_frozen_clip,save_tensor
from .gallery_panel_resume import semantic_inputs
from .recoverable_patch_information import to_tokens
from .retrieval_diagnosis import stratified_sample
from .semantic_neuron_gallery import batches


FAMILY="persistent-drawing-evidence"
GROUP_COUNT=6
PARTITIONS=("grid","appearance","shuffled")


def divide(native):
    b=len(native);device=native.device
    y,x=torch.meshgrid(torch.arange(7,device=device),torch.arange(7,device=device),indexing="ij")
    grid=((y*2//7)*3+x*3//7).flatten().expand(b,-1)
    perm=torch.randperm(49,generator=torch.Generator().manual_seed(60013)).to(device)
    shuffled=grid[:,perm]
    z=F.normalize(native.float().flatten(2).transpose(1,2),dim=-1)
    current=z.new_zeros(b,1,768)
    distance=1-(z*F.normalize(z.mean(1),dim=1)[:,None]).sum(-1)
    chosen=torch.zeros(b,49,dtype=torch.bool,device=device)
    rows=torch.arange(b,device=device)
    axes=[]
    for _ in range(GROUP_COUNT):
        idx=distance.masked_fill(chosen,-torch.inf).argmax(1)
        chosen[rows,idx]=True
        current=z[rows,idx]
        axes.append(current)
        delta=1-(z*current[:,None]).sum(-1)
        distance=delta if len(axes)==1 else torch.minimum(distance,delta)
    centers=torch.stack(axes,1)
    for _ in range(4):
        assignment=(z@centers.transpose(1,2)).argmax(2)
        one=F.one_hot(assignment,GROUP_COUNT).transpose(1,2).float()
        new=F.normalize(one@z,dim=-1)
        centers=torch.where(one.sum(2,keepdim=True)>0,new,centers)
    appearance=(z@centers.transpose(1,2)).argmax(2)
    return torch.stack((grid,appearance,shuffled),1)


def subset_masks(groups):
    subset=torch.arange(1<<GROUP_COUNT,device=groups.device,dtype=torch.int64)
    return ((subset[:,None,None]>>groups[None])&1).bool()


def run_masks(v,t,groups):
    clauses=subset_masks(groups)
    b=len(t)
    fill=t.float().mean((-1,-2),keepdim=True)
    returned=[]
    for mask in clauses.split(8):
        m=mask.reshape(len(mask),b,1,7,7)
        # Mean-token replacement depends only on this image, even for the empty subset.
        tokens=torch.where(m,t[None],fill[None]).flatten(0,1)
        returned.append(to_tokens(v,tokens).reshape(len(mask),b,512))
    return torch.cat(returned).transpose(0,1).contiguous()


@torch.inference_mode()
def extract(w,ds,stage):
    cfg,qs_path,qs,pp,gs,rawq,rawp,*_=semantic_inputs(w,ds)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    if stage=="pilot-query":
        ids=torch.cat([qi[stratified_sample(y,36,63241)] for qi,y in qs.values()]).unique(sorted=True)
        paths=qs_path;baseline=rawq
    else:
        ids=torch.randperm(len(pp),generator=torch.Generator().manual_seed(94175))[:240].sort().values
        paths=pp;baseline=rawp
    root=w/"cache"/FAMILY/ds/stage
    root.mkdir(parents=True,exist_ok=True)
    meta={"model":CHECKPOINT,"paths":digest([str(paths[int(i)]) for i in ids]),"partitions":PARTITIONS,
          "value":"every subset of 6 groups, unobserved tokens replaced by the mean native token for this image","size":len(ids)}
    sig=digest(meta)
    if (root/"manifest.json").is_file():
        if read(root/"manifest.json")["signature"]!=sig:raise ValueError("Different subset features")
        if (root/"done").is_file():return
    atomic_json(root/"manifest.json",meta|{"signature":sig})
    save_tensor({"ids":ids},root/"ids.pt")
    for start in range(0,len(ids),128):
        file=root/f"{start:06}.pt"
        if file.exists():continue
        ix=ids[start:start+128]
        bank,labels,replay=[],[],[]
        for batch in batches(paths,ix,cfg.workers,16):
            pixels=batch.cuda(non_blocking=True).to(model.conv1.weight.dtype)
            token=model.conv1(pixels)
            base=F.normalize(model(pixels).float())
            assignments=divide(token)
            out=torch.stack([run_masks(model,token,assignments[:,i]) for i in range(len(PARTITIONS))],1)
            repeated=(out[:,:,-1]-base[:,None]).norm(dim=-1).max(1).values
            empty=(out[:,:,0]-out[:,:1,0]).norm(dim=-1).max(1).values
            if repeated.max()>.012 or empty.max()>.01:raise ValueError("Full or empty subset readout changes between partitions")
            bank.append(out.half().cpu());labels.append(assignments.byte().cpu());replay.append(torch.stack((repeated,empty),1).cpu())
        z=torch.cat(bank)
        err=(1-(z[:,0,-1].float()*baseline[ix]).sum(1)).abs().max()
        if err>.002:raise ValueError("Different image order")
        save_tensor({"ids":ix,"subsets":z,"groups":torch.cat(labels),"checks":torch.cat(replay)},file)
        print(stage,start+len(ix),len(ids),flush=True)
    (root/"done").touch()


def subset_bank(w,ds,stage):
    folder=w/"cache"/FAMILY/ds/stage
    if not (folder/"done").is_file():raise ValueError("Incomplete subset bank")
    ids=torch.load(folder/"ids.pt",weights_only=True)["ids"]
    files=[torch.load(p,weights_only=True,mmap=True) for p in sorted(folder.glob("[0-9]*.pt"))]
    data={k:torch.cat([d[k] for d in files]) for k in files[0]}
    if not torch.equal(data["ids"],ids):raise ValueError("Unexpected row order")
    return data


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",default="sketchy",choices=("sketchy","tuberlin"))
    p.add_argument("--stage",required=True,choices=("pilot-query","diagnostic-photo"))
    a=p.parse_args()
    extract(a.workspace,a.dataset,a.stage)
