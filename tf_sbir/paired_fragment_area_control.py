"""Break semantic membership without changing the precise number of native patches retained by any fragment."""

import argparse
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,read
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .drawing_evidence_subsets import PARTITIONS,divide
from .encoding import load_frozen_clip,save_tensor
from .gallery_panel_resume import semantic_inputs
from .paired_fragment_inputs import FAMILY,STEP
from .recoverable_patch_information import to_tokens
from .retrieval_diagnosis import stratified_sample
from .semantic_neuron_gallery import batches


def measure(v,px):
    t=v.conv1(px.to(v.conv1.weight.dtype))
    b=len(t)
    fill=t.float().mean((-1,-2),keepdim=True).to(t.dtype)
    groups=divide(t)[:,PARTITIONS.index("appearance")]
    perm=torch.randperm(49,generator=torch.Generator().manual_seed(60013)).to(groups.device)
    null=groups[:,perm]
    ids=torch.arange(6,device=t.device)[:,None,None]
    real=(ids==groups[None])
    shuffled=(ids==null[None])
    if not torch.equal(shuffled.sum(2),real.sum(2)):raise ValueError("The per-image areas do not match")
    patches=torch.where(shuffled.reshape(6,b,1,7,7),t[None],fill[None]).flatten(0,1)
    z=torch.cat([to_tokens(v,sub) for sub in patches.split(384)])
    return z.reshape(6,b,512).transpose(0,1),real.sum(2).T


@torch.inference_mode()
def run(w,ds,stage):
    inputs=semantic_inputs(w,ds)
    cfg,qp,qs,pp,*_=inputs
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    v=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    if stage=="photo":
        ids=torch.arange(len(pp));paths=pp
    else:
        ids=torch.cat([qi[stratified_sample(y,36,63241)] for qi,y in qs.values()]).unique(sorted=True) if stage=="pilot-query" else torch.arange(len(qp))
        paths=qp
    folder=w/"cache"/FAMILY/ds/("area-"+stage)
    folder.mkdir(parents=True,exist_ok=True)
    sig=digest({"paths":[str(paths[int(i)]) for i in ids],"model":CHECKPOINT,"operation":"spatially permuted native appearance group membership, seed 60013"})
    if (folder/"protocol.json").is_file():
        if read(folder/"protocol.json")["signature"]!=sig:raise ValueError("Changed area control")
        if (folder/"done").exists():return
    atomic_json(folder/"protocol.json",{"signature":sig,"count":len(ids)})
    save_tensor({"ids":ids},folder/"ids.pt")
    for first in range(0,len(ids),STEP):
        file=folder/f"{first:06}.pt"
        if file.is_file():continue
        chosen=ids[first:first+STEP]
        out,sizes=[],[]
        for x in batches(paths,chosen,cfg.workers,96):
            z,size=measure(v,x.cuda(non_blocking=True))
            out.append(z.half().cpu());sizes.append(size.byte().cpu())
        save_tensor({"ids":chosen,"partial":torch.cat(out),"areas":torch.cat(sizes)},file)
        if first%(STEP*4)==0 or first+STEP>=len(ids):print(stage,"area preserved",first+len(chosen),len(ids),flush=True)
    (folder/"done").touch()


def read_area(w,ds,stage,ids):
    folder=w/"cache"/FAMILY/ds/("area-"+stage)
    if not (folder/"done").is_file():raise ValueError("Area-preserving comparison is unfinished")
    known=torch.load(folder/"ids.pt",weights_only=True)["ids"]
    loc=torch.searchsorted(known,ids.contiguous())
    if (loc>=len(known)).any() or not torch.equal(known[loc],ids):raise ValueError("Unknown fragment image")
    out=torch.empty((len(ids),6,512),dtype=torch.float16)
    for j in loc.div(STEP,rounding_mode="floor").unique():
        start=int(j)*STEP
        use=(loc>=start)&(loc<start+STEP)
        item=torch.load(folder/f"{start:06}.pt",mmap=True,weights_only=True)
        pos=loc[use]-start
        if not torch.equal(ids[use],item["ids"][pos]):raise ValueError("Ordering of control views changed")
        out[use]=item["partial"][pos]
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    parser.add_argument("--stage",choices=("photo","pilot-query","all-query"),required=True)
    a=parser.parse_args()
    run(a.workspace,a.dataset,a.stage)
