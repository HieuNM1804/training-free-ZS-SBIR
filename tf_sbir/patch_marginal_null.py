"""Measure what the frozen CLIP representation retains under different joint distributions with the same patch marginal."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import inspect
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json,checksum,read
from .conditional_agreement_study import Replay
from .crossdomain_benchmark import digest,revision
from .crossdomain_inputs import CHECKPOINT
from .data import Images,configure_image_worker,normal_transform
from .encoding import load_frozen_clip,save_tensor
from .map_diagnosis import map_values
from .query_fragment_likelihood import GalleryWalk
from .query_fragment_reference_inputs import dataset,load
from .query_fragment_reference_study import split_context
from .quickdraw_locked import KD_FUNCTION_SHA
from .recoverable_patch_information import to_tokens
from .retrieval_evidence import load_kd,paired_delta


FAMILY="patch-marginal-null"
N=49
DRAWS=16
SEED=101249
PRIMARY="iid/0.8/walk"


def designs():
    g=torch.Generator().manual_seed(SEED)
    perm=torch.stack([torch.randperm(N,generator=g) for _ in range(DRAWS)])
    iid=torch.randint(N,(DRAWS,N),generator=g)
    tiled=torch.arange(N)[:,None].expand(-1,N)
    if not torch.equal(perm.sort(1).values,torch.arange(N).expand_as(perm)):raise ValueError("Unbalanced permuted controls")
    return {"iid":iid,"permuted":perm,"tiled":tiled}


def encode(visual,pixels,plans):
    base=visual.conv1(pixels.to(visual.conv1.weight.dtype))
    flat=base.flatten(2);b=len(base)
    z=to_tokens(visual,base)
    rows={};uncertainty={}
    for name,indices in plans.items():
        chunks=[]
        for idx in indices.split(6):
            seq=flat[:,:,idx].permute(2,0,1,3).reshape(len(idx)*b,flat.size(1),7,7)
            chunks.append(to_tokens(visual,seq).reshape(len(idx),b,-1))
        vectors=torch.cat(chunks).float()
        rows[name]=vectors.mean(0)
        if name!="tiled":
            first,second=vectors.chunk(2)
            uncertainty[name]=F.cosine_similarity(first.mean(0),second.mean(0)).cpu()
    return z,rows,uncertainty


@torch.inference_mode()
def run(w):
    root=w/"results"/FAMILY;root.mkdir(parents=True,exist_ok=True)
    cache=w/"cache"/FAMILY/"sketchy";cache.mkdir(parents=True,exist_ok=True)
    cfg,qfiles,qs,pfiles,gs,q,p=inputs=dataset(w,"sketchy");reference=load(w,"sketchy","pilot")
    plans={name:x.cuda() for name,x in designs().items()}
    spec={"code":revision(),"checkpoint":CHECKPOINT,"seed":SEED,"draws":DRAWS,"primary":PRIMARY,"evaluator":KD_FUNCTION_SHA,
          "queries_per_class":36,"coefficient":[.8,1.],"temperature":64.,"matched_six_group_reference":True}
    sig=digest(spec)
    if (root/"protocol.json").is_file() and read(root/"protocol.json")["signature"]!=sig:raise ValueError("Incompatible pilot")
    atomic_json(root/"protocol.json",spec|{"signature":sig,"device":torch.cuda.get_device_name()})
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ids=reference["ids"];visual=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    for first in range(0,len(ids),512):
        dest=cache/f"{first:05}.pt"
        if dest.is_file():continue
        ix=ids[first:first+512]
        loader=DataLoader(Images([qfiles[int(i)] for i in ix],normal_transform(),"rgb"),batch_size=96,num_workers=4,worker_init_fn=configure_image_worker,pin_memory=True)
        responses=defaultdict(list);error=defaultdict(list);seen=0
        for pix in loader:
            z,values,half=encode(visual,pix.cuda(non_blocking=True),plans)
            if (1-(z*q[ix[seen:seen+len(pix)]].cuda()).sum(1)).abs().amax()>.002:raise ValueError("Changed original encoding")
            for name,t in values.items():responses[name].append(t.cpu())
            for name,t in half.items():error[name].append(t)
            seen+=len(pix)
        if seen!=len(ix):raise ValueError("Partial pilot bank")
        save_tensor({"ids":ix,"responses":{k:torch.cat(v) for k,v in responses.items()},"half":{k:torch.cat(v) for k,v in error.items()}},dest)
        print("encoded",first+seen,len(ids),flush=True)
    del visual
    files=[torch.load(f,mmap=True,weights_only=True) for f in sorted(cache.glob("*.pt"))]
    if not torch.equal(ids,torch.cat([f["ids"] for f in files])):raise ValueError("Query order changed")
    ref={k:torch.cat([file["responses"][k] for file in files]) for k in plans}
    ref["six"]=reference["modes"]["appearance"].float().mean(1)
    ref["empty"]=reference["modes"]["empty_token"][:,0].float()
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Wrong KD evaluator")
    for s in qs:
        dest=root/("pilot-"+s+".json")
        if dest.exists():
            info=read(dest)
            if info["signature"]!=sig or checksum(dest.with_suffix(".pt"))!=info["per_query_sha256"]:raise ValueError("Incompatible saved scores")
            continue
        data=split_context(w,s,"sketchy",inputs,reference,"pilot",False)
        g=data["gallery"];label=data["gl"];gt=label.cuda();graph=GalleryWalk(g)
        replay=Replay(kd,label,s);err=0.;scores=defaultdict(list);tops=defaultdict(list)
        with ThreadPoolExecutor(max_workers=min(5,len(os.sched_getaffinity(0)))) as pool:
            for first in range(0,len(data["ids"]),64):
                sl=slice(first,first+64);y=data["labels"][sl];x=data["q"][sl];pos=data["pos"][sl]
                for name,values in ref.items():
                    m=values[pos].cuda()
                    for beta in (.8,1.):
                        raw=(x-beta*m)@g.T
                        for order,mat in (("direct",raw),("walk",graph.log(64.*raw))):
                            tag=f"{name}/{beta:g}/{order}"
                            v,rank,good,_=map_values(mat,y.cuda(),gt,s)
                            exact=replay.score(rank,y,pool)
                            err=max(err,float((exact-v.cpu()).abs().amax()))
                            if err>2e-6:raise ValueError("KD-SBIR and vectorized metrics differ")
                            scores[tag].append(exact);tops[tag].append(good[:,0].cpu())
        values={k:torch.cat(v) for k,v in scores.items()};top={k:torch.cat(v) for k,v in tops.items()}
        cls=data["labels"].unique(return_inverse=True)[1]
        save_tensor({"ids":data["ids"],"labels":data["labels"],"metrics":values,"top1":top},dest.with_suffix(".pt"))
        atomic_json(dest,{"signature":sig,"queries":len(cls),"gallery":len(gt),"classes":len(cls.unique()),"KD_max_error":err,
                          "original_evaluator":True,"values":{k:v.double().mean(0).tolist() for k,v in values.items()},
                          "paired":{k:paired_delta(values[PRIMARY],v,cls) for k,v in values.items()},
                          "split_half_cosine":{k:float(torch.cat([f["half"][k] for f in files])[data["pos"]].mean()) for k in ("iid","permuted")},
                          "per_query_sha256":checksum(dest.with_suffix(".pt"))})
        print("saved",s,flush=True)
        del data,graph


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
