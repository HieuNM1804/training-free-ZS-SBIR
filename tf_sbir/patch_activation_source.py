"""Which part of the early MLP creates a sign-even discrepancy, and do the same units carry it on independent photos and sketches?"""

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import revision
from .data import Images, configure_image_worker, normal_transform
from .encoding import load_frozen_clip, save_tensor
from .gallery_panel_resume import semantic_inputs
from .patch_mediation_inputs import attention, mlp, random_axes
from .patch_response_mediation import FAMILY, stem
from .recoverable_patch_information import PatchInformation
from .recoverable_patch_inputs import make_index


def allocate(block,a,b):
    fc,activation,last=tuple(block.mlp.children())
    ln=block.ln_2
    shape=a.shape
    def h(x):
        # FP32 accounting of the fixed MLP removes cancellation between the two large native states.
        x=F.layer_norm(x,ln.normalized_shape,ln.weight.float(),ln.bias.float(),ln.eps)
        return F.linear(x,fc.weight.float(),fc.bias.float())
    a,b=a.float(),b.float()
    u,v=h(a),h(b)
    middle=h(.5*(a+b))
    through_mean=activation(.5*(u+v))
    nonlinear=.5*(activation(u)+activation(v))-through_mean
    previous=through_mean-activation(middle)
    deviation=nonlinear+previous
    w=last.weight.float()
    change=F.linear(deviation,w)
    energy=change.square().sum((1,2)).clamp_min(1e-24)
    # Each unit's signed contribution projected onto the actual vector. The sum over units is one.
    costate=change@w
    attr=(deviation*costate).sum(1)/energy[:,None]
    local=(nonlinear*costate).sum((1,2))/energy
    aligned=(previous*costate).sum((1,2))/energy
    err=(attr.sum(1)-1).abs().max()
    if err>1e-3:raise ValueError(f"Unit contributions failed conservation, {float(err):g}")
    # The common comparison is large native units, independently of their response to perturbation.
    native_amp=(activation(middle).square().sum(1)*w.square().sum(0)).clamp_min(0)
    native_amp=native_amp/native_amp.sum(1,keepdim=True).clamp_min(1e-20)
    return attr, torch.stack((local,aligned,energy.sqrt()),1),native_amp


def trace(v,reader,px):
    t=v.conv1(px.to(v.conv1.weight.dtype)).float()
    _,delta,_,_,_=reader.parts(px)
    scale=.005*t.flatten(1).norm(dim=1)[:,None,None,None]/7
    tests=(delta["hidden"],random_axes(t)[0]*scale)
    b=len(px)
    state=stem(v,torch.cat([t+sg*2*d for d in tests for sg in (1,-1)]))
    result={}
    for i,block in enumerate(v.transformer.resblocks[:3],1):
        after=attention(v,block,state)
        if i in (2,3):
            states=after.reshape(4,b,50,768)
            for j,mode in enumerate(("hidden","uniform")):
                result[f"{mode}/{i}"]=allocate(block,*states[2*j:2*j+2])
        state=mlp(block,after)
    return result


def concentration(signed):
    magnitude=signed.abs()
    total=magnitude.sum(1,keepdim=True).clamp_min(1e-12)
    ranks=magnitude.topk(32,1).values/total
    return {k:float(ranks[:,:int(k)].sum(1).median()) for k in ("1","8","32")}


@torch.inference_mode()
def run(w,n):
    inputs=semantic_inputs(w,"sketchy")
    cfg,qp,_,pp,*_=inputs
    pilot=make_index(w,"sketchy",inputs,"pilot-query")
    gen=torch.Generator().manual_seed(65537)
    # Draw calibration photos from all unique Sketchy gallery items without reference to sketches, classes or ranks.
    pids=torch.randperm(len(pp),generator=gen)[:2*n]
    jobs=(("gallery_discovery",pp,pids[:n]),("gallery_holdout",pp,pids[n:]),
          ("sketch",qp,pilot[torch.randperm(len(pilot),generator=gen)[:n]]))
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    v=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    reader=PatchInformation(v)
    out,summary,reference={},{},{}
    for ds,paths,ids in jobs:
        d=Images([paths[int(i)] for i in ids],normal_transform(),"rgb")
        dl=DataLoader(d,batch_size=24,num_workers=cfg.workers,pin_memory=True,worker_init_fn=configure_image_worker)
        scores,metrics,activ=defaultdict(list),defaultdict(list),defaultdict(list)
        for px in dl:
            blocks=trace(v,reader,px.cuda(non_blocking=True))
            for k,(at,de,na) in blocks.items():
                scores[k].append(at.cpu());metrics[k].append(de.cpu());activ[k].append(na.cpu())
        pack={}
        rep={}
        for k in scores:
            signed=torch.cat(scores[k])
            parts=torch.cat(metrics[k])
            amp=torch.cat(activ[k])
            ab=signed.abs()
            norm=ab.sum(1,keepdim=True).clamp_min(1e-12)
            if ds=="gallery_discovery":
                reference[k]=(ab/norm).mean(0).argsort(descending=True,stable=True)[:64]
                reference["native/"+k]=amp.mean(0).argsort(descending=True,stable=True)[:64]
            rankings=reference[k]
            native_order=reference["native/"+k]
            rep[k]={"images":len(ids),"median_GELU_projection":float(parts[:,0].median()),
                    "median_preGELU_projection":float(parts[:,1].median()),"median_local_gap_norm":float(parts[:,2].median()),
                    "per_image_abs_concentration":concentration(signed),"gallery_discovery_units":rankings[:12].tolist(),
                    "gallery_discovery_top8_native":native_order[:8].tolist(),
                    "median_absolute_share_at_discovery_units":{str(size):float((ab[:,rankings[:size]]/norm).sum(1).median()) for size in (1,8,32,64)},
                    "median_absolute_share_at_large_native_units":{str(size):float((ab[:,native_order[:size]]/norm).sum(1).median()) for size in (1,8,32,64)}}
            pack[k]={"signed_unit_projection":signed.half(),"mechanism":parts,"native_activity_share":amp.half()}
        out[ds]={"ids":ids,"readings":pack}
        summary[ds]=rep
        print(ds,[(k,round(x["median_GELU_projection"],3),round(x["median_absolute_share_at_discovery_units"]["8"],3)) for k,x in rep.items()],flush=True)
    path=w/"results"/FAMILY/"activation-source.pt"
    save_tensor(out,path)
    atomic_json(path.with_suffix(".json"),{"revision":revision(),"seed":65537,"per_query_sha256":checksum(path),
                "samples":summary,"unit":"signed projection of each unit contribution onto the full Jensen gap measured before the next block; concentration uses absolute contributions",
                "discovery":"uniform unlabelled gallery subset, separate held-out photos and sketches","device":torch.cuda.get_device_name()})


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    cli.add_argument("--n",type=int,default=240)
    args=cli.parse_args()
    run(args.workspace,args.n)
