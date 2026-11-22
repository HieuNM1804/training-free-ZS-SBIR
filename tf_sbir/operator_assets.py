"""Build reusable frozen image features without consulting the TU-Berlin outcome."""

import argparse
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, read
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .encoding import save_tensor
from .gallery_panel_resume import original_ranker, region_views, semantic_inputs
from .response_geometry_views import encode
from .shared_semantic_operators import FAMILY


@torch.inference_mode()
def run(w,task):
    ds="tuberlin" if task=="tu-gallery" else "sketchy"
    inputs=semantic_inputs(w,ds)
    cfg,qp,qs,pp,gs,q,photo,src,tgt,scale=inputs
    if task=="tu-gallery":
        folder=w/"cache"/FAMILY/"gallery-views"/ds
        folder.mkdir(parents=True,exist_ok=True)
        path=folder/"plan.pt"
        sig=digest({"checkpoint":CHECKPOINT,"photos":list(map(str,pp)),"order":"original gallery file IDs"})
        plan={"order":torch.arange(len(pp)),"priority_count":0}
        if path.exists() and read(path.with_suffix(".json"))["signature"]!=sig:
            raise ValueError("Changed TU photo identifiers")
        if not path.exists():
            save_tensor(plan,path)
            atomic_json(path.with_suffix(".json"),{"signature":sig,"gallery_union":len(pp),"use":"photo encoding only"})
        encode(w,ds,inputs,folder,plan,"all")
    else:
        s="sketchy_1"
        qi,_=qs[s]
        pi,_=gs[s]
        pack,_=region_views(w,s,cfg,qp,qi,q)
        original_ranker(w,s,qi,pi,photo,pp,q,src,tgt,scale,pack)
    print("ready",task,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--task",choices=("tu-gallery","ext1-baseline"),required=True)
    args=p.parse_args()
    run(args.workspace,args.task)
