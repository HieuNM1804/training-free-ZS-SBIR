"""Schedule query-independent photo view extraction, prioritizing matched error pairs for diagnosis."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json, read
from .cross_view_region_cache import two_pose_regions
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .data import Images, configure_image_worker, normal_transform
from .encoding import load_frozen_clip, save_tensor
from .gallery_panel_resume import region_views, semantic_inputs
from .predictive_depiction_study import baseline
from .retrieval_diagnosis import stratified_sample


FAMILY="paired-concept-response-geometry"
PAIR_NAMES=("rank_one_and_best_opponent","first_wrong_above_right","first_right_above_wrong")
SHARD=384


def study_base(w,s,inputs,ids):
    existing=w/"cache/predictive-depiction-distance"/(s+"-regional-"+digest(ids.tolist())[:16]+".pt")
    return torch.load(existing,weights_only=True,mmap=True) if existing.is_file() else baseline(w,s,inputs,ids)


def pairs(base,y,g,pi):
    panels=[]
    for x,ys in zip(base.split(64),y.split(64),strict=True):
        order=x.cuda().argsort(dim=1,descending=True,stable=True)
        labels=g.cuda()[order]==ys.cuda()[:,None]
        best_pos=labels.int().argmax(1)
        best_neg=(~labels).int().argmax(1)
        row=torch.arange(len(x),device="cuda")
        first=torch.stack((order[row,best_pos],order[row,best_neg]),1)
        contents=[first]
        for want in (False,True):
            left=labels[:,:199]==want
            right=labels[:,1:200]!=want
            ok=left&right
            j=ok.int().argmax(1)
            p,n=(j,j+1) if want else (j+1,j)
            ordered=torch.stack((order[row,p],order[row,n]),1)
            contents.append(torch.where(ok.any(1)[:,None],ordered,-1))
        local=torch.stack(contents,1).cpu()
        out=torch.where(local>=0,pi[local.clamp_min(0)],-1)
        panels.append(out)
    return torch.cat(panels)


def inputs_plan(w,ds,inputs):
    cfg,qp,qs,pp,gs,q,p,src,tgt,scale=inputs
    folder=w/"cache"/FAMILY/ds
    folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/"plan.pt"
    signature=digest({"photos":list(map(str,pp)),"queries":list(map(str,qp)),"seed":63241,
                      "model":CHECKPOINT,"reader":"canonical two orientations of original thirteen CLIP windows; v1"})
    if manifest.is_file():
        if read(manifest.with_suffix(".json"))["signature"]!=signature:raise ValueError("Changed photo view manifest")
        return folder,torch.load(manifest,weights_only=True,mmap=True)
    selected,data=[],{}
    for s,(qi,labels) in qs.items():
        pi,g=gs[s]
        ids=stratified_sample(labels,36,63241) if ds=="sketchy" else torch.arange(len(labels))
        base=study_base(w,s,inputs,ids)
        pair=pairs(base,labels[ids],g,pi)
        selected.append(pair[pair>=0].flatten())
        data[s]={"ids":ids,"pairs":pair}
    priority=torch.cat(selected).unique().sort().values
    used=torch.zeros(len(pp),dtype=torch.bool)
    used[priority]=True
    sequence=torch.cat((priority,torch.where(~used)[0]))
    stored={"order":sequence,"priority_count":len(priority),"splits":data}
    save_tensor(stored,manifest)
    atomic_json(manifest.with_suffix(".json"),{"signature":signature,"priority_photos":len(priority),"gallery_union":len(pp),
               "key":"file ids from merge_inputs; labels affect extraction order only","pair_names":PAIR_NAMES})
    return folder,stored


def encode(w,ds,inputs,folder,plan,phase):
    cfg,_,_,pp,_,_,native,*_=inputs
    visual=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    sequence=plan["order"]
    cut=((plan["priority_count"]+SHARD-1)//SHARD)*SHARD
    begin=0 if phase in ("priority","all") else cut
    end=min(cut,len(pp)) if phase=="priority" else len(pp)
    for offset in range(begin,end,SHARD):
        selected=sequence[offset:offset+SHARD]
        file=folder/f"photos-{offset:06}.pt"
        if file.is_file():
            continue
        dl=DataLoader(Images([pp[i] for i in selected],normal_transform(),"rgb"),batch_size=96,
                      num_workers=cfg.workers,pin_memory=True,worker_init_fn=configure_image_worker)
        rows=[]
        for j,pixels in enumerate(dl):
            px=pixels.cuda(non_blocking=True)
            if j==0:
                feat=F.normalize(visual(px.to(visual.conv1.weight.dtype)).float())
                err=float((1-(feat*native[selected[:len(px)]].cuda()).sum(1)).abs().max())
                if err>.002:raise ValueError("Different CLIP photo features")
            _,both=two_pose_regions(visual,px)
            rows.append(both.cpu())
        points=torch.cat(rows)
        if points.shape!=(len(selected),13,2,512):raise ValueError("Incomplete photo views")
        save_tensor({"ids":selected,"views":points},file)
        if offset%(4*SHARD)==0 or offset+SHARD>=end:
            print("photo views",ds,phase,offset+len(selected),end,flush=True)
    (folder/("ready-"+phase)).touch()


def query_pack(w,split,inputs,ids):
    cfg,paths,qs,_,_,q,*_=inputs
    qp=qs[split][0][ids]
    return region_views(w,split,cfg,paths,qp,q)[0]["regions"]


def read_photos(folder,plan,ids):
    inverse=torch.empty_like(plan["order"])
    inverse[plan["order"]]=torch.arange(len(inverse))
    loc=inverse[ids]
    mapping=torch.empty((len(ids),13,2,512),dtype=torch.float16)
    for chunk in loc.div(SHARD,rounding_mode="floor").unique():
        offset=int(chunk)*SHARD
        use=(loc>=offset)&(loc<offset+SHARD)
        saved=torch.load(folder/f"photos-{offset:06}.pt",mmap=True,weights_only=True)
        positions=loc[use]-offset
        if not torch.equal(saved["ids"][positions],ids[use]):
            raise ValueError("Incorrect photo order")
        mapping[use]=saved["views"][positions]
    return mapping


@torch.inference_mode()
def main(w,ds,phase):
    inputs=semantic_inputs(w,ds)
    folder,plan=inputs_plan(w,ds,inputs)
    print("sources",ds,len(plan["order"]),plan["priority_count"],flush=True)
    encode(w,ds,inputs,folder,plan,phase)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    cli.add_argument("--dataset",choices=("sketchy","tuberlin"),required=True)
    cli.add_argument("--phase",choices=("priority","rest","all"),required=True)
    args=cli.parse_args()
    main(args.workspace,args.dataset,args.phase)
