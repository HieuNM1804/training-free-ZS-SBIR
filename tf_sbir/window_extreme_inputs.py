"""Read the same regional questions on independent photo controls and single sketches, with disjoint name and null panels."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json,read
from .conditional_word_inputs import tree
from .cross_view_region_cache import two_pose_regions
from .crossdomain_benchmark import digest,revision
from .data import Images,configure_image_worker,normal_transform
from .encoding import load_frozen_clip,save_tensor
from .gallery_panel_resume import semantic_inputs,region_views
from .query_fragment_reference_inputs import dataset
from .query_partition_tree import folds


FAMILY="gallery-calibrated-window-extremes"
SEED=196177
DEPTH=2
SHARD=384


def check_scope(w,ds):
    if ds=="tuberlin" and not (w/"results"/FAMILY/"freeze.json").is_file():raise ValueError("Freeze before loading TU")


@torch.inference_mode()
def photo_views(w,ds):
    check_scope(w,ds)
    cfg,_,_,paths,_,_,p=dataset(w,ds)
    root=w/"cache"/FAMILY/ds/"photo-views";root.mkdir(parents=True,exist_ok=True)
    stamp=digest({"model":cfg.clip_checkpoint,"paths":paths,"views":"full two poses and 13 paired windows, B/32"})
    if (root/"done.json").is_file():
        if read(root/"done.json")["signature"]!=stamp:raise ValueError("Query control bank changed")
        return
    missing=[i for i in range(0,len(paths),SHARD) if not (root/f"{i:06}.pt").exists()]
    encoder=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    if missing:
        ix=torch.cat([torch.arange(i,min(len(paths),i+SHARD)) for i in missing])
        loader=DataLoader(Images([paths[j] for j in ix.tolist()],normal_transform(),"rgb"),batch_size=128,num_workers=cfg.workers,
                          pin_memory=True,persistent_workers=cfg.workers>0,worker_init_fn=configure_image_worker)
        offset=0;idx=0;n=0;acc=[]
        for batch in loader:
            qids=ix[offset:offset+len(batch)]
            batch=batch.cuda(non_blocking=True)
            flip,crops=two_pose_regions(encoder,batch);full=torch.stack((p[qids].cuda(),flip),1)[:,None]
            if offset==0:
                orig=torch.nn.functional.normalize(encoder(batch.to(encoder.conv1.weight.dtype)).float())
                if (1-(orig*p[qids].cuda()).sum(1)).abs().amax()>.002:raise ValueError("Different full-image encoding")
            acc.append(torch.cat((full,crops),1).half().cpu())
            n+=len(batch);offset+=len(batch);start=missing[idx];needed=min(SHARD,len(paths)-start)
            if n==needed:
                save_tensor(torch.cat(acc),root/f"{start:06}.pt");idx+=1;n=0;acc=[]
                if idx%8==1 or idx==len(missing):print("gallery windows",ds,offset,len(ix),flush=True)
            elif n>needed:raise ValueError("Window block crossed a disk boundary")
        if n or idx!=len(missing):raise ValueError("Incomplete photo bank")
    atomic_json(root/"done.json",{"signature":stamp,"images":len(paths),"code":revision()})


def make_banks(w,s,g,target,scale,ht,parts):
    # Construct the full-public-word probabilities anew if the independently saved identical gallery input is unavailable.
    prior=w/"cache/shared-private-name-evidence"/s/"photo-words.pt"
    bank=None
    if prior.is_file():
        check=torch.load(prior,mmap=True,weights_only=True)
        if check.shape==(len(g),len(target)):
            example=(scale*(g[:16]@target.T)).softmax(1).cpu()
            if (example-check[:16]).abs().amax()<.001:bank=check
    if bank is None:
        bank=torch.empty((len(g),len(target)))
        for i in range(0,len(g),256):bank[i:i+256]=(scale*(g[i:i+256]@target.T)).softmax(1).cpu()
    groups=ht.ids[DEPTH];k=ht.sizes[DEPTH];out=[]
    for donor in parts:
        word=g.new_zeros(k,len(target))
        for ix in donor.split(384):word.index_add_(0,groups[ix],bank[ix.cpu()].cuda())
        background=word.sum(0)/len(donor)
        counts=torch.bincount(groups[donor],minlength=k)
        out.append((word+background)/(counts[:,None]+1))
    return torch.stack(out)


def observations(z,words,cells,scale):
    n,h,p,_=z.shape
    flat=z.reshape(-1,512);scores=[]
    den=cells.flatten(0,1).T.contiguous()
    for i in range(0,len(flat),448):
        u=(scale*(flat[i:i+448].float()@words.T)).softmax(1)
        scores.append((u@den).clamp_min(1e-30).log())
    return torch.cat(scores).reshape(n,h,p,len(cells),cells.size(1)).cpu()


@torch.inference_mode()
def measure(w,ds):
    check_scope(w,ds)
    cfg,qp,qs,pp,gs,q,p,source,target,scale=semantic_inputs(w,ds)
    directory=w/"cache"/FAMILY/ds/"photo-views";stamp=read(directory/"done.json")
    tensors=[torch.load(f,mmap=True,weights_only=True) for f in sorted(directory.glob("[0-9]*.pt"))]
    photos=torch.cat(tensors);del tensors
    if len(photos)!=stamp["images"]:raise ValueError("Incomplete photographic controls")
    for s,(pi,labels) in gs.items():
        root=w/"cache"/FAMILY/s;root.mkdir(parents=True,exist_ok=True)
        if (root/"ready.json").is_file():continue
        g=p[pi].cuda();torch.set_float32_matmul_precision("highest")
        ht=tree(w,s,g);parts=folds(len(g),g.device,SEED)
        torch.set_float32_matmul_precision("high");banks=make_banks(w,s,g,target,scale,ht,parts)
        gfile=root/"structure.pt"
        save_tensor({"parts":[z.cpu() for z in parts],"groups":ht.ids[DEPTH].cpu(),"upper":ht.parents[DEPTH].cpu(),
                     "levels":ht.sizes[DEPTH:DEPTH+2]},gfile)
        for domain,words in (("photo",target),("source",source)):
            dest=root/("photos-"+domain+".pt")
            if dest.is_file():continue
            acc=[]
            for ix in pi.split(128):acc.append(observations(photos[ix].cuda(),words,banks,scale))
            save_tensor(torch.cat(acc),dest);print(s,domain,len(pi),flush=True)
        dest=root/"sketches.pt"
        if not dest.is_file():
            qi,_=qs[s];packed,_=region_views(w,s,cfg,qp,qi,q);acc=[]
            for i in range(0,len(qi),128):
                ids=qi[i:i+128];full=torch.stack((q[ids],packed["flipped"][i:i+len(ids)].float()),1)[:,None]
                regions=packed["regions"][i:i+len(ids)]
                z=torch.cat((full,regions.float()),1).cuda()
                acc.append(observations(z,source,banks,scale))
            save_tensor(torch.cat(acc),dest)
        atomic_json(root/"ready.json",{"code":revision(),"words":banks.size(2),"groups":banks.size(1),"candidate_photos":len(pi),
                    "queries":len(qs[s][1]),"reference_folds":3,"seed":SEED})
        print("ready",s,flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__);a.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    a.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy");a.add_argument("--stage",choices=("encode","measure","all"),default="all")
    args=a.parse_args()
    if args.stage in ("encode","all"):photo_views(args.workspace,args.dataset)
    if args.stage in ("measure","all"):measure(args.workspace,args.dataset)
