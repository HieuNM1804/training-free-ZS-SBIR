"""Resume the already frozen estimator on a fresh Colab without repeating Sketchy selection."""

import argparse
import hashlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import bare_words
from .cross_view_region_cache import two_pose_regions
from .cross_view_regions import CandidateRegions
from .crossdomain_benchmark import SPLIT_HASHES, digest
from .crossdomain_inputs import CHECKPOINT, runtime
from .data import Images, UNSEEN_CLASSES, configure_image_worker, normal_transform
from .encoding import load_frozen_clip, load_global_inputs, save_tensor
from .foundation import load_words
from .gallery_region_panel_full import read_valid, split_result
from .gallery_region_panels import CONFIG, FAMILY, Index, compile_all
from .quickdraw_locked import KD_FUNCTION_SHA
from .regional_intervention_study import index
from .retrieval_diagnosis import SKETCHY_SPLITS, merge_inputs
from .retrieval_evidence import load_kd
from .semantic import read_words
from .shared_concept_likelihood import emission


def load_archive(w):
    source=Path(__file__).resolve().parents[1]/"docs/results"/FAMILY
    root=w/"results"/FAMILY
    root.mkdir(parents=True,exist_ok=True)
    for name in ("protocol","pilot","shortlist","full-protocol","freeze","sketchy_1"):
        file=source/(name+".json")
        target=root/file.name
        if not target.exists():
            target.write_bytes(file.read_bytes())
        if checksum(file)!=checksum(target):
            raise ValueError("Conflicting recovered record "+name)
    first=root/"sketchy_1.pt"
    if not first.exists():
        with zipfile.ZipFile(source/"sketchy_1-official.zip") as archive:
            if archive.namelist()!=["sketchy_1.pt"]:
                raise ValueError("Missing original per-query archive")
            first.write_bytes(archive.read("sketchy_1.pt"))
    lock=read(root/"freeze.json")
    if digest({k:v for k,v in lock.items() if k!="sha256"})!=lock["sha256"]:
        raise ValueError("Stored full Sketchy lock failed authentication")
    if not read_valid(root/"sketchy_1.json",lock["sha256"]):
        raise ValueError("Lost original full Ext-1 result")
    return root,lock


def semantic_inputs(w,ds):
    splits=SKETCHY_SPLITS if ds=="sketchy" else ("tuberlin",)
    workers=min(4,max(1,len(os.sched_getaffinity(0))))
    cfg=runtime(w,ds,batch=256,workers=workers)
    qpaths,qs=merge_inputs(cfg.root,splits,"sketch")
    ppaths,gs=merge_inputs(cfg.root,splits,"photo")
    q,p,_=load_global_inputs(cfg,qpaths,ppaths,False)
    words=read_words(cfg.vocabulary)
    if len(words)!=20931:
        raise ValueError("Changed public vocabulary")
    src,tgt=(v.cuda() for v in load_words(cfg,words))
    _,scale=bare_words(cfg,words)
    return cfg,qpaths,qs,ppaths,gs,q,p,src,tgt,scale


def region_views(w,s,cfg,paths,qi,native):
    folder=w/"cache"/FAMILY/"source-windows"
    folder.mkdir(parents=True,exist_ok=True)
    selected=[paths[int(i)] for i in qi]
    info={"images":digest(list(map(str,selected))),"checkpoint":CHECKPOINT,"recipe":"same two poses and 13 windows"}
    f=folder/(s+"-"+digest(info)[:16]+".pt")
    if not f.exists():
        visual=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
        data=torch.empty((len(qi),13,2,512),dtype=torch.float16)
        mirror=torch.empty((len(qi),512),dtype=torch.float16)
        count=0
        for start in range(0,len(qi),768):
            stop=min(start+768,len(qi))
            shard=f.with_name(f.stem+f"-{start}.shard")
            if shard.exists():
                pair=torch.load(shard,weights_only=True,mmap=True)
            else:
                loader=DataLoader(Images(selected[start:stop],normal_transform(),"rgb"),batch_size=96,
                                  num_workers=cfg.workers,pin_memory=True,worker_init_fn=configure_image_worker)
                crops,flips=[],[]
                for pixels in loader:
                    pixels=pixels.cuda(non_blocking=True)
                    if not crops:
                        direct=F.normalize(visual(pixels.to(visual.conv1.weight.dtype)).float())
                        if (1-(direct*native[qi[start:start+len(pixels)]].cuda()).sum(1)).abs().max()>.002:
                            raise ValueError("Different source normalization")
                    flip,regions=two_pose_regions(visual,pixels)
                    crops.append(regions.cpu())
                    flips.append(flip.cpu())
                pair={"regions":torch.cat(crops),"flipped":torch.cat(flips)}
                save_tensor(pair,shard)
            data[start:stop]=pair["regions"]
            mirror[start:stop]=pair["flipped"]
            count+=stop-start
            if start%3072==0 or stop==len(qi):
                print("windows",s,stop,len(qi),flush=True)
        if count!=len(qi):
            raise ValueError("Missing two-pose sketch")
        save_tensor({"regions":data,"flipped":mirror},f)
        atomic_json(f.with_suffix(".json"),info)
    packed=torch.load(f,mmap=True,weights_only=True)
    reverse=torch.full((len(paths),),-1,dtype=torch.int64)
    reverse[qi]=torch.arange(len(qi))
    def read_views(ids):
        local=reverse[ids]
        if (local<0).any():
            raise ValueError("Query outside locked split")
        return packed["regions"][local].cuda().transpose(1,2)
    return packed,read_views


def original_ranker(w,s,qi,pi,p,pp,q,src,tgt,scale,packed):
    dest=w/"cache"/FAMILY/(s+"-regional-reference.pt")
    hist=w/"cache/pragmatic-gallery-competition"/(s+"-named-rgb16.pt")
    if dest.exists() and hist.exists():
        return
    photo=p[pi].cuda()
    if not hist.exists():
        graph=Index(SimpleNamespace(p=photo),rgb_only=True)
        prob=torch.empty((len(photo),len(tgt)),device=photo.device)
        for start in range(0,len(photo),192):
            sub=photo[start:start+192]
            prob[start:start+len(sub)]=(scale*(sub@tgt.T)).softmax(1)
        scores=torch.empty((len(qi),len(pi)))
        for start in range(0,len(qi),192):
            sub=q[qi[start:start+192]].cuda()
            scores[start:start+len(sub)]=graph.score(CONFIG,emission(sub,src,scale)@prob.T).cpu()
        save_tensor(scores,hist)
        del graph,prob,scores
    historical=torch.load(hist,mmap=True,weights_only=True)
    stamp=digest({"gallery":[str(pp[j]) for j in pi],"model":CHECKPOINT,"source":"regional-64/RGB16"})
    words,bank=index(w,s,photo,tgt,scale,stamp[:16])
    baseline=torch.empty_like(historical,device="cpu")
    for start in range(0,len(qi),48):
        sl=slice(start,start+48)
        crops=packed["regions"][sl].cuda()
        reader=CandidateRegions(q[qi[sl]].cuda(),packed["flipped"][sl].cuda(),crops[:,:,0],crops[:,:,1],
                                src,words,bank,src.new_empty((0,len(src))),historical[sl].cuda())
        reader.prepare(scale)
        baseline[sl]=(.65*reader.first+.35*reader.read("self",1.,"gallery")).cpu()
    save_tensor(baseline,dest)
    print("rebuilt original regional score",s,tuple(baseline.shape),flush=True)


@torch.inference_mode()
def run(w,s,workers,batch):
    root,lock=load_archive(w)
    if read_valid(root/(s+".json"),lock["sha256"]) is not None:
        return
    ds="tuberlin" if s=="tuberlin" else "sketchy"
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Original KD evaluator changed")
    print("resume",s,torch.cuda.get_device_name(),flush=True)
    cfg,qp,qs,pp,gs,q,p,src,tgt,scale=semantic_inputs(w,ds)
    qi,_=qs[s]
    pi,_=gs[s]
    if digest(UNSEEN_CLASSES[s])!=SPLIT_HASHES[s]:
        raise ValueError("Changed test categories")
    packed,view=region_views(w,s,cfg,qp,qi,q)
    original_ranker(w,s,qi,pi,p,pp,q,src,tgt,scale,packed)
    compile_all(w,ds,keep_splits=(s,))
    torch.backends.cuda.matmul.allow_tf32=True
    split_result(w,s,(qs,gs,src,scale,view),lock["official_rows"],root/(s+".json"),lock["sha256"],
                 kd,workers,batch,lock["nominee"])


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    cli.add_argument("--split",choices=("tuberlin","sketchy_2"),required=True)
    cli.add_argument("--workers",type=int,default=2)
    cli.add_argument("--batch",type=int,default=96)
    args=cli.parse_args()
    run(args.workspace,args.split,args.workers,args.batch)
