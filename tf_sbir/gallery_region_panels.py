"""Estimate regional evidence under independently resampled unlabeled gallery vocabularies."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .cluster_mine import cluster_assignments
from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import bare_words
from .cross_diffusion_map import Config, Index
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .encoding import save_tensor
from .foundation import load_words
from .retrieval_diagnosis import global_inputs
from .semantic import read_words
from .shared_concept_likelihood import fit_prior
from .shared_gallery_diagnosis import FAMILY


SIZES = (32,64,128)
FOLDS = 6
GROUPS = ("iid","cluster")
CONFIG = Config(neighbors=16,bandwidth=.36,iterations=2,mode="rgb")


def donors(photo, seed=72937, kinds=GROUPS):
    n=len(photo)
    random=torch.randperm(n,generator=torch.Generator().manual_seed(seed))
    image_parts=list(random.tensor_split(FOLDS))
    if tuple(kinds)==("iid",):
        return {"iid":image_parts}
    groups=cluster_assignments(F.normalize(photo-photo.mean(0)),48).cpu()
    counts=groups.bincount(minlength=48)
    batches=[[] for _ in range(FOLDS)]
    loads=[0]*FOLDS
    for j in counts.argsort(descending=True).tolist():
        recipient=min(range(FOLDS),key=loads.__getitem__)
        ids=(groups==j).nonzero().flatten()
        batches[recipient].append(ids)
        loads[recipient]+=len(ids)
    clusters=[torch.cat(v) for v in batches]
    for collection in (image_parts,clusters):
        if not torch.equal(torch.cat(collection).sort().values,torch.arange(n)):
            raise ValueError("Each donor split must use every gallery photo once")
    return {"iid":image_parts,"cluster":clusters}


def compile_all(w,ds,seed=72937,capacities=SIZES,kinds=GROUPS,keep_splits=None):
    cfg,_,_,paths,galleries,_,raw=global_inputs(w,ds)
    vocab=read_words(cfg.vocabulary)
    if len(vocab)!=20931:
        raise ValueError("Changed public name vocabulary")
    tgt=load_words(cfg,vocab)[1].cuda()
    _,scale=bare_words(cfg,vocab)
    for split,(pi,_) in galleries.items():
        if keep_splits is not None and split not in keep_splits:
            continue
        folder=w/"cache"/FAMILY
        settings={"checkpoint":CHECKPOINT,"vocab":checksum(cfg.vocabulary),"gallery":digest([str(paths[i]) for i in pi]),
                  "capacities":capacities,"folds":FOLDS,"types":kinds,"recipe":"v1 separate donor prior/prototypes, same frozen full-gallery RGB"}
        default=seed==72937 and tuple(capacities)==SIZES and tuple(kinds)==GROUPS
        if not default:
            settings["seed"]=seed
        signature=digest(settings)
        path=folder/(split+("-donor-indices.pt" if default else f"-donor-{signature[:16]}.pt"))
        if path.exists() and path.with_suffix(".json").exists() and read(path.with_suffix(".json"))["signature"]==signature:
            continue
        p=raw[pi].cuda()
        groups=donors(p,seed,kinds)
        graph=Index(SimpleNamespace(p=p),rgb_only=True)
        probs=torch.empty((len(p),len(vocab)),device=p.device)
        for start in range(0,len(p),128):
            probs[start:start+128]=(scale*(p[start:start+128]@tgt.T)).softmax(1)
        stores={}
        meta={}
        for kind,partitions in groups.items():
            for j,ids in enumerate(partitions):
                subset=ids.cuda()
                prior,info=fit_prior(probs[subset])
                prior=prior["mle"]
                centered=F.normalize(p-p[subset].mean(0))
                for capacity in capacities:
                    terms=prior.topk(capacity).indices
                    logw=prior[terms].clamp_min(1e-30).log()
                    prompt=scale*(p@tgt[terms].T)+logw
                    centroid=F.normalize(prompt[subset].softmax(1).T@centered[subset])
                    estimate=.5*prompt+.5*(scale*(centered@centroid.T)+logw)
                    bank=graph.score(CONFIG,estimate.softmax(1).T.contiguous()).T.contiguous()
                    bank=bank/bank.sum(1,keepdim=True).clamp_min(1e-30)
                    if not torch.isfinite(bank).all() or float((bank.sum(1)-1).abs().max())>.001:
                        raise ValueError("Invalid conditional word probabilities")
                    # Keep positive low-mass entries at float32 precision.
                    stores[kind,capacity,j]=(terms.cpu(),bank.cpu())
                meta[kind+str(j)]={"photos":len(ids),"iterations":info["iterations"]}
            print("references",split,kind,flush=True)
        overlap={}
        for k in kinds:
            for c in capacities:
                ss=[set(stores[k,c,j][0].tolist()) for j in range(FOLDS)]
                overlap[k+"/"+str(c)]=sum(len(a&b)/len(a|b) for i,a in enumerate(ss) for b in ss[i+1:])/(FOLDS*(FOLDS-1)/2)
        save_tensor({"banks":stores,"partitions":groups},path)
        record={"signature":signature,"gallery":len(p),"overlap_Jaccard":overlap,"donors":meta}
        if not default:
            record["seed_settings"]=settings
        atomic_json(path.with_suffix(".json"),record)
        print("stored",split,path.stat().st_size,overlap,flush=True)


def read_regions(crop, src, refs, scale, temps):
    # Keep the score calibration of each pose and field of view. Variability due to
    # a single common multiplicative constant in a donor panel carries no ranking signal.
    n,poses,regions,d=crop.shape
    prior=crop.new_tensor([1/18]*9+[1/8]*4)[None,None,:,None]
    swap=torch.tensor([1,2,0,4,5,3,7,8,6,10,9,12,11],device=crop.device)
    acc={}
    sum_values=None
    max_mass_error=0.
    for index,(names,bank) in enumerate(refs):
        z=scale*(crop@src[names].T)
        weights=(z-z.amax((1,2,3),keepdim=True)).exp()
        likelihood=(weights.flatten(0,2)@bank.T).reshape(n,poses,regions,-1).clamp_min(1e-30)
        means=likelihood.mean(-1,keepdim=True).clamp_min(1e-30)
        scale_ref=(prior*means).sum(2,keepdim=True).mean(1,keepdim=True)
        value=likelihood/scale_ref.clamp_min(1e-30)
        ratio=(likelihood.log()-means.log())
        sum_values=value.clone() if sum_values is None else sum_values+value
        for t in temps:
            select=(prior.log()+t*ratio).softmax(2)
            same=(select*value).sum(2)
            wrong=(select*value.index_select(2,swap)).sum(2)
            if index==0:
                acc[t]=[select,same,wrong]
            else:
                acc[t][0]+=select;acc[t][1]+=same;acc[t][2]+=wrong
        if index==0:
            max_mass_error=float(((prior*value.mean(-1,keepdim=True)).sum(2).mean(1)-1).abs().max())
    h=len(refs)
    pooled=(prior*(sum_values/h)).sum(2).mean(1).clamp_min(1e-30).log()
    result={}
    for t,(a,same,wrong) in acc.items():
        product=(a*sum_values).sum(2)
        bad=(a*sum_values.index_select(2,swap)).sum(2)
        cross=((product-same)/(h*(h-1))).mean(1)
        shuffled=((bad-wrong)/(h*(h-1))).mean(1)
        self_=(same/h).mean(1)
        independent=(product/h**2).mean(1)
        # Check the finite-sample mixture identity directly before taking logs.
        mix=(self_+(h-1)*cross)/h
        error=float(((independent-mix).abs()/independent.clamp_min(1e-12)).amax())
        if error>1e-4 or max_mass_error>.001:
            raise ValueError(f"Panel formula or normalization changed: {error}, {max_mass_error}")
        result[t]={k:v.clamp_min(1e-30).log() for k,v in (("cross",cross),("same",self_),("average",independent),("wrong",shuffled))}
    return result,pooled


@torch.inference_mode()
def main(w,ds):
    compile_all(w,ds)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    cli.add_argument("--dataset",choices=("sketchy","tuberlin"),required=True)
    args=cli.parse_args();main(args.workspace,args.dataset)
