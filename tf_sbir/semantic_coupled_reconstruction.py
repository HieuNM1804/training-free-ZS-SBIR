"""Predict a photo distribution using shared local coordinates inferred from one sketch."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .depiction_path_diagnostic import class_interval
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .geometry_word_flat import KD_FUNCTION_SHA, PrefixReplay, load_kd, make_word_bank, score_rows


FAMILY = "semantic-coupled-reconstruction"


def coordinates(query, atoms):
    centered = atoms - atoms.mean(1, keepdim=True)
    rhs = (centered @ (query-atoms.mean(1)).unsqueeze(-1)).squeeze(-1)
    gram = centered @ centered.transpose(1,2)
    ridge = .1 * gram.diagonal(dim1=1,dim2=2).mean(1)
    gram.diagonal(dim1=1,dim2=2).add_(ridge[:,None].clamp_min(1e-9))
    weights = torch.linalg.solve(gram,rhs)
    weights += (1-weights.sum(1,keepdim=True))/weights.shape[1]
    torch.testing.assert_close(weights.sum(1),torch.ones_like(ridge),atol=1e-5,rtol=0)
    return weights


def gallery_decoder(features, probabilities, reference):
    mass = torch.zeros(probabilities.shape[1],device=features.device)
    numerator = torch.zeros((len(mass),features.shape[1]),device=features.device)
    for batch in reference.split(512):
        p=probabilities[batch]
        mass += p.sum(0)
        numerator.addmm_(p.T,features[batch])
    assert mass.min()>0
    mean=features[reference].mean(0)
    centered=features[reference]-mean
    cov=centered.T@centered/len(reference)
    cov.diagonal().add_(.02*cov.diagonal().mean())
    return numerator/mass[:,None]-mean,mean,torch.linalg.cholesky(cov)


def evidence(weights, mean_offsets, photos, mean, cholesky, isotropic=False):
    delta=(weights[:,None,:]@mean_offsets).squeeze(1)
    direction=delta if isotropic else torch.cholesky_solve(delta.T,cholesky).T
    return direction@(photos-mean).T-.5*(direction*delta).sum(1,keepdim=True)


def compute(query, photos, bank, source, target, scale, folds, models, sizes):
    logits=scale*(query@source.T)
    result={"visual":query@photos.T,"word":logits.softmax(1)@bank.T}
    top=logits.topk(max(sizes)).indices
    for k in sizes:
        ids=top[:,:k]
        weights=coordinates(query,source[ids])
        soft=logits.gather(1,ids).softmax(1)
        result[f"text_{k}"]=F.normalize((weights[:,None,:]@target[ids]).squeeze(1),dim=-1)@photos.T
        for label,wts,isotropic in (("affine",weights,False),("soft",soft,False),("isotropic",weights,True)):
            scores=torch.empty((len(query),len(photos)),device=query.device)
            for held,(offsets,mean,chol) in zip(folds,models):
                scores[:,held]=evidence(wts,offsets[ids],photos[held],mean,chol,isotropic)
            result[f"{label}_{k}"]=scores
    return result


@torch.inference_mode()
def run(dataset,stage,k):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    work=Path("/content/sbir")
    out=work/"results"/FAMILY
    _,_,queries,_,galleries,qf,pf,source,target,scale=semantic_inputs(work,dataset)
    kd,_=load_kd(work)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    for split,(all_ids,all_labels) in queries.items():
        permutation=torch.randperm(len(all_ids),generator=torch.Generator().manual_seed(42))
        if stage=="pilot":
            assert dataset=="sketchy"
            selected=permutation[:1536]
        elif stage=="confirmation":
            assert dataset=="sketchy"
            selected=permutation[1536:]
        else:
            assert dataset=="tuberlin"
            selected=torch.arange(len(all_ids))
        ids,labels=all_ids[selected],all_labels[selected]
        pi,gl=galleries[split]
        photos=F.normalize(pf[pi].cuda().float(),dim=1)
        bank=make_word_bank(photos,target,scale,192)
        folds=[f.cuda() for f in torch.randperm(len(pi),generator=torch.Generator().manual_seed(42)).tensor_split(2)]
        models=[gallery_decoder(photos,bank,ref) for ref in reversed(folds)]
        assert torch.equal(torch.cat(folds).cpu().sort().values,torch.arange(len(pi)))
        evaluator=PrefixReplay(kd,gl,split)
        sizes=(16,32,64) if stage=="pilot" else (k,)
        rows={}
        with ThreadPoolExecutor(max_workers=2) as pool:
            for batch in torch.arange(len(ids)).split(48):
                query=F.normalize(qf[ids[batch]].cuda().float(),dim=1)
                scores=compute(query,photos,bank,source,target,scale,folds,models,sizes)
                for name,val in scores.items():
                    rows.setdefault(name,[]).append(score_rows(val,labels[batch],gl.cuda(),split,evaluator,pool)[:,:2])
        raw={a:torch.cat(v).double() for a,v in rows.items()}
        stem=split+"-"+stage
        stats={}
        for size in sizes:
            for ref in ("word",f"soft_{size}",f"isotropic_{size}",f"text_{size}"):
                delta=100*(raw[f"affine_{size}"]-raw[ref])
                stats[f"affine_{size} minus {ref}"]=dict(points=delta.mean(0).tolist(),classes=class_interval(delta[:,0],labels,seed=42))
        save_tensor(dict(qi=ids,pi=pi,labels=labels,rows=raw),out/(stem+".pt"))
        metrics={a:(100*v.mean(0)).tolist() for a,v in raw.items()}
        atomic_json(out/(stem+".json"),dict(commit=revision(),gpu=torch.cuda.get_device_name(),seed=42,
            evaluator=KD_FUNCTION_SHA,queries=len(ids),gallery=len(pi),stage=stage,metrics=metrics,contrasts=stats))
        print(stem,metrics,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    parser.add_argument("--stage",choices=("pilot","confirmation","transfer"),default="pilot")
    parser.add_argument("--k",choices=(16,32,64),type=int,default=32)
    args=parser.parse_args()
    run(args.dataset,args.stage,args.k)
