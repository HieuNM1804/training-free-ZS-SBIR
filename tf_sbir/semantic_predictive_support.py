"""Marginalize over which target photographs express an under-specified sketch observation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import itertools
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .depiction_path_diagnostic import class_interval
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .gallery_region_panels import CONFIG, Index
from .geometry_word_flat import KD_FUNCTION_SHA, PrefixReplay, load_kd, make_word_bank, score_rows


FAMILY="semantic-predictive-support"


def integral(ratios):
    n=ratios.shape[-1]
    nodes,weights=np.polynomial.legendre.leggauss((n+2)//2)
    terms=[]
    for rho,weight in zip((nodes+1)/2,weights/2):
        terms.append(torch.log1p(float(rho)*(ratios-1)).sum(-1)+math.log(weight))
    return torch.logsumexp(torch.stack(terms),0)


def validate():
    r=torch.tensor([[.03,.6,1.4,12.],[1.,1.,1.,1.]],dtype=torch.double)
    for n in range(1,5):
        exact=torch.zeros(2,dtype=r.dtype)
        for k in range(n+1):
            for subset in itertools.combinations(range(n),k):
                exact+=r[:,list(subset)].prod(1)/((n+1)*math.comb(n,k))
        torch.testing.assert_close(integral(r[:,:n]).exp(),exact)


def neighborhoods(features,n):
    rows=[]
    for first in range(0,len(features),384):
        ids=torch.arange(first,min(first+384,len(features)),device=features.device)
        score=features[ids]@features.T
        score[torch.arange(len(ids),device=features.device),ids]=-torch.inf
        ranks=score.argsort(dim=1,descending=True,stable=True)[:,:n-1]
        rows.append(torch.cat((ids[:,None],ranks),1))
    return torch.cat(rows)


@torch.inference_mode()
def run(stage,dataset,size):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    validate()
    w=Path("/content/sbir")
    out=w/"results"/FAMILY
    _,_,qs,_,gs,qf,pf,src,tgt,scale=semantic_inputs(w,dataset)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    for split,(all_ids,all_labels) in qs.items():
        perm=torch.randperm(len(all_ids),generator=torch.Generator().manual_seed(42))
        if stage=="transfer":
            assert dataset=="tuberlin"
            use=torch.arange(len(all_ids))
        else:
            assert dataset=="sketchy"
            use=perm[:1536] if stage=="pilot" else perm[1536:]
        ids,labels=all_ids[use],all_labels[use]
        pi,gl=gs[split]
        photos=F.normalize(pf[pi].cuda().float(),dim=1)
        bank=make_word_bank(photos,tgt,scale,192)
        sizes=(4,8,16) if stage=="pilot" else (size,)
        near=neighborhoods(photos,max(sizes))
        graph=Index(SimpleNamespace(p=photos),rgb_only=True)
        graph.prepare(CONFIG)
        evaluator=PrefixReplay(kd,gl,split)
        raw={}
        with ThreadPoolExecutor(max_workers=2) as pool:
            for batch in torch.arange(len(ids)).split(24):
                q=F.normalize(qf[ids[batch]].cuda().float(),dim=1)
                direct=(scale*(q@src.T)).softmax(1)@bank.T
                ratio=direct/direct.mean(1,keepdim=True)
                assert ratio.min()>0
                scores=dict(word=direct,diffusion=graph.score(CONFIG,direct))
                gathered=ratio[:,near]
                for n in sizes:
                    group=gathered[:,:,:n]
                    scores[f"one_{n}"]=group.mean(-1)
                    scores[f"all_{n}"]=group.log().sum(-1)
                    scores[f"half_{n}"]=torch.log1p(.5*(group-1)).sum(-1)
                    scores[f"marginal_{n}"]=integral(group)
                    scores[f"anchored_{n}"]=ratio.log()+integral(group[:,:,1:])
                for name,value in scores.items():
                    raw.setdefault(name,[]).append(score_rows(value,labels[batch],gl.cuda(),split,evaluator,pool)[:,:2])
        raw={a:torch.cat(v).double() for a,v in raw.items()}
        contrasts={}
        for n in sizes:
            for arm in (f"marginal_{n}",f"anchored_{n}"):
                for ref in ("word","diffusion",f"one_{n}",f"all_{n}",f"half_{n}"):
                    diff=100*(raw[arm]-raw[ref])
                    contrasts[arm+" minus "+ref]=dict(points=diff.mean(0).tolist(),classes=class_interval(diff[:,0],labels,seed=42))
        stem=split+"-"+stage
        metrics={a:(100*v.mean(0)).tolist() for a,v in raw.items()}
        save_tensor(dict(qi=ids,pi=pi,labels=labels,rows=raw),out/(stem+".pt"))
        atomic_json(out/(stem+".json"),dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),
                    evaluator=KD_FUNCTION_SHA,queries=len(ids),gallery=len(pi),metrics=metrics,contrasts=contrasts))
        print(stem,metrics,flush=True)
        del graph,bank,raw


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    parser.add_argument("--stage",choices=("pilot","confirmation","transfer"),default="pilot")
    parser.add_argument("--size",type=int,choices=(4,8,16),default=8)
    args=parser.parse_args()
    run(args.stage,args.dataset,args.size)
