"""Evaluate incomplete semantic support versus conditional means on the identical diffusion geometry."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
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


FAMILY="semantic-predictive-diffusion"


def project(graph, signals):
    # The same positive linear operator is applied to all observations, with explicit row normalization.
    n=signals.shape[-1]
    flat=signals.reshape(-1,n)
    return graph.score(CONFIG,flat).reshape_as(signals)


def marginal(graph, ratio, sizes, order):
    nodes,weights=np.polynomial.legendre.leggauss(order)
    observables=torch.stack([torch.log1p(float(rho)*(ratio-1)) for rho in (nodes+1)/2])
    local=project(graph,observables)
    weights=torch.tensor(np.log(weights/2),dtype=local.dtype,device=local.device)[:,None,None]
    return {f"marginal_{n}":torch.logsumexp(n*local+weights,dim=0) for n in sizes}


@torch.inference_mode()
def run(dataset,stage,size):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    work=Path("/content/sbir")
    out=work/"results"/FAMILY
    _,_,queries,_,galleries,qf,pf,src,tgt,scale=semantic_inputs(work,dataset)
    kd,_=load_kd(work)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    for split,(all_ids,all_labels) in queries.items():
        perm=torch.randperm(len(all_ids),generator=torch.Generator().manual_seed(42))
        if stage=="transfer":
            assert dataset=="tuberlin"
            use=torch.arange(len(all_ids))
        else:
            assert dataset=="sketchy"
            use=perm[:1536] if stage=="pilot" else perm[1536:]
        ids,labels=all_ids[use],all_labels[use]
        pi,gl=galleries[split]
        photos=F.normalize(pf[pi].cuda().float(),dim=1)
        bank=make_word_bank(photos,tgt,scale,192)
        graph=Index(SimpleNamespace(p=photos),rgb_only=True)
        graph.prepare(CONFIG)
        sizes=(4,8,16) if stage=="pilot" else (size,)
        evaluator=PrefixReplay(kd,gl,split)
        raw={}
        replay={}
        with ThreadPoolExecutor(max_workers=2) as pool:
            for batch in torch.arange(len(ids)).split(24):
                q=F.normalize(qf[ids[batch]].cuda().float(),dim=1)
                word=(scale*(q@src.T)).softmax(1)@bank.T
                ratio=word/word.mean(1,keepdim=True)
                vals=project(graph,torch.stack((ratio,torch.log1p(.5*(ratio-1)),ratio.log())))
                scores={"word":word,"diffusion":vals[0],"half":vals[1],"all":vals[2]}
                scores.update(marginal(graph,ratio,sizes,16))
                rows={k:score_rows(v,labels[batch],gl.cuda(),split,evaluator,pool)[:,:2] for k,v in scores.items()}
                if not raw:
                    check=marginal(graph,ratio,sizes,32)
                    for k,v in check.items():
                        drift=v-scores[k]
                        delta=100*(score_rows(v,labels[batch],gl.cuda(),split,evaluator,pool)[:,:2]-rows[k])
                        replay[k]=dict(centered_max=float((drift-drift.mean(1,keepdim=True)).abs().max()),
                                       drift_points=delta.mean(0).tolist(),max_points=delta.abs().amax(0).tolist())
                for k,v in rows.items():
                    raw.setdefault(k,[]).append(v)
        raw={k:torch.cat(v).double() for k,v in raw.items()}
        contrasts={}
        for n in sizes:
            for name in ("word","diffusion","half","all"):
                diff=100*(raw[f"marginal_{n}"]-raw[name])
                contrasts[f"marginal_{n} minus {name}"]=dict(points=diff.mean(0).tolist(),classes=class_interval(diff[:,0],labels,seed=42))
        stem=split+"-"+stage
        metrics={k:(100*v.mean(0)).tolist() for k,v in raw.items()}
        save_tensor(dict(qi=ids,pi=pi,labels=labels,rows=raw),out/(stem+".pt"))
        atomic_json(out/(stem+".json"),dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),
                    evaluator=KD_FUNCTION_SHA,queries=len(ids),gallery=len(pi),metrics=metrics,contrasts=contrasts,quadrature=replay))
        print(stem,metrics,flush=True)
        del graph,bank,raw


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    parser.add_argument("--stage",choices=("pilot","confirmation","transfer"),default="pilot")
    parser.add_argument("--size",type=int,choices=(4,8,16),default=16)
    args=parser.parse_args()
    run(args.dataset,args.stage,args.size)
