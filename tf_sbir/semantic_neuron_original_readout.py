"""Replace the original all-vocabulary global branch of the regional ranker, with its frozen 0.65 aggregation weight."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import hashlib
import inspect

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .crossdomain_benchmark import digest, revision
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .gallery_region_panels import CONFIG, Index
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd, paired_delta
from .semantic_neuron_counterfactuals import restore
from .semantic_neuron_gallery import FAMILY
from .semantic_neuron_retrieval import setup
from .shared_concept_likelihood import emission


@torch.inference_mode()
def run(w,ds,phase):
    inputs=semantic_inputs(w,ds)
    cfg,_,qs,_,gs,_,p,src,tgt,scale=inputs
    torch.backends.cuda.matmul.allow_tf32=False
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Changed KD evaluator")
    root=w/"results"/FAMILY
    lock=read(root/"freeze.json")
    if digest(lock["configuration"])!=lock["sha256"]:raise ValueError("Changed freeze")
    options=lock["configuration"]["selected"]
    arms=sorted({options[k].rsplit("/",2)[0] for k in ("nominee","constant","zero","shuffled","early","matched_units","random_units")})
    main=options["nominee"].rsplit("/",2)[0]
    mode="pilot-query" if phase=="pilot" else "all-query"
    folder=root/"original-readout"
    for s in qs:
        file=folder/((phase+"-"+s)+".json")
        sig=digest({"freeze":lock["sha256"],"rows":arms,"gallery":s,"stage":phase,"global_weight":.65,
                    "word_count":len(tgt),"operation":"substitute the original all-word RGB-diffused score, keep all original regional contributions"})
        if file.is_file():
            meta=read(file)
            if meta["signature"]!=sig or checksum(file.with_suffix(".pt"))!=meta["per_query_sha256"]:raise ValueError("Changed report")
            continue
        d,info=restore(w,s,mode)
        ix=setup(w,s,inputs,d,mode)
        positions=[info["arms"].index(k) for k in arms]
        y,g=ix["y"],ix["g"]
        raw=p[gs[s][0]].cuda()
        graph=Index(SimpleNamespace(p=raw),rgb_only=True)
        probabilities=torch.empty((len(raw),len(tgt)),device=raw.device)
        for j in range(0,len(raw),192):
            probabilities[j:j+192]=(scale*(raw[j:j+192]@tgt.T)).softmax(1)
        truth=g.cuda()
        metrics,tops=defaultdict(list),defaultdict(list)
        historic=torch.load(w/"cache/pragmatic-gallery-competition"/(s+"-named-rgb16.pt"),mmap=True,weights_only=True)
        err=0.;replay=Replay(kd,g,s)
        # A row-wise constant is immaterial to any rank. Check that the native global score reconstructs the stored branch.
        native_errors=[]
        with ThreadPoolExecutor(max_workers=2) as pool:
            for i in range(0,len(y),24):
                sl=slice(i,i+24)
                yy=y[sl];base=ix["base"][sl].cuda()
                z=d["native"][sl].cuda()
                zz=d["change"][sl][:,positions].cuda().float().transpose(0,1)+z[None]
                states=torch.cat([z[None],zz]).flatten(0,1)
                names=emission(states,src,scale)
                matrix=graph.score(CONFIG,names@probabilities.T).reshape(len(arms)+1,len(z),len(raw)).clamp_min_(1e-30).log_()
                old=historic[ix["ids"][sl]].cuda().clamp_min(1e-30).log()
                dist=matrix[0]-old
                native_errors.append((dist-dist.mean(1,keepdim=True)).abs().amax(1).cpu())
                trials={"regional":base,"original_global":matrix[0]}
                for k,new in zip(arms,matrix[1:],strict=True):
                    trials["regional/"+k]=base+.65*(new-matrix[0])
                    if k==main:trials["global/"+k]=new
                for k,val in trials.items():
                    cal,order,hits,_=map_values(val,yy.cuda(),truth,s)
                    exact=replay.score(order,yy,pool)
                    err=max(err,float((exact-cal.cpu()).abs().max()))
                    if err>2e-6:raise ValueError("Ranks differ from original KD-SBIR")
                    metrics[k].append(exact);tops[k].append(hits[:,0].cpu())
                if i%288==0:print("native all-word branch",s,i,len(y),flush=True)
        met={k:torch.cat(v) for k,v in metrics.items()}
        top={k:torch.cat(v) for k,v in tops.items()}
        check=torch.cat(native_errors)
        # Bound the actual perturbation in native scores and retain it per query to audit any exceptions.
        if check.median()>.02:raise ValueError("Native global branch differs from the archived implementation")
        target="regional/"+main
        labels=y.unique(return_inverse=True)[1]
        save_tensor({"ids":ix["ids"],"labels":labels,"metrics":met,"top1":top,"native_branch_centered_max_error":check},file.with_suffix(".pt"))
        atomic_json(file,{"signature":sig,"revision":revision(),"freeze":lock["sha256"],"queries":len(y),"gallery":len(g),
                          "main":target,"word_count":len(tgt),"global_weight":.65,"values":{k:v.double().mean(0).tolist() for k,v in met.items()},
                          "paired_main_minus_each":{k:paired_delta(met[target],v,labels) for k,v in met.items()},
                          "top1":{"fixed":int((top[target]&~top["regional"]).sum()),"broken":int((~top[target]&top["regional"]).sum())},
                          "KD_max_error":err,"centered_native_replay_bound_median_max":[float(check.median()),float(check.max())],
                          "per_query_sha256":checksum(file.with_suffix(".pt"))})
        print("all words",s,met[target].mean(0).tolist(),met["regional"].mean(0).tolist(),flush=True)
        del probabilities,graph,ix,d


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    cli.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    cli.add_argument("--stage",choices=("pilot","full"),default="pilot")
    args=cli.parse_args()
    run(args.workspace,args.dataset,args.stage)
