"""Test if fragment-only similarity is unusually high on incorrect photos when correct and incorrect candidates are adjacent in original rank."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,checksum,read
from .crossdomain_benchmark import revision
from .drawing_evidence_subsets import PARTITIONS,subset_bank
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .paired_fragment_area_control import read_area
from .paired_fragment_inputs import FAMILY
from .recoverable_patch_study import prepare
from .whole_fragment_likelihood import catalog,log_overlap,prob,target_tables


BANDS=((0,200),(200,1000),(0,1000),(1000,5000))


def paired_boundaries(scores,b,g,y):
    order=b.argsort(1,descending=True,stable=True)
    rel=g[order]==y[:,None]
    observed={}
    for name,raw in scores.items():
        z=(raw-raw.mean(1,keepdim=True))/raw.std(1,keepdim=True,unbiased=False).clamp_min(1e-7)
        z=z.gather(1,order)
        for start,stop in BANDS:
            pos=rel[:,start:stop]
            vals=z[:,start:stop]
            def orientation(mask,right_minus_left):
                n=mask.sum(1)
                prob=(mask*((right_minus_left>0).float()+.5*(right_minus_left==0))).sum(1)/n.clamp_min(1)
                margin=(mask*right_minus_left).sum(1)/n.clamp_min(1)
                return prob,margin,n
            above=pos[:,:-1]&~pos[:,1:]
            below=~pos[:,:-1]&pos[:,1:]
            a,wa,na=orientation(above,vals[:,1:]-vals[:,:-1])
            c,wc,nc=orientation(below,vals[:,:-1]-vals[:,1:])
            observed[name+f"/{start+1}-{stop}"]=torch.stack((.5*(a+c),.5*(wa+wc),na,nc,a,c),1).cpu()
    return observed


def report(v,labels):
    use=(v[:,2]>0)&(v[:,3]>0)
    if not use.any():return {"queries":0}
    x=v[use]
    y=labels[use].unique(return_inverse=True)[1]
    counts=y.bincount().double()
    total=torch.zeros_like(counts).index_add_(0,y,x[:,0].double())
    ix=torch.randint(len(counts),(4000,len(counts)),generator=torch.Generator().manual_seed(80931))
    est=total[ix].sum(1)/counts[ix].sum(1)
    return {"queries":len(x),"boundaries":int(x[:,2:4].sum()),"wrong_receives_higher_signal":float(x[:,0].mean()),
            "class_bootstrap_95":est.quantile(torch.tensor([.025,.975],dtype=torch.float64)).tolist(),
            "wrong_minus_relevant_in_gallery_SD":float(x[:,1].mean()),"initially_right_above":float(x[:,4].mean()),
            "initially_wrong_above":float(x[:,5].mean())}


@torch.inference_mode()
def run(w):
    inputs=semantic_inputs(w,"sketchy")
    torch.backends.cuda.matmul.allow_tf32=False
    subset=subset_bank(w,"sketchy","pilot-query")
    root=w/"results"/FAMILY
    choices=read(root/"pilot.json")
    probes={"leading_fragment_penalty":choices["best"]["parts"],"leading_four_way_interaction":choices["main"]}
    saved,summary={},{}
    for s in inputs[2]:
        p=prepare(w,s,inputs,subset)
        qids=inputs[2][s][0][p["ids"]]
        cat=catalog(w,s,inputs)
        branches={}
        for label,tag in probes.items():
            n,left,right,graph,_,_=tag.split("/");n=int(n)
            target=target_tables(w,s,"sketchy",inputs,cat,n)
            axes=inputs[7][cat["order"][:n]]
            scale=inputs[-1]
            x=inputs[5][qids].cuda().float()
            xp=subset["subsets"][p["positions"]][:,PARTITIONS.index(left),2**torch.arange(6)].cuda()
            xnull=read_area(w,"sketchy","pilot-query",qids).cuda() if left=="appearance" else subset["subsets"][p["positions"]][:,PARTITIONS.index("shuffled"),2**torch.arange(6)].cuda()
            null="area_shuffle" if right=="appearance" else "shuffled"
            fragment=prob(xp,axes,scale).mean(1);scrambled=prob(xnull,axes,scale).mean(1)
            whole=prob(x,axes,scale)
            splits=defaultdict(list)
            for off in range(0,len(qids),32):
                sl=slice(off,off+32);b=p["base"][sl].cuda()
                a00=log_overlap(fragment[sl],target[right,graph])
                a11=log_overlap(whole[sl],target["whole",graph])
                a01=log_overlap(fragment[sl],target["whole",graph])
                a10=log_overlap(whole[sl],target[right,graph])
                shuffled=log_overlap(scrambled[sl],target[null,graph])
                clusterperm=log_overlap(fragment[sl],target[right,graph][cat["permutation"].cuda()])
                # A descriptive residual from random placement of the same sized pieces.
                fs={"fragments":a00,"whole":a11,"whole_query_fragment_photo":a10,"fragment_query_whole_photo":a01,
                    "matched_positions_shuffled":shuffled,"cluster_shuffled":clusterperm,"fragment_minus_position_null":a00-shuffled,
                    "four_way":a11-a01-a10+a00,"raw_CLIP":x[sl]@p["rgb"].T}
                stats=paired_boundaries(fs,b,p["g"].cuda(),p["y"][sl].cuda())
                for k,ans in stats.items():splits[k].append(ans)
            branches[label]={k:torch.cat(v) for k,v in splits.items()}
            del target
        saved[s]={"labels":p["y"],"ids":p["ids"],"readings":branches}
        summary[s]={tag:{k:report(traces,p["y"]) for k,traces in rows.items()} for tag,rows in branches.items()}
        print(s,summary[s]["leading_fragment_penalty"]["fragments/1-200"],flush=True)
        del p
    path=root/"false-matches-at-adjacent-ranks.pt"
    save_tensor(saved,path)
    atomic_json(path.with_suffix(".json"),{"revision":revision(),"cutoffs":BANDS,"probes":probes,"splits":summary,
                "scope":"Within each original rank band, condition on a relevant and irrelevant photo being adjacent. Average both original directions equally within each query, then average eligible queries. Signal is measured without gallery labels, excluded queries have no boundaries in one direction.",
                "per_query_sha256":checksum(path)})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
