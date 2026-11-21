"""Fixed post-selection measurement of ties created by two three-word assignments."""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import zipfile

import torch
from .concept_benchmark import atomic_json,checksum,read
from .conditional_semantic_index import bare_words,SIZES
from .crossdomain_benchmark import revision
from .crossdomain_inputs import CHECKPOINT
from .foundation import load_words
from .map_study import Data,config_id,evaluate
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import load_kd,paired_delta,tensor
from .retrieval_soft_assignment import assign
from .semantic import read_words


@dataclass(frozen=True)
class Case:
    order: str
    words_per_side: int = 3
    temperature_ratio: float = 1.


def secondary_order(sparse,full,mode):
    if mode=="words_only":
        return sparse
    if mode=="dense":
        return full
    random_order=torch.randperm(sparse.size(1),generator=torch.Generator().manual_seed(7632)).to(full.device)
    levels=torch.linspace(.9,-.9,sparse.size(1),device=full.device)
    result=torch.empty_like(sparse)
    for start in range(0,len(full),96):
        x=sparse[start:start+96]
        if mode=="random":
            first=random_order.expand(len(x),-1)
        else:
            secondary=full[start:start+len(x)]
            if mode=="dense_zeros_only":
                secondary=secondary.masked_fill(x!=0.,0.)
            first=secondary.argsort(dim=1,descending=True,stable=True)
        order=first.gather(1,x.gather(1,first).argsort(dim=1,descending=True,stable=True))
        result[start:start+len(x)].scatter_(1,order,levels.expand_as(x))
    return result


@torch.inference_mode()
def run(w):
    dest=w/"results/assignment-ties-full"
    kd,kd_sha=load_kd(w)
    cases=("words_only","random","dense_zeros_only","dense_all_ties","dense")
    report={"revision":revision(),"posthoc_fixed_diagnostic":True,"modes":cases,
            "random_tie_seed":7632,"checkpoint_sha256":CHECKPOINT,"kd_file_sha256":kd_sha,"splits":{}}
    for ds in ("sketchy","tuberlin"):
        a,_,qs,_,gs,raw,rgb=global_inputs(w,ds)
        words=read_words(a.vocabulary);_,scale=bare_words(a,words)
        src,tgt=(v.cuda() for v in load_words(a,words))
        for s,(qi,ql) in qs.items():
            pi,pl=gs[s]
            if (len(ql),len(pl),len(ql.unique()))!=SIZES[s]:raise ValueError(s)
            d=Data(raw[qi].cuda(),rgb[pi].cuda(),src,tgt,scale)
            q=d.query_emission/d.query_emission.sum(1,keepdim=True)
            dense=q@d.a.T
            q3=assign(q,1.,3,1);p3=assign(d.a,1.,3,1)
            sparse=q3@p3.T
            del d,q3,p3,q
            place=dest/s
            meta={"revision":revision(),"queries":len(ql),"gallery":len(pl),"classes":len(pl.unique()),
                  "gpu":torch.cuda.get_device_name(),"kd_sha256":kd_sha,"checkpoint_sha256":CHECKPOINT}
            stats={};values={}
            for mode in cases:
                cfg=Case(mode)
                scores=secondary_order(sparse,dense,mode)
                evaluate(a,place,s,scores,ql,pl,cfg,meta,mode=="dense_all_ties",kd)
                row=read(place/f"{s}-{config_id(cfg)}.json")
                values[mode]=tensor(place/"per-query"/f"{s}-{config_id(cfg)}.pt")
                stats[mode]={k:row[k] for k in ("mAP","precision","queries","classes","gallery","map_k","p_k")}
                if mode=="dense_all_ties":stats[mode]["kd"]=row["kd_check"]
                del scores
            for mode in cases:
                stats[mode]["versus_sparse"]=paired_delta(values[mode],values["words_only"],ql)["mAP"]
                stats[mode]["versus_all_words"]=paired_delta(values[mode],values["dense"],ql)["mAP"]
            coverage={}
            zero_pairs=(sparse==0).sum(1)
            for k in (100,200):
                coverage[str(k)]=float(((sparse.size(1)-zero_pairs)<k).float().mean())
            report["splits"][s]={"cases":stats,"fraction_all_pairs_zero":float(zero_pairs.double().sum()/sparse.numel()),
                "fraction_of_queries_with_fewer_positive_scores_than_k":coverage}
            atomic_json(dest/(s+".json"),report["splits"][s])
            print(s,{k:round(100*v["mAP"],4) for k,v in stats.items()},flush=True)
            del sparse,dense
    atomic_json(dest/"summary.json",report)
    files=sorted(p for p in dest.rglob("*") if p.is_file() and p.suffix in (".json",".pt"))
    z=w/"assignment-ties-full-evidence.zip"
    with zipfile.ZipFile(z,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=4) as out:
        for p in files:out.write(p,p.relative_to(w).as_posix())
    atomic_json(w/"assignment-ties-full-archive-info.json",{"sha":checksum(z),"size":z.stat().st_size,"count":len(files)})


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
