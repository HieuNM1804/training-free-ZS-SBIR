"""Isolate which side loses retrieval evidence when the same Gaussian assignment is truncated."""

from argparse import ArgumentParser
from pathlib import Path

import torch
from .capture_query_metrics import evaluate_with_details
from .concept_benchmark import atomic_json
from .concept_likelihood import ranked_scores
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import revision
from .foundation import load_words
from .map_study import Data
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import paired_delta
from .retrieval_soft_assignment import FAMILY,assign
from .semantic import read_words


def official(score,ql,pl,split,second=None):
    def rows():
        for start in range(0,len(ql),64):
            x=score[start:start+64]
            if second is None:
                yield from ranked_scores(x).cpu()
                continue
            previous=second[start:start+len(x)].argsort(dim=1,descending=True,stable=True)
            order=previous.gather(1,x.gather(1,previous).argsort(dim=1,descending=True,stable=True))
            levels=torch.linspace(.9,-.9,len(pl),device=x.device).expand_as(x)
            yield from torch.empty_like(x).scatter_(1,order,levels).cpu()
    _,details=evaluate_with_details(rows(),ql,pl,split,"soft support "+split,4)
    return details


def coverage(p):
    vals=p.topk(64,dim=1).values.cumsum(1)
    return {str(k):{"mean":float(vals[:,k-1].mean()),"median":float(vals[:,k-1].median())} for k in (1,3,16,64)}


@torch.inference_mode()
def run(w):
    results={"revision":revision(),"seed":9991,"inverse_temperature":"native frozen CLIP on both domains",
             "normalization":"L1 each image independently","inference":"fixed diagnostics after selection","splits":{}}
    for dataset in ("sketchy","tuberlin"):
        a,_,qs,_,gs,raw,rgb=global_inputs(w,dataset)
        words=read_words(a.vocabulary);_,scale=bare_words(a,words);src,tgt=(v.cuda() for v in load_words(a,words))
        for s,(qi,ql) in qs.items():
            pi,pl=gs[s]
            selected=stratified_sample(ql,24,9991);labels=ql[selected]
            data=Data(raw[qi[selected]].cuda(),rgb[pi].cuda(),src,tgt,scale)
            q=data.query_emission/data.query_emission.sum(1,keepdim=True)
            p=data.a
            full=q@p.T
            reference=official(full,labels,pl,s)
            info={"queries":len(q),"gallery":len(p),"classes":len(ql.unique()),
                  "source_mass":coverage(q),"target_mass":coverage(p),"conditions":{}}
            def record(name,scores,tie_break=None):
                vals=official(scores,labels,pl,s,tie_break)
                info["conditions"][name]={"mAP":float(vals[:,0].mean()),"precision":float(vals[:,1].mean()),
                    "versus_full":paired_delta(vals,reference,labels)["mAP"]}
            info["conditions"]["all"]={"mAP":float(reference[:,0].mean()),"precision":float(reference[:,1].mean())}
            for k in (3,16,64):
                qsparse=assign(q,1.,k,1);psparse=assign(p,1.,k,1)
                both=qsparse@psparse.T
                record(f"both_{k}",both)
                if k==3:
                    record("only_query_3",qsparse@p.T)
                    record("only_photo_3",q@psparse.T)
                    record("both_3_dense_breaks_ties",both,full)
                    record("both_3_cosine",torch.nn.functional.normalize(qsparse)@torch.nn.functional.normalize(psparse).T)
                del qsparse,psparse,both
            results["splits"][s]=info
            print(s,{k:round(v["mAP"]*100,3) for k,v in info["conditions"].items()},flush=True)
    atomic_json(w/"results"/FAMILY/"support-audit.json",results)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
