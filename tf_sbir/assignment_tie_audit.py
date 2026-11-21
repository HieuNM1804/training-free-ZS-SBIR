"""Measure how exact score ties affect sparse-assignment conclusions at fixed parameters."""

from argparse import ArgumentParser
from pathlib import Path
import torch

from .capture_query_metrics import evaluate_with_details
from .concept_benchmark import atomic_json
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import digest,revision
from .foundation import load_words
from .image_retrieval_round import verified
from .map_study import Data
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import paired_delta,tensor
from .retrieval_soft_assignment import FAMILY,Config,Index
from .semantic import read_words


def rank_rows(scores,permutation):
    scale=torch.linspace(.9,-.9,len(permutation),device=scores.device)
    for block in scores.split(100):
        order=permutation[block[:,permutation].argsort(dim=1,descending=True,stable=True)]
        yield from torch.empty_like(block).scatter_(1,order,scale.expand_as(block)).cpu()


@torch.inference_mode()
def run(w):
    root=w/"results"/FAMILY
    lock=verified(root/"freeze.json")
    results={"revision":revision(),"method_revision":lock["revision"],"photo_order_seed":7632,
             "query_sample_seed":9991,"no_parameter_selection":True,"splits":{}}
    for dataset in ("sketchy","tuberlin"):
        a,_,qs,_,gs,raw,rgb=global_inputs(w,dataset)
        words=read_words(a.vocabulary);_,scale=bare_words(a,words)
        src,tgt=(v.cuda() for v in load_words(a,words))
        for s,(qi,ql) in qs.items():
            pi,pl=gs[s]
            selected=stratified_sample(ql,24,9991)
            qlabel=ql[selected]
            index=Index(Data(raw[qi[selected]].cuda(),rgb[pi].cuda(),src,tgt,scale))
            k=200 if s=="sketchy_2" else 100
            stable=torch.arange(len(pl),device="cuda")
            shuffled=torch.randperm(len(pl),generator=torch.Generator().manual_seed(7632)).cuda()
            out={}
            for name in ("primary","hard_assignment","paper_sparse"):
                cfg=Config(**lock["controls"][name])
                scores=index.score(cfg)
                old=score(scores,stable,qlabel,pl,s,name)
                new=score(scores,shuffled,qlabel,pl,s,name)
                ranked=scores.topk(k,dim=1).values[:,-1,None]
                ties=(scores==ranked).sum(1)
                id=digest(lock["controls"][name])[:16]
                ref=tensor(root/("heldout" if s=="tuberlin" else "sketchy")/"per-query"/f"{s}-{id}.pt")[selected]
                out[name]={"official_sample_AP_P":ref.mean(0).tolist(),"original_order_AP_P":old.mean(0).tolist(),
                    "shuffled_ties_AP_P":new.mean(0).tolist(),"shuffled_minus_original":paired_delta(new,old,qlabel),
                    "max_abs_mean_recompute":float((ref.mean(0)-old.mean(0)).abs().max()),
                    "zero_scores_fraction":float((scores==0).float().mean()),
                    "fraction_queries_with_cutoff_tie":float((ties>1).float().mean()),
                    "median_photos_tied_at_cutoff":float(ties.float().median())}
                del scores
            results["splits"][s]={"queries":len(qlabel),"gallery":len(pl),"P_cutoff":k,"conditions":out}
            print(s,{k:round(100*v["shuffled_ties_AP_P"][0],3) for k,v in out.items()},flush=True)
    atomic_json(root/"tie-audit.json",results)


def score(scores,perm,ql,pl,s,name):
    _,vals=evaluate_with_details(rank_rows(scores,perm),ql,pl,s,name+" "+s,4)
    return vals


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
