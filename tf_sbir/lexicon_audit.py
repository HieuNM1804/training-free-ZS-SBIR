"""Measure coverage and alternate-seed retrieval after locking the gallery vocabulary method."""

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,read
from .crossdomain_benchmark import digest
from .data import UNSEEN_CLASSES
from .foundation import load_words
from .gallery_vocabulary_cover import Config,Index
from .residual_benchmark import load_lock
from .retrieval_diagnosis import global_inputs
from .semantic import read_words


def plain(s):
    return " ".join(s.lower().replace("-"," ").replace("_"," ").split())


@torch.inference_mode()
def run(w):
    root=w/"results"; family="gallery-vocabulary-cover"
    lock=load_lock(root/family/"confirm")
    joint=read(root/"residual-study/final-selection.json")
    sig=joint.pop("sha256")
    if digest(joint)!=sig or joint["families"][family]["freeze_sha256"]!=lock["freeze_sha256"]:
        raise ValueError("Unfrozen study")
    plan=read(root/"residual-study/lexicon-stability-plan.json")
    if plan["tu_rows"]!=0 or plan["utc"]>=joint["utc"]:
        raise ValueError("Stability check was not specified before heldout scoring")
    c=Config(**lock["winner"])
    out={"method":c.name,"freeze_sha256":lock["freeze_sha256"],"plan":plan,"splits":{}}
    for dataset in ("sketchy","tuberlin"):
        a,_,qs,_,gs,raw,photos=global_inputs(w,dataset)
        words=read_words(a.vocabulary)
        source,target=(x.cuda() for x in load_words(a,words))
        for s,(qi,ql) in qs.items():
            pi,pl=gs[s]; p,q=photos[pi].cuda(),raw[qi].cuda(); y,z=ql.cuda(),pl.cuda()
            index=Index(p,source,target)
            sample=torch.randperm(len(p),generator=torch.Generator().manual_seed(711))[:1536].cuda()
            keep=torch.ones(len(p),device="cuda",dtype=torch.bool)
            keep[sample]=False
            r={"coverage":{},"stability":{}}
            original_terms=None
            k=200 if s=="sketchy_2" else 100
            for selection in ("frequency","cover","hybrid","random"):
                chosen=index.select(replace(c,selection=selection))
                best=(p@target[chosen].T).amax(1)
                quality=((best-index.values[:,0])/.035).exp().clamp_max(1.)
                vocabulary=[words[j] for j in chosen.tolist()]
                literal={plain(word) for word in vocabulary}
                names=UNSEEN_CLASSES[s]
                r["coverage"][selection]={"mean_outside_reference":float(quality[keep].mean()),
                    "photos_outside_reference":int(keep.sum()),"terms":vocabulary,
                    "covered_exact_class_names":[n for n in names if plain(n) in literal],
                    "mean_coverage_by_class":{n:float(quality[(z==i)&keep].mean()) for i,n in enumerate(names)}}
            for seed in plan["seeds"]:
                index.orders.pop("hybrid",None)
                if c.selection in ("cover","hybrid"):
                    index.orders["cover"]=index.cover(seed)
                index.banks.clear()
                chosen=set(index.select(c).tolist())
                if original_terms is None:
                    original_terms=chosen
                perquery=[]; start=0
                for scores in index.scores(q,c):
                    ids=scores.argsort(dim=1,descending=True,stable=True)[:,:k]
                    relevant=z[ids]==y[start:start+len(ids),None]
                    perquery.append(relevant.float().mean(1).cpu()); start+=len(ids)
                precision=torch.cat(perquery)
                r["stability"][str(seed)]={"queries":len(precision),"precision":float(precision.mean()),
                    "overlap_with_original":len(chosen&original_terms)/len(chosen|original_terms),
                    "per_class":{n:float(precision[ql==i].mean()) for i,n in enumerate(UNSEEN_CLASSES[s])}}
                if c.selection not in ("cover","hybrid"):
                    for other in plan["seeds"][1:]:
                        r["stability"][str(other)]=r["stability"][str(seed)]
                    break
            out["splits"][s]=r
            print(s,[u["precision"] for u in r["stability"].values()],flush=True)
    atomic_json(root/"residual-study/lexicon-audit.json",out)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
