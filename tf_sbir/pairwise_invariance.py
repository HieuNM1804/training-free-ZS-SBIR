"""Check the fixed-uniform pair score under batch changes and removal of unrelated gallery images."""

import argparse
from pathlib import Path

import torch

from .concept_benchmark import atomic_json
from .conditional_semantic_index import bare_words
from .data import UNSEEN_CLASSES
from .foundation import load_words
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import paired_delta,tensor
from .semantic import read_words
from .shared_concept_likelihood import FAMILY,REFERENCE,emission


@torch.inference_mode()
def run(w):
    root=w/"results"/FAMILY
    data={}
    for dataset in ("sketchy","tuberlin"):
        args,_,qs,_,gs,raw,rgb=global_inputs(w,dataset)
        names=read_words(args.vocabulary)
        _,temp=bare_words(args,names)
        src,target=(t.cuda() for t in load_words(args,names))
        for s,(qi,ql) in qs.items():
            pi,pl=gs[s]
            chosen=stratified_sample(ql,1,9201); sample=stratified_sample(pl,8,5431)
            q=raw[qi[chosen]].cuda();photos=rgb[pi].cuda()
            def keys(images):
                return torch.cat([((block@target.T)*temp).softmax(1) for block in images.split(384)])
            full=emission(q,src,temp)@keys(photos).T
            reduced=emission(q,src,temp)@keys(photos[sample.flip(0)]).T
            small=full[:,sample]
            single=torch.cat([emission(v[None],src,temp) for v in q])@keys(photos[sample]).T
            sub="heldout" if s=="tuberlin" else "sketchy"
            result=tensor(root/sub/"per-query"/f"{s}-uniform.pt")
            native=tensor(w/"results"/REFERENCE/sub/"per-query"/f"{s}-native.pt")
            data[s]={"query_classes":len(chosen),"tested_gallery_items":len(sample),"full_gallery":len(pi),
                "max_abs_score_change_after_gallery_replacement":float((small-reduced.flip(1)).abs().max()),
                "max_abs_score_change_single_vs_query_batch":float((single-small).abs().max()),
                "uniform_vs_native":paired_delta(result,native,ql),
                "class_names":[UNSEEN_CLASSES[s][i] for i in ql.unique().tolist()]}
    atomic_json(root/"pairwise-invariance.json",data)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
