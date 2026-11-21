"""Check individual calibrated scores under reversed, truncated and singleton batches."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from .concept_benchmark import atomic_json, read
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import digest, revision
from .foundation import load_words
from .public_word_calibration import Config, Index
from .retrieval_diagnosis import global_inputs, stratified_sample
from .semantic import read_words


@torch.inference_mode()
def run(w):
    folder=w/"results/public-word-calibration"
    lock=read(folder/"freeze.json")
    if lock.pop("sha256")!=digest(lock):
        raise ValueError("Invalid configuration")
    c=Config(**lock["winner"])
    result={"revision":revision(),"config":lock["winner"],"splits":{}}
    for dataset in ("sketchy","tuberlin"):
        a,_,qs,_,gs,raw,photos=global_inputs(w,dataset)
        words=read_words(a.vocabulary)
        src,tgt=(x.cuda() for x in load_words(a,words))
        _,scale=bare_words(a,words)
        model=Index(SimpleNamespace(source=src,target=tgt,scale=scale))
        for split,(qi,ql) in qs.items():
            pi,pl=gs[split]
            probes=stratified_sample(ql,2,1183)
            keep=stratified_sample(pl,8,9057).flip(0)
            queries=raw[qi[probes]].cuda()
            gallery=photos[pi].cuda()
            scores=torch.cat([model.score_images(c,queries,part) for part in gallery.split(768)],dim=1)
            pruned=model.score_images(c,queries.flip(0),gallery[keep])
            err=float((pruned-scores.flip(0)[:,keep]).abs().max())
            single=max(float((model.score_images(c,queries[j:j+1],gallery[k:k+1])[0,0]-scores[j,k]).abs())
                       for j,k in enumerate(keep[:5].tolist()))
            if max(err,single)>1e-4:
                raise ValueError("Scores depend on batch")
            result["splits"][split]={"queries":len(queries),"full_photos":len(gallery),"pruned_photos":len(keep),
                 "max_abs_pruned":err,"max_abs_singleton":single,"singletons":5}
    atomic_json(folder/"pair-invariance.json",result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
