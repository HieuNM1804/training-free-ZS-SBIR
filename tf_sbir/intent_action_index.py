"""Index full public-word actions over all gallery photos, entirely before query evaluation."""

import argparse
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .encoding import save_tensor
from .foundation import load_words
from .intent_utility import exposure, normalized_membership, verify_identity
from .lexical_population_index import groups_for
from .retrieval_diagnosis import global_inputs
from .semantic import read_words

FAMILY="latent-intent-ranking"
TEMPS=(.35,1.)


@torch.inference_mode()
def run(w,ds):
    cfg,_,_,paths,galleries,_,raw=global_inputs(w,ds)
    words=read_words(cfg.vocabulary)
    if len(words)!=20931:raise ValueError("Unexpected public vocabulary")
    _,target=load_words(cfg,words)
    target=target.cuda()
    _,scale=bare_words(cfg,words)
    exact=verify_identity(target.device)
    for s,(pi,_) in galleries.items():
        folder=w/"cache"/FAMILY
        file=folder/(s+"-actions.pt")
        desc=file.with_suffix(".json")
        sig=digest({"gallery":digest([str(paths[i]) for i in pi.tolist()]), "word_sha256":checksum(cfg.vocabulary),
                    "checkpoint":CHECKPOINT, "action":"native full text-to-photo ranking with stable sort v1"})
        if file.is_file() and desc.is_file() and read(desc)["signature"]==sig:continue
        photos=raw[pi].cuda()
        groups=groups_for(photos)
        sizes=groups.bincount()
        n=len(sizes)
        pair=torch.empty((len(words),n,n))
        first=torch.empty((len(words),n))
        f={temp:torch.empty((len(words),n)) for temp in TEMPS}
        worst=0.
        for start in range(0,len(words),32):
            sl=slice(start,min(start+32,len(words)))
            scores=target[sl]@photos.T
            order=scores.argsort(dim=1,descending=True,stable=True)
            matrix,margin=exposure(groups[order],n)
            pair[sl]=matrix.cpu()
            first[sl]=margin.cpu()
            for t in TEMPS:
                xx=(scores-scores.amax(1,keepdim=True)).mul(scale*t).exp()
                means=xx.new_zeros((len(xx),n)).index_add_(1,groups,xx)/sizes
                u=means/means.amax(1,keepdim=True).clamp_min(1e-20)
                norm=normalized_membership(u,sizes)
                worst=max(worst,float(((norm*sizes).sum(1)-1).abs().max()))
                f[t][sl]=norm.cpu()
            if start%4096==0:print("actions",s,start,len(words),flush=True)
        if worst>3e-5:raise ValueError("Expected normalized membership mass is inconsistent")
        save_tensor({"pair":pair,"first":first,"memberships":f,"groups":groups.cpu()},file)
        atomic_json(desc,{"signature":sig,"words":len(words),"gallery":len(groups),"groups":n,"enumeration_error":exact,
                          "max_membership_mass_error":worst})
        print("indexed",s,len(words),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),required=True)
    a=p.parse_args()
    run(a.workspace,a.dataset)
