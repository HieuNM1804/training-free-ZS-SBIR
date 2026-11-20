"""Check target-free and cross-fitted priors without changing either official retrieval gallery."""

import argparse
from datetime import datetime,timezone
from pathlib import Path

import torch

from .capture_query_metrics import evaluate_with_details
from .concept_benchmark import atomic_json,checksum,read
from .concept_likelihood import ranked_scores
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import digest,revision
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import paired_delta,tensor
from .semantic import read_words
from .shared_concept_likelihood import FAMILY,PRIMARY,fit_prior,posterior,score_blocks,target_inputs


@torch.inference_mode()
def source(w):
    root=w/"results"/FAMILY
    plan=read(root/"transfer-tests-plan.json")
    if any(plan["tu_rows"].values()):
        raise ValueError("Strict transfer diagnostic was not preregistered")
    a,_,_,pp,_,_,photos=global_inputs(w,"sketchy")
    if len(set(pp))!=len(pp):
        raise ValueError("Duplicate gallery images in the source prior")
    words=read_words(a.vocabulary)
    _,scale=bare_words(a,words)
    _,photo_words=load_words(a,words)
    em=target_inputs(photos.cuda(),photo_words.cuda(),scale)
    priors,stat=fit_prior(em)
    file=root/"strict/sketchy-prior.pt"
    if (root/"strict/source-lock.json").exists():
        old=read(root/"strict/source-lock.json")
        if old["prior_sha256"]!=checksum(file):
            raise ValueError("Damaged immutable source prior")
        return
    save_tensor(priors["mle"].cpu(),file)
    info={"utc":datetime.now(timezone.utc).isoformat(),"revision_of_diagnostic":revision(),"estimation":"unique Sketchy gallery files",
          "photos":len(pp),"words":len(words),"prior_sha256":checksum(file),"plan_sha256":digest(plan),"fit":stat,
          "new_tu_scores_at_lock":len(list((root/"heldout").glob("tuberlin-*.json")))}
    atomic_json(root/"strict/source-lock.json",info)
    print("Frozen source prior",info["photos"],info["prior_sha256"],info["new_tu_scores_at_lock"],flush=True)


@torch.inference_mode()
def evaluate(w,dataset):
    root=w/"results"/FAMILY; where=root/"strict"
    original=read(root/"protocol.json")
    lock=read(where/"source-lock.json"); plan=read(root/"transfer-tests-plan.json")
    if checksum(where/"sketchy-prior.pt")!=lock["prior_sha256"] or digest(plan)!=lock["plan_sha256"]:
        raise ValueError("Invalid strict transfer lock")
    if dataset=="tuberlin":
        both=read(root/"joint-freeze.json")
        if plan["utc"]>=both["utc"] or original["sha256"]!=both["studies"][FAMILY]["freeze_sha256"]:
            raise ValueError("Late transfer plan")
    a,_,qs,_,gs,raw,photos=global_inputs(w,dataset)
    words=read_words(a.vocabulary);_,scale=bare_words(a,words)
    source_words,target_words=(x.cuda() for x in load_words(a,words))
    fixed=tensor(where/"sketchy-prior.pt").cuda()
    out={"immutable_prior":lock["prior_sha256"],"splits":{}}
    for s,(qi,ql) in qs.items():
        pi,pl=gs[s];q=raw[qi].cuda()
        em=target_inputs(photos[pi].cuda(),target_words,scale)
        random=torch.randperm(len(pi),generator=torch.Generator().manual_seed(plan["seed"]))
        ids=[v.cuda() for v in random.tensor_split(2)]
        table={}
        sub=root/("heldout" if dataset=="tuberlin" else "sketchy")
        standard=tensor(sub/"per-query"/f"{s}-{PRIMARY}.pt")
        for name in ("source-prior","reference-half-a","reference-half-b"):
            if name=="source-prior":
                prior=fixed; opposite=torch.arange(len(pi),device="cuda"); fit="source"
            else:
                part=0 if name.endswith("-a") else 1
                priors,fit=fit_prior(em[ids[part]])
                prior=priors["mle"];opposite=ids[1-part]
            held=em[opposite]
            ll_gain=(held@prior).clamp_min(1e-38).log()-held.mean(1).clamp_min(1e-38).log()
            del held
            prob=posterior(em,prior)
            def rows():
                for scores in score_blocks(q,prob,source_words,scale,PRIMARY):
                    yield from ranked_scores(scores).cpu()
            (ap,p,*_),per=evaluate_with_details(rows(),ql,pl,s,f"Transfer of prior {s} {name}",4)
            save_tensor(per,where/"per-query"/f"{s}-{name}.pt")
            # Additional diagnostic below has a smaller gallery, explicitly
            # distinguished from the full-gallery benchmark above.
            held_labels=pl[opposite.cpu()].cuda()
            h_count=0; start=0; k=200 if s=="sketchy_2" else 100
            for scores in score_blocks(q,prob,source_words,scale,PRIMARY):
                idx=scores[:,opposite].topk(k).indices
                h_count+=int((held_labels[idx]==ql[start:start+len(idx),None].cuda()).sum())
                start+=len(idx)
            table[name]={"mAP":float(ap),"precision":float(p),"full_gallery":len(pi),"full_queries":len(qi),
                "delta_vs_gallery_fitted":paired_delta(per,standard,ql),"mean_predictive_loglik_gain_over_uniform":float(ll_gain.mean()),
                "diagnostic_opposite_gallery":len(opposite),"diagnostic_opposite_precision":h_count/(len(q)*k),
                "fit":fit,"per_class_AP_P":{n:per[ql==i].mean(0).tolist() for i,n in enumerate(UNSEEN_CLASSES[s])}}
            del prob
        out["splits"][s]=table
        print(s,{k:(v["mAP"],v["precision"]) for k,v in table.items()},flush=True)
    atomic_json(where/f"{dataset}-transfer.json",out)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--stage",choices=("source","sketchy","tuberlin"),required=True)
    opt=p.parse_args()
    if opt.stage=="source":
        source(opt.workspace)
    else:
        evaluate(opt.workspace,opt.stage)
