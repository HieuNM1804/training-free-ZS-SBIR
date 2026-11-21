"""A fixed, domain-name-free scorer and controlled tests of domain-specific assumptions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import os
from pathlib import Path
import time

import torch

from .concept_benchmark import atomic_json,checksum
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES,bare_words
from .crossdomain_benchmark import SPLIT_HASHES,digest,revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .domain_neutral_witness import DomainNeutralIndex,WordWitness,native_diffusion
from .encoding import save_tensor
from .foundation import load_words
from .gallery_lexical_feedback import LexicalFeedback
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import load_kd,paired_delta
from .semantic import read_words
from .shared_concept_likelihood import emission


FAMILY="domain-agnostic-word-feedback"
PRIMARY="literal_native"
CONDITIONS=(PRIMARY,"literal_identity","literal_legacy","named_native","named_legacy")


def check_distinct_witness(profiles,queries):
    # Twelve real gallery profiles, including one single-owner word to exercise
    # unsupported witnesses. This check is independent of benchmark labels.
    p=profiles[:12].clone()
    w=int(p[0].argmax())
    p[1:,w]=0
    small=WordWitness(p)
    p=small.profile
    l=queries[:3]
    out=small.scores(l)
    e=l@p.T
    truth=[]
    independent=[]
    for i in range(len(p)):
        other=p.clone()
        other[i]=0
        conditional=(e@other)/other.sum(0).clamp_min(1e-30)
        truth.append((conditional*l*p[i]).sum(1))
        independent.append(e[:,i]*(conditional*p[i]).sum(1))
    ref=torch.stack(truth,1)
    denom=ref.abs().amax(1,keepdim=True).clamp_min(1e-30)
    error=max(float(((out["witness"]-ref).abs()/denom).max()),
              float(((out["independent"]-torch.stack(independent,1)).abs()/denom).max()))
    perm=torch.arange(len(p)-1,-1,-1,device=p.device)
    reordered=WordWitness(p[perm]).scores(l)["witness"][:,perm]
    single=torch.cat([small.scores(row)["witness"] for row in l.split(1)])
    perm_error=float(((out["witness"]-reordered).abs()/denom).max())
    batch_error=float(((out["witness"]-single).abs()/denom).max())
    if max(error,perm_error,batch_error)>1e-4:
        raise ValueError("A public scoring-path invariant failed")
    return {"explicit_double_sum_error":error,"gallery_permutation_error":perm_error,
            "separate_query_error":batch_error,"fixture_single_owner_word_index":w}


def build(kind,p,bare,named,scale):
    if kind==PRIMARY:
        index=DomainNeutralIndex(p,bare,scale)
        return index.score,index.model.profile,index.graph_statistics
    text,graph=kind.split("_")
    source,target=(bare,bare) if text=="literal" else named
    stats={}
    if graph=="legacy":
        old=LexicalFeedback(p,target,scale)
        profiles=old.index
        del old
    else:
        profiles=((p@target.T)*scale).softmax(1)
        if graph=="native":
            profiles,stats=native_diffusion(p,profiles,scale)
    model=WordWitness(profiles)
    def score(queries):
        return model.scores(emission(queries,source,scale))
    return score,model.profile,stats


def score_split(folder,meta,q,p,ql,pl,bare,named,scale,kd,workers):
    split=meta["split"]
    replay=Replay(kd,pl,split)
    labels=pl.cuda()
    values,errors,checks,statistics={}, {},{},{}
    t0=time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for kind in CONDITIONS:
            run,profiles,stats=build(kind,p,bare,named,scale)
            source=bare if kind.startswith("literal") else named[0]
            checks[kind]=check_distinct_witness(profiles,emission(q[:3],source,scale))
            statistics[kind]=stats
            for start in range(0,len(q),192):
                inp=q[start:start+192]
                y=ql[start:start+len(inp)]
                scores=run(inp)
                if kind==PRIMARY:
                    scores["rgb"]=inp@p.T
                for method,score in scores.items():
                    key=kind+"_"+method
                    if not torch.isfinite(score).all():
                        raise ValueError("Nonfinite "+key)
                    fast,order,_,_=map_values(score,y.cuda(),labels,split)
                    original=replay.score(order,y,pool)
                    err=float((fast.cpu()-original).abs().max())
                    if err>2e-6:
                        raise ValueError("Original evaluator mismatch "+key)
                    errors[key]=max(errors.get(key,0.),err)
                    values.setdefault(key,[]).append(original)
            del run,profiles
            print(split,kind,"complete",flush=True)
    values={k:torch.cat(v) for k,v in values.items()}
    if any(x.shape!=(len(q),2) for x in values.values()):
        raise ValueError("Incomplete rankings")
    details=folder/(split+".pt")
    save_tensor({"query_labels":ql,"metrics":values},details)
    primary=values[PRIMARY+"_witness"]
    result={"protocol":meta,"scores":{k:dict(zip(("mAP","precision"),v.mean(0).tolist())) for k,v in values.items()},
            "primary_vs":{k:paired_delta(primary,v,ql) for k,v in values.items()},
            "within_condition":{k:{m:paired_delta(values[k+"_witness"],values[k+"_"+m],ql)
                                   for m in ("linear","independent")} for k in CONDITIONS},
            "map_k":replay.cutoffs[0],"p_k":replay.cutoffs[1],"kd_queries_per_method":len(ql),
            "kd_max_abs_error":errors,"implementation_checks":checks,"gallery_statistics":statistics,
            "elapsed_seconds":time.perf_counter()-t0,"per_query_sha256":checksum(details)}
    atomic_json(folder/(split+".json"),result)
    print(split,result["scores"],flush=True)


@torch.inference_mode()
def run(w,workers):
    folder=w/"results"/FAMILY
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Changed KD evaluator")
    protocol={"primary":PRIMARY,"conditions":CONDITIONS,"revision":revision(),"checkpoint":CHECKPOINT,
              "selection":"Primary fixed before any score in this study; no parameter selection from any split",
              "text":"literal public term, identical for both domains","gallery":"one all-gallery nonself Gibbs step",
              "scale":"frozen checkpoint image-text logit scale","query":"one image at a time, no query statistics"}
    atomic_json(folder/"protocol.json",protocol|{"sha256":digest(protocol)})
    for dataset in ("sketchy","tuberlin"):
        if dataset=="tuberlin":
            atomic_json(folder/"freeze.json",{"primary":PRIMARY,"revision":revision(),"protocol_sha256":checksum(folder/"protocol.json"),
                        "selection":"none","previous_scores":{s:checksum(folder/(s+".json")) for s in ("sketchy_1","sketchy_2")}})
        cfg,_,queries,_,galleries,qraw,praw=global_inputs(w,dataset)
        words=read_words(cfg.vocabulary)
        if len(words)!=20931:
            raise ValueError("Unexpected public vocabulary")
        bare,scale=bare_words(cfg,words)
        named=tuple(x.cuda() for x in load_words(cfg,words))
        for split,(qi,ql) in queries.items():
            pi,pl=galleries[split]
            if (len(ql),len(pl),len(pl.unique()))!=SIZES[split] or digest(UNSEEN_CLASSES[split])!=SPLIT_HASHES[split]:
                raise ValueError("Unexpected split")
            meta={"split":split,"queries":len(ql),"gallery":len(pl),"classes":len(pl.unique()),"gpu":torch.cuda.get_device_name(),
                  "revision":revision(),"checkpoint":CHECKPOINT,"kd_function_sha256":KD_FUNCTION_SHA,"logit_scale":scale,
                  "vocabulary_sha256":checksum(cfg.vocabulary),"protocol_sha256":checksum(folder/"protocol.json")}
            score_split(folder,meta,qraw[qi].cuda(),praw[pi].cuda(),ql,pl,bare,named,scale,kd,workers)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--workers",type=int,default=min(10,len(os.sched_getaffinity(0))))
    args=parser.parse_args()
    run(args.workspace,args.workers)
