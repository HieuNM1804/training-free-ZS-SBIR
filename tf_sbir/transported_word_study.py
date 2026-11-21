"""Estimate visually grounded public concept meanings and transport them to query space."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from .concept_benchmark import atomic_json,checksum,read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES,bare_words
from .cross_diffusion_map import Config,Index
from .crossdomain_benchmark import digest,revision,SPLIT_HASHES
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import load_kd,paired_delta
from .semantic import read_words
from .shared_concept_likelihood import emission,fit_prior
from .transported_word_witness import discover,queries_and_scores,verify_rotation


FAMILY="transported-word-witness"


def key(c):return json.dumps(list(c),separators=(",",":"))


def configs():
    return [(n,r,.3,s,m,beta,g,mix) for n in (48,128,384) for r in (1,5)
            for m in ("stable","none") for s in ((.45,1.) if m=="stable" else (0.,))
            for beta in (.7,1.4) for g in (0.,.5) for mix in (.4,.8)]


def pick(table,mode,count):
    return [tuple(json.loads(k)) for k in sorted((k for k in table if json.loads(k)[4]==mode),
                                                  key=table.get,reverse=True)[:count]]


def raw_base(w,split,p,prob,q,source,scale):
    # The historical configuration is available independently from both newer studies.
    cache=w/"cache"/"pragmatic-gallery-competition"/(split+"-named-rgb16.pt")
    if cache.is_file():
        return torch.load(cache,mmap=True,weights_only=True)
    graph=Index(SimpleNamespace(p=p),rgb_only=True)
    cfg=Config(neighbors=16,bandwidth=.36,iterations=2,mode="rgb")
    rows=[graph.score(cfg,emission(qq,source,scale)@prob.T).cpu() for qq in q.split(256)]
    base=torch.cat(rows)
    # Use a separate writer path to allow concurrent studies to build the same fixed baseline.
    if not cache.is_file():
        save_tensor(base,cache)
    return base


def measure(w,split,query,y,p,g,source,target,scale,prior,base,indices,config_list,replay,workers):
    values,errors,checked,tops={},{},{},{}
    def record(c,s,y,pool):
        fast,order,_,_=map_values(s,y.cuda(),g.cuda(),split)
        nm=key(c)
        if replay:
            official=replay.score(order,y,pool)
            error=float((fast.cpu()-official).abs().max())
            errors[nm]=max(errors.get(nm,0.),error)
            if error>2e-6: raise ValueError("Evaluator mismatch")
            tops.setdefault(nm,[]).append(order[:,:25].int().cpu())
        else: official=fast.cpu()
        values.setdefault(nm,[]).append(official)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if replay:
            for ids in indices.split(192):record(("historical",),base[ids].cuda(),y[ids],pool)
        for count,r,anchor in sorted(set(c[:3] for c in config_list)):
            eligible=[c for c in config_list if c[:3]==(count,r,anchor)]
            bank=discover(p,source,target,scale,prior,count,r,anchor,54321)
            checks={"rotation":verify_rotation(bank),"reliable_fraction":float((bank["gate"]>0).float().mean()),
                    "mean_gate":float(bank["gate"].mean()),"mean_two_fold_margin":float(bank["margins"].mean()),
                    "selected_words":bank["named"].tolist(),"effective_labels_per_photo":float(bank["effective"])}
            checked[key((count,r,anchor))]=checks
            for ids in indices.split(192):
                qq=query[ids]
                orig=base[ids].cuda().clamp_min(1e-30).log()
                for cfg in eligible:
                    simple=queries_and_scores(qq,bank,scale,cfg[:7]).clamp_min(1e-30).log()
                    score=(1-cfg[7])*orig+cfg[7]*simple
                    record(cfg,score,y[ids],pool)
            print(split,"word hypotheses",count,r,"eligible",len(eligible),flush=True)
    return {k:torch.cat(v) for k,v in values.items()},errors,checked,{k:torch.cat(v) for k,v in tops.items()}


@torch.inference_mode()
def run(w,workers):
    folder=w/"results"/FAMILY
    protocol={"revision":revision(),"candidates":configs(),"method":"source-to-target word-specific isometry with cross-fitted gallery support",
              "pilot":"24 random sketches per class, seed 78215","selection":"mean official Sketchy mAP, then precision"}
    atomic_json(folder/"protocol.json",protocol|{"sha256":digest(protocol)})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Incorrect evaluator")
    for stage in ("pilot","shortlist","final"):
        if stage=="pilot": variants=configs()
        elif stage=="shortlist":
            pre=read(folder/"pilot.json")["means"]
            variants=pick(pre,"stable",3)+pick(pre,"none",2)
        else:
            pre=read(folder/"shortlist.json")["means"]
            best=pick(pre,"stable",1)[0]
            simple=pick(pre,"none",1)[0]
            lock={"primary":best,"optimized_untransported":simple,"full_sketchy_sha256":checksum(folder/"shortlist.json"),
                  "revision":revision()}
            atomic_json(folder/"freeze.json",lock|{"sha256":digest(lock)})
            variants={best,simple,(*best[:3],0.,"none",*best[5:]),(*best[:7],1.)}
            for m in ("raw_displacement","permuted_displacement","ungated","same_fold","visual_only"):
                variants.add((*best[:4],m,*best[5:]))
            variants=sorted(variants)
        rows={}
        for dataset in (("sketchy","tuberlin") if stage=="final" else ("sketchy",)):
            cfg,_,qs,_,gs,qraw,praw=global_inputs(w,dataset)
            words=read_words(cfg.vocabulary)
            if len(words)!=20931:raise ValueError("Wrong public word list")
            _,scale=bare_words(cfg,words)
            source,target=(v.cuda() for v in load_words(cfg,words))
            for split,(qi,y) in qs.items():
                pi,g=gs[split]
                if (len(qi),len(pi),len(g.unique()))!=SIZES[split] or digest(UNSEEN_CLASSES[split])!=SPLIT_HASHES[split]:
                    raise ValueError("Incomplete split")
                ids=stratified_sample(y,24,78215) if stage=="pilot" else torch.arange(len(qi))
                q,p=qraw[qi].cuda(),praw[pi].cuda()
                probs=(scale*(p@target.T)).softmax(1)
                base=raw_base(w,split,p,probs,q,source,scale)
                priors,_=fit_prior(probs)
                per,errors,checks,tops=measure(w,split,q,y,p,g,source,target,scale,priors["mle"],base,ids,variants,
                                              Replay(kd,g,split) if stage=="final" else None,workers)
                rows[split]={k:v.mean(0).tolist() for k,v in per.items()}
                k=max(rows[split],key=rows[split].get)
                print(stage,split,"best",k,rows[split][k],flush=True)
                if stage=="final":
                    pathname=folder/(split+".pt")
                    save_tensor({"metrics":per,"query_labels":y,"top25":tops[key(best)]},pathname)
                    baseline=key(("historical",))
                    stat=per[key(best)]-per[baseline]
                    check=checks[key(best[:3])]
                    report={"protocol":{"split":split,"queries":len(qi),"gallery":len(pi),"classes":len(g.unique()),"gpu":torch.cuda.get_device_name(),
                              "checkpoint":CHECKPOINT,"kd_sha256":KD_FUNCTION_SHA,"vocabulary_sha256":checksum(cfg.vocabulary),"revision":revision(),
                              "map_k":200 if split=="sketchy_2" else 0,"p_k":200 if split=="sketchy_2" else 100},
                            "scores":{k:dict(zip(("mAP","precision"),v)) for k,v in rows[split].items()},
                            "primary":key(best),"optimized_untransported":key(simple),"baseline":baseline,
                            "vs_controls":{k:paired_delta(per[key(best)],v,y) for k,v in per.items()},
                            "checks":checks,"top_selected_by_gallery_prior":[words[i] for i in check["selected_words"][:12]],
                            "query_ap_changes":{"up":int((stat[:,0]>0).sum()),"down":int((stat[:,0]<0).sum()),
                                               "quantiles":stat[:,0].quantile(torch.tensor([.1,.5,.9])).tolist()},
                            "kd_queries_per_condition":len(y),"kd_max_abs_error":errors,"per_query_sha256":checksum(pathname)}
                    atomic_json(folder/(split+".json"),report)
                del probs,priors,base
        if stage!="final":
            means={k:[sum(rows[s][k][i] for s in ("sketchy_1","sketchy_2"))/2 for i in range(2)] for k in rows["sketchy_1"]}
            atomic_json(folder/(stage+".json"),{"splits":rows,"means":means})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--workers",type=int,default=min(10,len(os.sched_getaffinity(0))))
    args=parser.parse_args()
    run(args.workspace,args.workers)
