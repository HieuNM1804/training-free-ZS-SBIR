"""Compare concordant layout evidence with pose aggregation alone on the same gallery index."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,checksum,read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES,bare_words
from .crossdomain_benchmark import digest,revision,SPLIT_HASHES
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .pose_layout_cache import FAMILY,POSES
from .pose_layout_evidence import target_index,code,check_worlds
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import load_kd,paired_delta
from .semantic import read_words


def key(c):return json.dumps(list(c),separators=(",",":"))


def grid():
    return [(prior,h,p,m,b) for prior in ("mle","uniform") for h in (2,8)
            for p in ("native","geometric","arithmetic") for m in ("original","cross","unbounded")
            for b in ((0.,) if m=="original" else (.5,1.5,3.))]


def pick(table,mode,num):
    return [tuple(json.loads(k)) for k in sorted((k for k in table if json.loads(k)[3]==mode),
                                                 key=table.get,reverse=True)[:num]]


@torch.inference_mode()
def run(w,workers):
    folder=w/"results"/FAMILY
    protocol={"revision":revision(),"grid":grid(),"primary":"cross-pose same-arrangement evidence",
              "pilot_seed":51073,"pilot_queries_each_class":24,"transfer":"decide on both full Sketchy splits"}
    atomic_json(folder/"protocol.json",protocol|{"sha256":digest(protocol)})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Changed KD function")
    for stage in ("pilot","shortlist","final"):
        if stage=="pilot": variants=grid()
        elif stage=="shortlist":
            means=read(folder/"pilot.json")["means"]
            variants=pick(means,"cross",3)+pick(means,"unbounded",2)+pick(means,"original",2)
        else:
            means=read(folder/"shortlist.json")["means"]
            best=pick(means,"cross",1)[0]
            simpler=pick(means,"unbounded",1)[0]
            tta=pick(means,"original",1)[0]
            lock={"primary":best,"best_unbounded":simpler,"best_tta":tta,"revision":revision(),
                  "full_sketchy_sha256":checksum(folder/"shortlist.json")}
            atomic_json(folder/"freeze.json",lock|{"sha256":digest(lock)})
            variants={best,simpler,tta,("uniform",1,"native","original",0.),(best[0],1,"native","original",0.),
                      (best[0],1,"native","mean",best[-1]),(*best[:3],"original",0.)}
            variants.update((*best[:3],m,best[-1]) for m in ("mean","squared","misaligned","phase_shift","unbounded"))
            variants=sorted(variants)
        table={}
        for dataset in (("sketchy","tuberlin") if stage=="final" else ("sketchy",)):
            cfg,_,qs,_,gs,qraw,praw=global_inputs(w,dataset)
            words=read_words(cfg.vocabulary)
            if len(words)!=20931:raise ValueError("Unexpected dictionary")
            _,scale=bare_words(cfg,words)
            src,tgt=(a.cuda() for a in load_words(cfg,words))
            for split,(qi,y) in qs.items():
                pi,g=gs[split]
                if (len(qi),len(pi),len(g.unique()))!=SIZES[split] or digest(UNSEEN_CLASSES[split])!=SPLIT_HASHES[split]:
                    raise ValueError("Incomplete split")
                ids=stratified_sample(y,24,51073) if stage=="pilot" else torch.arange(len(qi))
                actual=qi[ids]
                xf=torch.stack([torch.load(w/"cache"/FAMILY/dataset/f"pose-{h}.pt",weights_only=True,mmap=True)[actual]
                                for h in range(len(POSES))],1).cuda()
                reference=Replay(kd,g,split) if stage=="final" else None
                per,err,checks,top={}, {},{},{}
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    for prior in sorted(set(c[0] for c in variants)):
                        bank=target_index(w,split,praw[pi].cuda(),tgt,scale,prior)
                        these=[c for c in variants if c[0]==prior]
                        for start in range(0,len(ids),80):
                            bx=xf[start:start+80].float()
                            labels=y[ids[start:start+len(bx)]]
                            z=(bx.flatten(0,2)@src.T*scale).view(len(bx),len(POSES),4,-1)
                            if start==0:checks[prior]=check_worlds(z)
                            for c in these:
                                evidence=code(z,*c[1:])@bank.T
                                approx,order,_,_=map_values(evidence,labels.cuda(),g.cuda(),split)
                                nm=key(c)
                                if reference:
                                    observed=reference.score(order,labels,executor)
                                    error=float((approx.cpu()-observed).abs().max())
                                    err[nm]=max(err.get(nm,0.),error)
                                    if error>2e-6:raise ValueError("Original evaluator mismatch")
                                    if c==best:top.setdefault(nm,[]).append(order[:,:25].int().cpu())
                                else:observed=approx.cpu()
                                per.setdefault(nm,[]).append(observed)
                        del bank
                        print(stage,split,prior,"complete",flush=True)
                per={a:torch.cat(b) for a,b in per.items()}
                if any(v.shape!=(len(ids),2) for v in per.values()):raise ValueError("Incomplete rankings")
                table[split]={k:v.mean(0).tolist() for k,v in per.items()}
                winner=max(table[split],key=table[split].get)
                print(stage,split,"best",winner,table[split][winner],flush=True)
                if stage=="final":
                    path=folder/(split+".pt")
                    save_tensor({"metrics":per,"query_labels":y,"top25":torch.cat(top[key(best)])},path)
                    old=torch.load(w/"results/layout-conditioned-word-evidence"/(split+".pt"),weights_only=True,map_location="cpu")
                    if not torch.equal(y,old["query_labels"]):raise ValueError("Query mismatch with parent study")
                    oldinfo=read(w/"results/layout-conditioned-word-evidence"/(split+".json"))
                    obj={"protocol":{"split":split,"queries":len(y),"gallery":len(g),"classes":len(g.unique()),
                              "checkpoint":CHECKPOINT,"vocabulary_sha256":checksum(cfg.vocabulary),"kd_sha256":KD_FUNCTION_SHA,
                              "gpu":torch.cuda.get_device_name(),"revision":revision(),"map_k":200 if split=="sketchy_2" else 0,
                              "p_k":200 if split=="sketchy_2" else 100},
                         "scores":{k:dict(zip(("mAP","precision"),v)) for k,v in table[split].items()},"primary":key(best),
                         "best_contrast":key(simpler),"best_tta":key(tta),"baseline":key(("uniform",1,"native","original",0.)),
                         "vs_controls":{k:paired_delta(per[key(best)],v,y) for k,v in per.items()},
                         "vs_previous":{k:paired_delta(per[key(best)],old["metrics"][oldinfo[k]],y) for k in ("primary","best_contrast")},
                         "orbit_checks":checks,"native_validation":[read(w/"cache"/FAMILY/dataset/f"pose-{h}.json")["raw_pose_max_one_minus_cosine"] for h in range(len(POSES))],
                         "original_kd_queries_per_condition":len(qi),"kd_max_abs_error":err,"per_query_sha256":checksum(path)}
                    atomic_json(folder/(split+".json"),obj)
                del xf
        if stage!="final":
            means={k:[sum(table[s][k][i] for s in ("sketchy_1","sketchy_2"))/2 for i in range(2)] for k in table["sketchy_1"]}
            atomic_json(folder/(stage+".json"),{"splits":table,"means":means})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--workers",type=int,default=min(10,len(os.sched_getaffinity(0))))
    args=parser.parse_args()
    run(args.workspace,args.workers)
