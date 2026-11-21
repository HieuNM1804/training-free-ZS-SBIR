"""Select and evaluate disjoint, cross-fitted ordinal tests of competing gallery modes."""

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
from .ordinal_rival_witness import OrdinalRivals
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import load_kd,paired_delta
from .semantic import read_words
from .shared_concept_likelihood import fit_prior


FAMILY="ordinal-rival-witness"


def key(c):return json.dumps(list(c),separators=(",",":"))


def grid():
    return [(groups,view,length,mode,expo,weight,op) for groups in (32,96) for view in ("rgb","joint")
            for length in (4,18) for mode in ("odds","offset") for expo in (1.,4.)
            for weight in (.25,1.) for op in ("add","replace")]


def check(model,q,base,count,mode):
    x=(q@model.source.T)
    a=model.from_word_scores(x,count,mode)
    phi=x+x.pow(3)*.4+2
    b=model.from_word_scores(phi,count,mode)
    err=float((a-b).abs().max())
    projected,_,w=model.calibration(base,a,2.)
    pair=base.new_zeros((len(base),model.count,model.count))
    left,right=model.ordering
    pair[:,left,right]=a
    pair[:,right,left]=-a
    normal=(w[:,None,:]*(projected[:,:,None]-projected[:,None,:]-pair)).sum(2)
    res=float(normal.abs().max())
    if res>3e-4:raise ValueError("Pairwise weighted least squares failed")
    return {"monotone_transform_max_error":err,"weighted_least_squares_residual":res,
            "mean_reliability":float(model.stability[:,:count].mean()),"word_count":len(model.terms)}


def evaluate(configs,query,y,p,g,prob,banks,prior,indices,base,split,replay,workers):
    per,errors,tops,checks={}, {},{},{}
    def record(c,s,labels,pool):
        fast,order,_,_=map_values(s,labels.cuda(),g.cuda(),split)
        code=key(c)
        if replay:
            val=replay.score(order,labels,pool)
            err=float((fast.cpu()-val).abs().max())
            if err>2e-6:raise ValueError("Original KD disagreement")
            errors[code]=max(errors.get(code,0.),err)
            tops.setdefault(code,[]).append(order[:,:25].cpu().int())
        else:val=fast.cpu()
        per.setdefault(code,[]).append(val)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if replay:
            for ids in indices.split(128):record(("historical",),base[ids].cuda(),y[ids],pool)
        for n,view in sorted(set(c[:2] for c in configs)):
            game=OrdinalRivals(p,prob,banks[1],banks[0],prior,n,view)
            these=[c for c in configs if c[:2]==(n,view)]
            for ids in indices.split(96):
                scores=base[ids].cuda()
                q=query[ids]
                cache={}
                for c in these:
                    length,mode,expo,weight,op=c[2:]
                    if (length,mode) not in cache:cache[length,mode]=game.comparisons(q,length,mode)
                    record(c,game.rank(scores,cache[length,mode],expo,weight,op),y[ids],pool)
            checks[key((n,view))]=check(game,query[indices[:4]],base[indices[:4]].cuda(),18,"odds")
            del game
            print(split,n,view,"finished",flush=True)
    return {k:torch.cat(v) for k,v in per.items()},errors,{k:torch.cat(v) for k,v in tops.items()},checks


@torch.inference_mode()
def run(w,workers):
    folder=w/"results"/FAMILY
    spec={"revision":revision(),"grid":grid(),"selection":"balanced pilot, best 5 then equal Sketchy full mAP, precision",
          "folds":"two balanced reference-gallery halves within each unlabeled mode","pilot_seed":39771,"pilot_per_class":24}
    atomic_json(folder/"protocol.json",spec|{"sha256":digest(spec)})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("KD evaluator changed")
    for stage in ("pilot","shortlist","final"):
        if stage=="pilot": configs=grid()
        elif stage=="shortlist":
            tab=read(folder/"pilot.json")["means"]
            configs=[tuple(json.loads(k)) for k in sorted(tab,key=tab.get,reverse=True)[:5]]
        else:
            tab=read(folder/"shortlist.json")["means"]
            chosen=tuple(json.loads(max(tab,key=tab.get)))
            obj={"primary":chosen,"revision":revision(),"full_sha256":checksum(folder/"shortlist.json")}
            atomic_json(folder/"freeze.json",obj|{"sha256":digest(obj)})
            configs={chosen,(*chosen[:4],0.,*chosen[5:]),(*chosen[:6],"replace" if chosen[-1]=="add" else "add")}
            configs.update((*chosen[:3],m,*chosen[4:]) for m in ("signed","continuous","identity_shuffle","uniform_edges","odds","offset"))
            configs=sorted(configs)
        table={}
        for dataset in (("sketchy","tuberlin") if stage=="final" else ("sketchy",)):
            cfg,_,qs,_,gs,qraw,praw=global_inputs(w,dataset)
            words=read_words(cfg.vocabulary)
            if len(words)!=20931:raise ValueError("Different word bank")
            _,scale=bare_words(cfg,words)
            banks=tuple(v.cuda() for v in load_words(cfg,words))
            for split,(qi,y) in qs.items():
                pi,g=gs[split]
                if (len(qi),len(pi),len(g.unique()))!=SIZES[split] or digest(UNSEEN_CLASSES[split])!=SPLIT_HASHES[split]:
                    raise ValueError("Incomplete split")
                p,q=praw[pi].cuda(),qraw[qi].cuda()
                prob=(scale*(p@banks[1].T)).softmax(1)
                prior=fit_prior(prob)[0]["mle"]
                base=torch.load(w/"cache/pragmatic-gallery-competition"/(split+"-named-rgb16.pt"),mmap=True,weights_only=True)
                indices=stratified_sample(y,24,39771) if stage=="pilot" else torch.arange(len(y))
                per,err,tops,checks=evaluate(configs,q,y,p,g,prob,banks,prior,indices,base,split,
                                              Replay(kd,g,split) if stage=="final" else None,workers)
                table[split]={k:v.mean(0).tolist() for k,v in per.items()}
                bestcell=max(table[split],key=table[split].get)
                print(stage,split,bestcell,table[split][bestcell],flush=True)
                if stage=="final":
                    path=folder/(split+".pt")
                    save_tensor({"metrics":per,"top25":tops[key(chosen)],"query_labels":y},path)
                    basekey=key(("historical",))
                    old=g[tops[basekey][:,0].long()]==y
                    new=g[tops[key(chosen)][:,0].long()]==y
                    rec={"protocol":{"split":split,"queries":len(y),"gallery":len(g),"classes":len(g.unique()),"checkpoint":CHECKPOINT,
                                     "vocabulary_sha256":checksum(cfg.vocabulary),"kd_sha256":KD_FUNCTION_SHA,"revision":revision(),
                                     "gpu":torch.cuda.get_device_name(),"map_k":200 if split=="sketchy_2" else 0,"p_k":200 if split=="sketchy_2" else 100},
                         "scores":{k:dict(zip(("mAP","precision"),v)) for k,v in table[split].items()},"primary":key(chosen),"baseline":basekey,
                         "vs_controls":{k:paired_delta(per[key(chosen)],v,y) for k,v in per.items()},
                         "top1_changes":{"fixed":int((~old&new).sum()),"lost":int((old&~new).sum())},"checks":checks,
                         "original_kd_queries_per_condition":len(qi),"kd_max_abs_error":err,"per_query_sha256":checksum(path)}
                    atomic_json(folder/(split+".json"),rec)
        if stage!="final":
            means={k:[sum(table[s][k][i] for s in ("sketchy_1","sketchy_2"))/2 for i in range(2)] for k in table["sketchy_1"]}
            atomic_json(folder/(stage+".json"),{"splits":table,"means":means})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--workers",type=int,default=min(10,len(os.sched_getaffinity(0))))
    args=parser.parse_args()
    run(args.workspace,args.workers)
