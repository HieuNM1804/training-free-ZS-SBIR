"""Test robust, contrastive descriptions of unlabeled gallery alternatives in retrieval."""

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
from .pragmatic_gallery import GalleryGame,check_removed_photo_bounds,unit_rows
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import load_kd,paired_delta
from .semantic import read_words
from .shared_concept_likelihood import emission


FAMILY="pragmatic-gallery-competition"


def key(x): return json.dumps(list(x),separators=(",",":"))


def grid():
    return [(view,num,pool,alpha,bound,"sibling",within,weight)
            for view in ("rgb","joint") for num in (64,256)
            for pool in ("arith","geom") for alpha in (0,2,8,24)
            for within in (.35,1.) for weight in (.5,1.)
            for bound in ((False,) if alpha==0 else (True,))]


def choose(table,which,k):
    def accept(c): return (c[3]>0 if which=="main" else c[3]==0)
    return [tuple(json.loads(v)) for v in sorted((c for c in table if accept(json.loads(c))),
            key=table.get,reverse=True)[:k]]


def base_index(w,split,p,prob,q,source,scale):
    file=w/"cache"/FAMILY/(split+"-named-rgb16.pt")
    if not file.exists():
        graph=Index(SimpleNamespace(p=p),rgb_only=True)
        cfg=Config(neighbors=16,bandwidth=.36,iterations=2,mode="rgb")
        rows=[]
        for query in q.split(256):
            seed=emission(query,source,scale)@prob.T
            rows.append(graph.score(cfg,seed).cpu())
        save_tensor(torch.cat(rows),file)
    return torch.load(file,mmap=True,weights_only=True)


def evaluate(w,split,q,y,p,g,prob,banks,scale,configs,indices,base,kd,workers):
    scores,errors,tops={}, {},{}
    checks={}
    original=Replay(kd,g,split) if kd else None
    with ThreadPoolExecutor(max_workers=workers) as executor:
        def record(c,raw,labels):
            nm=key(c)
            quick,order,_,_=map_values(raw,labels.cuda(),g.cuda(),split)
            if original:
                value=original.score(order,labels,executor)
                error=float((value-quick.cpu()).abs().max())
                errors[nm]=max(errors.get(nm,0.),error)
                if error>2e-6: raise ValueError("Official KD metric mismatch")
                tops.setdefault(nm,[]).append(order[:,:25].cpu().int())
            else: value=quick.cpu()
            scores.setdefault(nm,[]).append(value)
        if original:
            for chunk in indices.split(144):
                record(("historical",),base[chunk].cuda(),y[chunk])
        for view,num in sorted(set((c[0],c[1]) for c in configs)):
            tree=GalleryGame(p,prob,banks[1],view,num)
            arith=tree.summarize("arith")
            here=[c for c in configs if c[:2]==(view,num)]
            for pool in sorted(set(c[2] for c in here)):
                summary=arith if pool=="arith" else tree.summarize(pool)
                coord=key((view,num,pool))
                checks[coord]={"bound_error":check_removed_photo_bounds(tree,summary,pool),
                               "leaf_min":int(tree.size[:tree.nleaf].min()),"leaf_max":int(tree.size[:tree.nleaf].max())}
                combinations=sorted(set(tuple(c[3:6]) for c in here if c[2]==pool))
                # Materialize each speaker separately. GPU memory depends on gallery
                # and vocabulary sizes, not the number of hyperparameter trials.
                for alpha,bound,contrast in combinations:
                    speaker=tree.speakers(summary,alpha,bound,contrast)
                    compare=[c for c in here if c[2:6]==(pool,alpha,bound,contrast)]
                    for chunk in indices.split(144):
                        l=emission(q[chunk],banks[0],scale)
                        base_raw=base[chunk].cuda()
                        mean=tree.mean_base(base_raw)
                        logbase=base_raw.clamp_min(1e-30).log()
                        path=tree.path_likelihood(l,speaker)
                        lit=tree.literal_path(l,arith)
                        diff=path-lit
                        if alpha==0 and pool=="arith":
                            err=float(diff.abs().max())
                            if err>4e-4: raise ValueError(f"Literal telescope failed: {err}")
                            checks[coord]["literal_telescope_max"]=max(checks[coord].get("literal_telescope_max",0),err)
                        for c in compare:
                            raw=tree.ranking(logbase,mean,diff,c[6],c[7])
                            record(c,raw,y[chunk])
                    del speaker
                if pool!="arith": del summary
            del tree,arith
            print(split,"gallery game",view,num,"finished",flush=True)
    return ({k:torch.cat(v) for k,v in scores.items()},errors,checks,{k:torch.cat(v) for k,v in tops.items()})


@torch.inference_mode()
def run(w,workers):
    root=w/"results"/FAMILY
    protocol={"revision":revision(),"primary":"deletion-robust pragmatic speaker","selection":"Sketchy mean official mAP then precision",
              "grid":grid(),"pilot":"24 queries/class, seed 31109, then top 3 full Sketchy candidates",
              "gallery_names":"Open Images public vocabulary","graph":"historical RGB diffusion"}
    atomic_json(root/"protocol.json",protocol|{"sha256":digest(protocol)})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA: raise ValueError("Changed evaluator")
    for stage in ("pilot","shortlist","final"):
        if stage=="pilot": configs=grid()
        elif stage=="shortlist":
            avg=read(root/"pilot.json")["means"]
            configs=choose(avg,"main",3)+choose(avg,"literal",2)
        else:
            avg=read(root/"shortlist.json")["means"]
            best=choose(avg,"main",1)[0]
            pure=choose(avg,"literal",1)[0]
            decision={"primary":best,"optimized_pool_only":pure,"revision":revision(),
                      "full_sketchy_sha256":checksum(root/"shortlist.json")}
            atomic_json(root/"freeze.json",decision|{"sha256":digest(decision)})
            configs=sorted({best,pure,(*best[:4],False,*best[5:]),(*best[:3],0,False,*best[5:]),
                            (*best[:5],"global",*best[6:]),(*best[:5],"unrelated",*best[6:]),
                            (*best[:6],1.,best[7]),(*best[:6],0.,best[7])})
        result={}
        for dataset in (("sketchy","tuberlin") if stage=="final" else ("sketchy",)):
            cfg,_,queries,_,galleries,qraw,praw=global_inputs(w,dataset)
            terms=read_words(cfg.vocabulary)
            if len(terms)!=20931: raise ValueError("Unexpected vocabulary")
            _,scale=bare_words(cfg,terms)
            banks=tuple(v.cuda() for v in load_words(cfg,terms))
            for split,(qi,y) in queries.items():
                pi,g=galleries[split]
                if (len(qi),len(pi),len(g.unique()))!=SIZES[split] or digest(UNSEEN_CLASSES[split])!=SPLIT_HASHES[split]:
                    raise ValueError("Incomplete split")
                indices=stratified_sample(y,24,31109) if stage=="pilot" else torch.arange(len(qi))
                q,p=qraw[qi].cuda(),praw[pi].cuda()
                prob=((p@banks[1].T)*scale).softmax(1)
                base=base_index(w,split,p,prob,q,banks[0],scale)
                per,errors,diag,tops=evaluate(w,split,q,y,p,g,prob,banks,scale,configs,indices,base,
                                              kd if stage=="final" else None,workers)
                result[split]={k:v.mean(0).tolist() for k,v in per.items()}
                top=max(result[split],key=result[split].get)
                print(stage,split,"winning",top,result[split][top],flush=True)
                if stage=="final":
                    details=root/(split+".pt")
                    save_tensor({"metrics":per,"query_labels":y,"top25":tops[key(best)]},details)
                    top_success=g[tops[key(best)].long()[:,0]]==y
                    was=g[tops[key(("historical",))].long()[:,0]]==y
                    rec={"protocol":{"split":split,"queries":len(qi),"gallery":len(pi),"classes":len(g.unique()),
                                     "checkpoint":CHECKPOINT,"kd_sha256":KD_FUNCTION_SHA,"vocabulary_sha256":checksum(cfg.vocabulary),
                                     "revision":revision(),"gpu":torch.cuda.get_device_name(),"map_k":200 if split=="sketchy_2" else 0,
                                     "p_k":200 if split=="sketchy_2" else 100},
                         "scores":{k:dict(zip(("mAP","precision"),v)) for k,v in result[split].items()},"primary":key(best),
                         "optimized_pool_only":key(pure),"literal_baseline":key(("historical",)),"checks":diag,
                         "vs_controls":{k:paired_delta(per[key(best)],v,y) for k,v in per.items()},
                         "top1":{"wrong_to_right":int((top_success&~was).sum()),"right_to_wrong":int((~top_success&was).sum())},
                         "original_kd_queries_per_condition":len(y),"kd_errors":errors,"per_query_sha256":checksum(details)}
                    atomic_json(root/(split+".json"),rec)
                del base,prob
        if stage!="final":
            means={k:[sum(result[s][k][i] for s in ("sketchy_1","sketchy_2"))/2 for i in range(2)]
                   for k in result["sketchy_1"]}
            atomic_json(root/(stage+".json"),{"splits":result,"means":means})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--workers",type=int,default=min(10,len(os.sched_getaffinity(0))))
    args=parser.parse_args()
    run(args.workspace,args.workers)
