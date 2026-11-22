"""Separate gallery-observable depiction information from indistinguishable word variation."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
from pathlib import Path
import time

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES, bare_words
from .cross_view_gallery import run as build_gallery
from .cross_view_region_cache import FAMILY as VIEWS, run as build_views
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .depiction_channel import GalleryChannel, algebra_checks, global_score, region_score, shuffle_photos
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .regional_intervention_study import index as compact_index
from .retrieval_diagnosis import global_inputs, stratified_sample
from .retrieval_evidence import load_kd, paired_delta
from .semantic import read_words


FAMILY = "gallery-visible-depiction"
PRIOR_RUN = "paired-prompt-head-intervention"
PAIRS = ((.35,0.),(.8,0.),(1.6,0.),(3.2,0.),(.8,-.4),(1.6,-.4),
         (1.6,.35),(3.2,.8),(.35,1.6),(0.,.8),(0.,-.8),(-.8,0.))
RAW = (-1.,-.5,-.2,.2,.5,1.,1.6,3.2)
READ = ("global","local","both")
KINDS = ("visible","scalar","shuffled")


def key(kind, a, b, support):
    return f"{kind}/{a:g}/{b:g}/{support}"


def cells():
    return ["regional", *(key(k,a,b,t) for k in KINDS for a,b in PAIRS for t in READ),
            *(key(k,a,a,t) for k in ("raw","raw_free") for a in RAW for t in READ)]


def parse(name):
    k,a,b,r = name.split("/")
    return k,float(a),float(b),r


def views(w,ds,paths):
    sig=digest({"model":CHECKPOINT,"selected":[str(p) for p in paths],
                "views":"13 corresponding two-pose regions, four native head blocks v1"})
    previous=w/"cache"/PRIOR_RUN/ds/("source-"+sig[:16]+".pt")
    if previous.is_file() and read(previous.with_suffix(".json"))["signature"]==sig:
        saved=torch.load(previous,weights_only=True,mmap=True)
        return lambda ix: saved["regions"][ix].cuda().transpose(1,2)
    build_views(w,ds)
    saved=torch.load(w/"cache"/VIEWS/ds/"views.pt",weights_only=True,mmap=True)
    return lambda ix: torch.stack((saved["original_regions"][ix],saved["regions"][ix]),1).cuda()


def get_split(w,ds,src,tgt,scale,inputs,s):
    qp,qs,pp,gals,raw,photo=inputs
    qi,y=qs[s]
    pi,g=gals[s]
    if (len(y),len(g),len(g.unique()))!=SIZES[s]:
        raise ValueError("Unexpected split dimensions")
    stamp=digest({"gallery":[str(pp[j]) for j in pi.tolist()],"model":CHECKPOINT,"source":"regional-64/RGB16"})
    words,small=compact_index(w,s,photo[pi].cuda(),tgt,scale,stamp[:16])
    full=torch.load(w/"cache/pose-conditional-layout"/(s+"-gallery-uniform.pt"),mmap=True,weights_only=True).cuda()
    hist=torch.load(w/"cache/pragmatic-gallery-competition"/(s+"-named-rgb16.pt"),mmap=True,weights_only=True)
    locked_path=w/"cache"/PRIOR_RUN/"full"/(s+".pt")
    locked=torch.load(locked_path,weights_only=True,mmap=True)["regional_score"] if locked_path.is_file() else None
    if small.size(0)!=len(g) or full.shape!=(len(g),len(src)):
        raise ValueError("Missing columns or rows in a target bank")
    return (qi,y,g,raw[qi],words,small,full,hist,locked)


def corrections(logits, contrast, actual, shuffled, requested):
    first = actual.analyze(logits,contrast)
    fake = shuffled.analyze(logits,contrast) if shuffled is not None else None
    yield None, first.statistics, first.checks
    for k,a,b in requested:
        if k=="raw_free":
            # This control also changes a crop's total likelihood.
            updated=logits+a*contrast
        elif k=="raw":
            updated=actual.adjusted(logits,a*first.centered)
        else:
            visible=fake.observable if k=="shuffled" else first.columns(k)
            delta=b*first.centered+(a-b)*visible
            updated=actual.adjusted(logits,delta)
        yield (k,a,b),updated,None


def eval_split(w,ds,s,cfg,inputs,src,tgt,scale,read_views,ids,lookup,reference,workers):
    qi,y,g,q,words,small,full,history,locked=get_split(w,ds,src,tgt,scale,inputs,s)
    dictionary=(GalleryChannel(full),GalleryChannel(small))
    parsed={parse(k) for k in lookup if k!="regional"}
    need_null=any(row[0]=="shuffled" for row in parsed)
    nulls=((GalleryChannel(shuffle_photos(full,719)),GalleryChannel(shuffle_photos(small,719)))
           if need_null else (None,None))
    params=sorted({(k,a,b) for k,a,b,_ in parsed})
    global_params=[p for p in params if (*p,"global") in parsed or (*p,"both") in parsed]
    local_params=[p for p in params if (*p,"local") in parsed or (*p,"both") in parsed]
    metrics,top,diag=defaultdict(list),defaultdict(list),[]
    errs={}
    worst=torch.zeros(4,device="cuda")
    evaluator=Replay(reference,g,s) if reference else None
    gallery_labels=g.cuda()
    with (ThreadPoolExecutor(max_workers=workers) if reference else nullcontext()) as pool:
        for batch_id,block in enumerate(ids.split(24)):
            batch=len(block)
            features=q[block].cuda()
            crops=read_views(qi[block]).contiguous().reshape(batch*26,512).float()
            a=features@src.T*scale
            d=features@(src-tgt).T*scale
            az=crops@src[words].T*scale
            dz=crops@(src[words]-tgt[words]).T*scale
            old_global=global_score(a,full)
            old_local=region_score(az,small,batch)
            seed=.65*history[block].cuda().clamp_min(1e-30).log()+.35*old_local
            if locked is not None:
                stored=locked[block].cuda()
                diff=stored-seed
                worst[0]=torch.maximum(worst[0],(diff-diff.mean(1,keepdim=True)).abs().amax())
                seed=stored
            else:
                hist=history[block].cuda().clamp_min(1e-30).log()
                diff=hist-old_global
                worst[0]=torch.maximum(worst[0],(diff-diff.mean(1,keepdim=True)).abs().amax())
            first=corrections(a,d,dictionary[0],nulls[0],global_params)
            second=corrections(az,dz,dictionary[1],nulls[1],local_params)
            _,gs,gcheck=next(first)
            _,ls,lcheck=next(second)
            worst[1:]=torch.maximum(worst[1:],torch.maximum(gcheck.amax(0),lcheck.amax(0)))
            diag.append(torch.cat((gs,ls.reshape(batch,26,-1).mean(1)),1).cpu())
            truth=y[block].cuda()
            def record(k,score):
                got,r,rel,_=map_values(score,truth,gallery_labels,s)
                got=got.cpu()
                if reference:
                    exact=evaluator.score(r,y[block],pool)
                    err=float((got-exact).abs().max())
                    if err>2e-6:
                        raise ValueError("KD-SBIR evaluator differs")
                    errs[k]=max(errs.get(k,0.),err)
                    got=exact
                metrics[k].append(got)
                top[k].append(rel[:,0].cpu())
            record("regional",seed)
            for p in params:
                gg=ll=None
                if p in global_params:
                    param,logits,_=next(first)
                    if p!=param:
                        raise ValueError("Global setting order changed")
                    gg=.65*(global_score(logits,full)-old_global)
                if p in local_params:
                    param,logits,_=next(second)
                    if p!=param:
                        raise ValueError("Regional setting order changed")
                    ll=.35*(region_score(logits,small,batch)-old_local)
                for mode in READ:
                    if (*p,mode) in parsed:
                        correction=gg if mode=="global" else (ll if mode=="local" else gg+ll)
                        record(key(*p,mode),seed+correction)
            if batch_id%16==0:
                print("KD" if reference else "sweep",s,batch_id*24,len(ids),worst.tolist(),flush=True)
    if worst[0]>.02 or worst[1:].max()>.01:
        raise ValueError(f"Regional reconstruction or Bayes identity changed {worst.tolist()}")
    return ({k:torch.cat(v) for k,v in metrics.items()},{k:torch.cat(v) for k,v in top.items()},
            torch.cat(diag),worst.tolist(),errs)


def choose(means):
    def best(predicate):
        return max((k for k in means if predicate(k)),key=means.get)
    main=best(lambda k:k.startswith("visible/"))
    _,a,b,where=parse(main)
    controls=[key(k,a,b,where) for k in ("scalar","shuffled")]
    return {"nominee":main,"best_raw":best(lambda k:k=="regional" or k.startswith(("raw/","raw_free/"))),
            "best_scalar":best(lambda k:k.startswith("scalar/")),"best_shuffled":best(lambda k:k.startswith("shuffled/")),
            "matched":controls,"ablation":[key("visible",a,b,r) for r in READ]}


def describe(metrics,top,diagnostics,chosen,y):
    main=chosen["nominee"]
    base=metrics["regional"]
    dt=metrics[main][:,0]-base[:,0]
    evidence={"global_observable_variance_fraction":float(diagnostics[:,0].mean()),
              "regional_observable_variance_fraction":float(diagnostics[:,3].mean()),
              "global_distinctive_shape_rms":float(diagnostics[:,1].mean()),
              "regional_distinctive_shape_rms":float(diagnostics[:,4].mean()),
              "kinds":["global observable fraction","global shape rms","global raw sd",
                       "mean regional observable fraction","mean regional shape rms","mean regional raw sd"],
              "top1_fixed":int((~top["regional"]&top[main]).sum()),"top1_broken":int((top["regional"]&~top[main]).sum()),
              "queries_AP_up":int((dt>1e-6).sum()),"queries_AP_down":int((dt< -1e-6).sum())}
    return evidence,{k:paired_delta(metrics[main],v,y) for k,v in metrics.items()}


@torch.inference_mode()
def run(w,workers):
    started=time.monotonic()
    invariant=algebra_checks(torch.device("cuda"))
    cfg,*inputs=global_inputs(w,"sketchy")
    names=read_words(cfg.vocabulary)
    if len(names)!=20931:
        raise ValueError("Changed public vocabulary")
    src,tgt=[v.cuda() for v in load_words(cfg,names)]
    _,scale=bare_words(cfg,names)
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Unexpected evaluator code")
    root=w/"results"/FAMILY
    lookup=cells()
    spec={"revision":revision(),"cells":lookup,"vocabulary":checksum(cfg.vocabulary),"checkpoint":CHECKPOINT,
          "selection":"equal-weight mAP, then precision","pilots_per_class":36,"seed":63241,"KD_sha":KD_FUNCTION_SHA}
    sig=digest(spec)
    if (root/"protocol.json").is_file() and read(root/"protocol.json")["sha256"]!=sig:
        raise ValueError("Changed pilot source")
    atomic_json(root/"protocol.json",spec|{"sha256":sig})
    build_gallery(w,"sketchy")
    read_views=views(w,"sketchy",inputs[0])
    torch.backends.cuda.matmul.allow_tf32=True
    captures={}
    samples={s:stratified_sample(y,36,63241) for s,(_,y) in inputs[1].items()}
    for s,ids in samples.items():
        dest=root/("pilot-"+s+".json")
        if dest.is_file():
            old=read(dest)
            if old["signature"]!=sig or old["per_query_sha256"]!=checksum(dest.with_suffix(".pt")):
                raise ValueError("Changed pilot artifacts")
            captures[s]=old["values"]
            continue
        m,t,d,checks,_=eval_split(w,"sketchy",s,cfg,inputs,src,tgt,scale,read_views,ids,lookup,None,workers)
        saved={"metrics":m,"top1_correct":t,"query_ids":ids,"query_labels":inputs[1][s][1][ids],"depiction":d}
        save_tensor(saved,dest.with_suffix(".pt"))
        values={k:v.mean(0).tolist() for k,v in m.items()}
        atomic_json(dest,{"signature":sig,"queries":len(ids),"gallery":SIZES[s][1],"values":values,
                          "null_and_likelihood_checks":invariant,"max_errors":checks,"per_query_sha256":checksum(dest.with_suffix(".pt"))})
        captures[s]=values
    means={k:[sum(r[k][i] for r in captures.values())/len(captures) for i in (0,1)] for k in lookup}
    selected=choose(means)
    audit=sorted(set(["regional",selected["nominee"],selected["best_raw"],selected["best_shuffled"],selected["best_scalar"],
                      *selected["matched"],*selected["ablation"]]))
    output,results,deltas,errs={},{},{},{}
    for s,ids in samples.items():
        metrics,top,d,_,err=eval_split(w,"sketchy",s,cfg,inputs,src,tgt,scale,read_views,ids,audit,kd,workers)
        original=torch.load(root/("pilot-"+s+".pt"),mmap=True,weights_only=True)
        bound=max(float((metrics[k]-original["metrics"][k]).abs().max()) for k in audit)
        if bound>2e-6:
            raise ValueError(f"Scoring replay changed {bound}")
        evidence,paired=describe(metrics,top,d,selected,original["query_labels"])
        results[s],deltas[s]=evidence,paired
        output[s]={k:v.mean(0).tolist() for k,v in metrics.items()}
        errs[s]={"original_queries_per_condition":len(ids),"max_error":max(err.values()),"replay_error":bound}
    atomic_json(root/"pilot.json",selected|{"means":means,"audited":output,"paired":deltas,"diagnosis":results,"KD":errs,
                        "elapsed_seconds":time.monotonic()-started,"GPU":torch.cuda.get_device_name()})
    print("selected",selected,"scores",means[selected["nominee"]],means["regional"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    run(args.workspace,args.workers)
