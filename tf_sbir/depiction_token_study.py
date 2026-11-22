"""Evaluate whether the noun-carrying part of a depiction change repairs a frozen ranker."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
from pathlib import Path
import time

import open_clip
import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES, bare_words
from .cross_view_gallery import run as build_gallery
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .depiction_channel import global_score, region_score
from .depiction_token_paths import FAMILY, LAYERS
from .encoding import save_tensor
from .foundation import load_words
from .gallery_depiction_study import READ, get_split, views
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs, stratified_sample
from .retrieval_evidence import load_kd, paired_delta
from .semantic import read_words


PAIRS=((.5,0.),(1.5,0.),(4.,0.),(-1.5,0.),(-4.,0.),(0.,.5),(0.,-.5),
       (1.5,-.35),(-1.5,.35),(3.,.5),(-3.,-.5))
RAW=(-1.,-.5,-.2,.2,.5,1.,1.5,3.)
TYPES=("noun","collinear","exchanged","single")


def key(kind, layer, a, b, support):
    return f"{kind}/{layer}/{a:g}/{b:g}/{support}"


def parse(k):
    kind,l,a,b,m=k.split("/")
    return kind,int(l),float(a),float(b),m


def grid():
    return ["regional",*(key(k,l,a,b,m) for k in TYPES for l in LAYERS for a,b in PAIRS for m in READ),
            *(key(k,0,a,a,m) for k in ("raw","raw_free") for a in RAW for m in READ)]


def case_matched_shuffle(names):
    token=open_clip.get_tokenizer("ViT-B-32")
    lengths=torch.cat([token(names[i:i+512]).argmax(1)-1 for i in range(0,len(names),512)])
    g=torch.Generator().manual_seed(17889)
    perm=torch.arange(len(names))
    for n in lengths.unique():
        ids=(lengths==n).nonzero().flatten()
        order=ids[torch.randperm(len(ids),generator=g)]
        perm[order]=order.roll(1)
    return perm


def banks(w,names,source,target,needed):
    folder=w/"cache"/FAMILY
    desc=read(folder/"paired.json")
    if desc["checkpoint"]!=CHECKPOINT or desc["names"]!=len(names):
        raise ValueError("The paired public comparison index changed")
    pairs=torch.load(folder/"paired.pt",weights_only=True,mmap=True)
    perm=case_matched_shuffle(names).cuda()
    diff=source-target
    length=diff.norm(dim=1,keepdim=True).clamp_min(1e-10)
    out={}
    for l in sorted({l for kind,l in needed if l}):
        table=pairs[l]
        noun=table[:,0].cuda().float()
        for kind in TYPES:
            if (kind,l) not in needed:
                continue
            if kind=="noun":
                vec=noun
            elif kind=="collinear":
                vec=(noun.norm(dim=1,keepdim=True)/length)*diff
            elif kind=="exchanged":
                vec=F.normalize(noun[perm])*noun.norm(dim=1,keepdim=True)
            else:
                vec=table[:,2].cuda().float()
            out[kind,l]=vec
    return out,desc


def conserve(a,delta,log_mass):
    z=a+delta
    correction=torch.logsumexp(a+log_mass,1)-torch.logsumexp(z+log_mass,1)
    return z+correction[:,None]


def statistics(a,n,d,mass):
    pi=(a+mass).softmax(1)
    c=n-(pi*n).sum(1,keepdim=True)
    delta=d-(pi*d).sum(1,keepdim=True)
    x=(pi*c.square()).sum(1)
    y=(pi*delta.square()).sum(1).clamp_min(1e-9)
    cross=(pi*c*delta).sum(1)
    return torch.stack(((x/y).sqrt(),cross/(x.clamp_min(1e-9)*y).sqrt(),
                        ((x-cross.square()/y).clamp_min(0)/y).sqrt()),1)


def evaluate(w,ds,s,cfg,inputs,source,target,scale,read_views,ids,lookup,text_banks,kd,workers,
             reference_direction=None):
    qi,y,g,q,words,small,large,historical,locked=get_split(w,ds,source,target,scale,inputs,s)
    bank_g=large.sum(0,dtype=torch.float64).float().clamp_min(1e-30).log()
    bank_l=small.sum(0,dtype=torch.float64).float().clamp_min(1e-30).log()
    delta=source-target if reference_direction is None else reference_direction
    keys={parse(k) for k in lookup if k!="regional"}
    structure=defaultdict(set)
    for kind,l,a,b,support in keys:
        structure[kind,l].add((a,b))
    store,correct,measure=defaultdict(list),defaultdict(list),defaultdict(list)
    errors={}
    g_gpu=g.cuda()
    replay=Replay(kd,g,s) if kd else None
    match=0.
    with (ThreadPoolExecutor(max_workers=workers) if kd else nullcontext()) as pool:
        for offset,ix in enumerate(ids.split(24)):
            n=len(ix)
            native=q[ix].cuda()
            crop=read_views(qi[ix]).contiguous().reshape(n*26,512).float()
            global_log=native@source.T*scale
            regional_log=crop@source[words].T*scale
            gd=native@delta.T*scale
            ld=crop@delta[words].T*scale
            orig_g=global_score(global_log,large)
            orig_l=region_score(regional_log,small,n)
            base=.65*historical[ix].cuda().clamp_min(1e-30).log()+.35*orig_l
            if locked is not None:
                previous=locked[ix].cuda()
                err=previous-base
                match=max(match,float((err-err.mean(1,keepdim=True)).abs().amax()))
                base=previous
            def record(k,scores):
                v,order,t,_=map_values(scores,y[ix].cuda(),g_gpu,s)
                v=v.cpu()
                if kd:
                    actual=replay.score(order,y[ix],pool)
                    err=float((actual-v).abs().max())
                    if err>2e-6:
                        raise ValueError("Original KD-SBIR disagreed")
                    v=actual
                    errors[k]=max(err,errors.get(k,0.))
                store[k].append(v)
                correct[k].append(t[:,0].cpu())
            record("regional",base)
            for kind,l in sorted(structure):
                is_raw=kind in ("raw","raw_free")
                if is_raw:
                    noun,local=gd,ld
                else:
                    vectors=text_banks[kind,l]
                    noun=native@vectors.T*scale
                    local=crop@vectors[words].T*scale
                if kind in ("noun","mass"):
                    diag=torch.cat((statistics(global_log,noun,gd,bank_g),
                                    statistics(regional_log,local,ld,bank_l).reshape(n,26,-1).mean(1)),1)
                    measure[str(l)].append(diag.cpu())
                for a,b in sorted(structure[kind,l]):
                    global_delta=local_delta=None
                    prefix=(kind,l,a,b)
                    if (*prefix,"global") in keys or (*prefix,"both") in keys:
                        shift=b*gd+(a-b)*noun
                        after=global_log+shift if kind=="raw_free" else conserve(global_log,shift,bank_g)
                        global_delta=.65*(global_score(after,large)-orig_g)
                    if (*prefix,"local") in keys or (*prefix,"both") in keys:
                        shift=b*ld+(a-b)*local
                        after=regional_log+shift if kind=="raw_free" else conserve(regional_log,shift,bank_l)
                        local_delta=.35*(region_score(after,small,n)-orig_l)
                    for m in READ:
                        if (*prefix,m) in keys:
                            diff=global_delta if m=="global" else (local_delta if m=="local" else global_delta+local_delta)
                            record(key(*prefix,m),base+diff)
            if offset%16==0:
                print("original KD" if kd else "search",s,offset*24,len(ids),len(lookup),match,flush=True)
    if match>.02 or set(store)!=set(lookup):
        raise ValueError("Reference or candidate set changed")
    return ({k:torch.cat(v) for k,v in store.items()},{k:torch.cat(v) for k,v in correct.items()},
            {k:torch.cat(v) for k,v in measure.items()},errors,match)


def choose(means):
    def best(pred):
        return max((k for k in means if pred(k)),key=means.get)
    main=best(lambda k:k.startswith("noun/"))
    _,l,a,b,support=parse(main)
    return {"nominee":main,"best_raw":best(lambda k:k=="regional" or k.startswith(("raw/","raw_free/"))),
            "best_collinear":best(lambda k:k.startswith("collinear/")),
            "best_exchanged":best(lambda k:k.startswith("exchanged/")),"best_single":best(lambda k:k.startswith("single/")),
            "matched":[key(k,l,a,b,support) for k in TYPES if k!="noun"],
            "ablation":[key("noun",l,a,b,m) for m in READ]}


@torch.inference_mode()
def run(w,workers):
    start=time.monotonic()
    cfg,*inputs=global_inputs(w,"sketchy")
    words=read_words(cfg.vocabulary)
    source,target=(x.cuda() for x in load_words(cfg,words))
    _,scale=bare_words(cfg,words)
    lookup=grid()
    vec,cache_info=banks(w,words,source,target,{parse(k)[:2] for k in lookup if k!="regional"})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Different original evaluator")
    root=w/"results"/FAMILY
    spec={"revision":revision(),"checkpoint":CHECKPOINT,"vocabulary":checksum(cfg.vocabulary),
          "intervention_signature":cache_info["signature"],"layers":LAYERS,"pairs":PAIRS,"raw":RAW,"modes":TYPES,
          "support":READ,"seed":63241,"per_class":36,"evaluator":KD_FUNCTION_SHA}
    sig=digest(spec)
    if (root/"protocol.json").is_file() and read(root/"protocol.json")["sha256"]!=sig:
        raise ValueError("Pilot source changed")
    atomic_json(root/"protocol.json",spec|{"sha256":sig,"text_checks":cache_info["checks"],
                      "text_norms":cache_info["per_layer_mean_norm_noun_rest_interaction"]})
    build_gallery(w,"sketchy")
    getviews=views(w,"sketchy",inputs[0])
    torch.backends.cuda.matmul.allow_tf32=True
    rows={}
    ids={s:stratified_sample(y,36,63241) for s,(_,y) in inputs[1].items()}
    for s,ix in ids.items():
        path=root/("pilot-"+s+".json")
        if path.is_file():
            record=read(path)
            if record["signature"]!=sig or record["per_query_sha256"]!=checksum(path.with_suffix(".pt")):
                raise ValueError("Stale split")
            rows[s]=record["values"]
            continue
        metrics,top,measures,_,err=evaluate(w,"sketchy",s,cfg,inputs,source,target,scale,getviews,ix,lookup,vec,None,workers)
        m={k:v.mean(0).tolist() for k,v in metrics.items()}
        save_tensor({"metrics":metrics,"query_labels":inputs[1][s][1][ix],"query_ids":ix,
                     "top1_correct":top,"intervention_by_layer":measures},path.with_suffix(".pt"))
        atomic_json(path,{"signature":sig,"queries":len(ix),"gallery":SIZES[s][1],"values":m,
                          "mean_rms_correlation_residual_per_layer":{k:v.mean(0).tolist() for k,v in measures.items()},
                          "baseline_reconstruction":err,"per_query_sha256":checksum(path.with_suffix(".pt"))})
        rows[s]=m
    means={k:[sum(t[k][j] for t in rows.values())/len(rows) for j in (0,1)] for k in lookup}
    selected=choose(means)
    audit=sorted(set(["regional",selected["nominee"],selected["best_raw"],selected["best_collinear"],
                      selected["best_exchanged"],selected["best_single"],*selected["ablation"],*selected["matched"]]))
    actual,paired,checks,detail={},{},{},{}
    for s,ix in ids.items():
        m,t,_,errs,_=evaluate(w,"sketchy",s,cfg,inputs,source,target,scale,getviews,ix,audit,vec,kd,workers)
        stored=torch.load(root/("pilot-"+s+".pt"),mmap=True,weights_only=True)
        bound=max(float((stored["metrics"][k]-m[k]).abs().max()) for k in audit)
        if bound>2e-6:
            raise ValueError(f"Candidate replay changed {bound}")
        main=selected["nominee"]
        paired[s]={k:paired_delta(m[main],v,stored["query_labels"]) for k,v in m.items()}
        checks[s]={"original_KD_queries_per_condition":len(ix),"max_error":max(errs.values()),"replay_error":bound}
        actual[s]={k:v.mean(0).tolist() for k,v in m.items()}
        detail[s]={"top1_fixed":int((~t["regional"]&t[main]).sum()),"top1_broken":int((t["regional"]&~t[main]).sum())}
    atomic_json(root/"pilot.json",selected|{"means":means,"audited":actual,"paired":paired,"KD":checks,"top1":detail,
                       "device":torch.cuda.get_device_name(),"elapsed_seconds":time.monotonic()-start})
    print("selected",selected,"primary",means[main],"reference",means["regional"],flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    cli.add_argument("--workers",type=int,default=8)
    args=cli.parse_args()
    run(args.workspace,args.workers)
