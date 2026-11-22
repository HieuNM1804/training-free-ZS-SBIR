"""Test minimum-KL lexical or gallery updates under alternative source-domain prompts."""

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
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .depiction_channel import global_score, region_score
from .depiction_posterior_bounds import FAMILY, MODES, invariants, rank_permutation, solve
from .encoding import save_tensor
from .foundation import load_words
from .gallery_depiction_study import READ, get_split, views
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs, stratified_sample
from .retrieval_evidence import load_kd, paired_delta
from .semantic import read_words


GAINS=(-1.,-.5,-.2,.2,.5,1.,2.)
LOCATIONS=("word","photo")


def key(location,mode,gain,view):
    return f"{location}/{mode}/{gain:g}/{view}"


def parse(name):
    a,b,c,d=name.split("/")
    return a,b,float(c),d


def grid():
    return ["regional",*(key(location,mode,gain,view) for location in LOCATIONS for mode in MODES for gain in GAINS for view in READ),
            *(key("word","legacy",gain,view) for gain in GAINS for view in READ)]


def word_methods(x,drawing,sketch,legacy,gain,log_mass,modes,perm,reader):
    original=x+log_mass
    p0=original.softmax(-1)
    pa=(original+gain*drawing).softmax(-1)
    pb=(original+gain*sketch).softmax(-1)
    candidates,diag,err=solve(p0,pa,pb,modes-{"legacy"},perm)
    z=original.logsumexp(-1,keepdim=True)
    ranks={k:reader(v.clamp_min(1e-35).log()+z-log_mass) for k,v in candidates.items()}
    if "legacy" in modes:
        ranks["legacy"]=reader(x+gain*legacy)
    return ranks,diag,err


def bank_on_device(w,cfg):
    file=w/"cache"/FAMILY/"public.pt"
    record=read(file.with_suffix(".json"))
    if record["vocabulary"]!=checksum(cfg.vocabulary) or record["checkpoint"]!=CHECKPOINT:
        raise ValueError("Changed crossed wording bank")
    data=torch.load(file,mmap=True,weights_only=True)
    return (data["drawing"].cuda()-data["photo"].cuda(),data["sketch"].cuda()-data["photo"].cuda(),record)


def evaluate(w,ds,s,cfg,inputs,src,tgt,scale,view,ids,rows,public,kd,workers):
    qi,y,g,q,words,small,large,hist,locked=get_split(w,ds,src,tgt,scale,inputs,s)
    target_mass=[v.sum(0,dtype=torch.float64).float().clamp_min(1e-30).log() for v in (large,small)]
    permutations=[rank_permutation(length,src.device) for length in (len(src),len(words),len(g))]
    desired={parse(k) for k in rows if k!="regional"}
    gains=sorted({k[2] for k in desired})
    metrics,correct,observations=defaultdict(list),defaultdict(list),defaultdict(list)
    numeric=torch.zeros(2,device=src.device)
    errors={}
    replay=Replay(kd,g,s) if kd else None
    truth=g.cuda()
    with (ThreadPoolExecutor(max_workers=workers) if kd else nullcontext()) as pool:
        for block,ix in enumerate(ids.split(32)):
            n=len(ix)
            z=q[ix].cuda()
            crops=view(qi[ix]).contiguous().reshape(n*26,512).float()
            global_log=z@src.T*scale
            local_log=crops@src[words].T*scale
            global_derivatives=[z@v.T*scale for v in (*public,src-tgt)]
            local_derivatives=[crops@v[words].T*scale for v in (*public,src-tgt)]
            og=global_score(global_log,large)
            ol=region_score(local_log,small,n)
            base=.65*hist[ix].cuda().clamp_min(1e-30).log()+.35*ol
            if locked is not None:
                frozen=locked[ix].cuda()
                diff=base-frozen
                numeric[0]=torch.maximum(numeric[0],(diff-diff.mean(1,keepdim=True)).abs().amax())
                base=frozen
            def record(name,score):
                val,rank,rel,_=map_values(score,y[ix].cuda(),truth,s)
                val=val.cpu()
                if kd:
                    actual=replay.score(rank,y[ix],pool)
                    err=(actual-val).abs().max().item()
                    errors[name]=max(err,errors.get(name,0.))
                    if err>2e-6:
                        raise ValueError("Changed original ranking evaluator")
                    val=actual
                metrics[name].append(val)
                correct[name].append(rel[:,0].cpu())
            record("regional",base)
            for gain in gains:
                at={(loc,m,v) for loc,m,c,v in desired if c==gain}
                needed={}
                for side,other in (("global","local"),("local","global")):
                    choices={mode for loc,mode,v in at if loc=="word" and v in (side,"both")}
                    if any(loc=="photo" and v in (side,"both") for loc,_,v in at):
                        choices|={"drawing","sketch"}
                    needed[side]=choices
                if not any(needed.values()):
                    continue
                gg,ll={},{}
                if needed["global"]:
                    gscore,diag,err=word_methods(global_log,*global_derivatives,gain,target_mass[0],needed["global"],
                                                permutations[0],lambda v:global_score(v,large))
                    gg={k:.65*(v-og) for k,v in gscore.items()}
                    observations[f"word/global/{gain:g}"].append(diag.cpu())
                    numeric[1]=torch.maximum(numeric[1],err)
                if needed["local"]:
                    lscore,diag,err=word_methods(local_log,*local_derivatives,gain,target_mass[1],needed["local"],
                                                permutations[1],lambda v:region_score(v,small,n))
                    ll={k:.35*(v-ol) for k,v in lscore.items()}
                    observations[f"word/local/{gain:g}"].append(diag.reshape(n,26,-1).mean(1).cpu())
                    numeric[1]=torch.maximum(numeric[1],err)
                for side in READ:
                    def add(mode):
                        if side=="global":return base+gg[mode]
                        if side=="local":return base+ll[mode]
                        return base+gg[mode]+ll[mode]
                    for mode in sorted(m for loc,m,v in at if loc=="word" and v==side):
                        record(key("word",mode,gain,side),add(mode))
                    modes={m for loc,m,v in at if loc=="photo" and v==side}
                    if modes:
                        pa,pb=[add(m).softmax(-1) for m in ("drawing","sketch")]
                        candidates,diag,err=solve(base.softmax(-1),pa,pb,modes,permutations[2])
                        observations[f"photo/{side}/{gain:g}"].append(diag.cpu())
                        numeric[1]=torch.maximum(numeric[1],err)
                        for mode,v in candidates.items():
                            record(key("photo",mode,gain,side),v.clamp_min(1e-35).log())
            if block%12==0:
                print("KD" if kd else "scan",s,block*32,len(ids),numeric.tolist(),flush=True)
    if numeric[0]>.02 or numeric[1]>.003 or set(metrics)!=set(rows):
        raise ValueError(f"Frozen reconstruction, bound checks or arm list changed {numeric.tolist()}")
    return ({k:torch.cat(v) for k,v in metrics.items()},{k:torch.cat(v) for k,v in correct.items()},
            {k:torch.cat(v) for k,v in observations.items()},numeric.tolist(),errors)


def choose(mean):
    def best(pred):
        return max((k for k in mean if pred(k)),key=mean.get)
    selected=best(lambda k:"/box/" in k)
    where,_,gain,side=parse(selected)
    compare={"nominee":selected,"best_simple":best(lambda k:k=="regional" or parse(k)[1] in ("mean","legacy","drawing","sketch")),
             "best_equal_KL":best(lambda k:"/matched_kl/" in k),"best_segment":best(lambda k:"/shared_mixture/" in k),
             "best_shuffled":best(lambda k:"/shuffled_width/" in k),"matched":[key(where,m,gain,side) for m in MODES],
             "placements":[key(loc,"box",gain,side) for loc in LOCATIONS],"supports":[key(where,"box",gain,s) for s in READ]}
    return compare


def subset(chosen):
    return sorted({"regional",chosen["nominee"],chosen["best_simple"],chosen["best_equal_KL"],chosen["best_segment"],
                   chosen["best_shuffled"],*chosen["matched"],*chosen["placements"],*chosen["supports"]})


@torch.inference_mode()
def run(w,workers):
    start=time.monotonic()
    cfg,*inputs=global_inputs(w,"sketchy")
    terms=read_words(cfg.vocabulary)
    if len(terms)!=20931:
        raise ValueError("Vocabulary changed")
    src,tgt=(v.cuda() for v in load_words(cfg,terms))
    _,scale=bare_words(cfg,terms)
    da,db,record=bank_on_device(w,cfg)
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("The original evaluator changed")
    cells=grid()
    spec={"revision":revision(),"checkpoint":CHECKPOINT,"paired_prompts":record["signature"],
          "vocabulary":checksum(cfg.vocabulary),"arms":cells,"seed":63241,"queries_per_class":36,"KD":KD_FUNCTION_SHA}
    sig=digest(spec)
    root=w/"results"/FAMILY
    if (root/"protocol.json").is_file() and read(root/"protocol.json")["sha256"]!=sig:
        raise ValueError("Changed study source")
    atomic_json(root/"protocol.json",spec|{"sha256":sig,"algebra":invariants(src.device)})
    build_gallery(w,"sketchy")
    view=views(w,"sketchy",inputs[0])
    torch.backends.cuda.matmul.allow_tf32=True
    samples={s:stratified_sample(y,36,63241) for s,(_,y) in inputs[1].items()}
    tables={}
    for s,ix in samples.items():
        file=root/("pilot-"+s+".json")
        if file.exists():
            last=read(file)
            if last["signature"]!=sig or last["per_query_sha256"]!=checksum(file.with_suffix(".pt")):
                raise ValueError("Stale pilot outcome")
            tables[s]=last["values"]
            continue
        metrics,top,evidence,numeric,_=evaluate(w,"sketchy",s,cfg,inputs,src,tgt,scale,view,ix,cells,(da,db),None,workers)
        save_tensor({"metrics":metrics,"query_ids":ix,"query_labels":inputs[1][s][1][ix],
                     "top1_correct":top,"diagnostics":evidence},file.with_suffix(".pt"))
        vals={k:v.mean(0).tolist() for k,v in metrics.items()}
        atomic_json(file,{"signature":sig,"values":vals,"queries":len(ix),"gallery":SIZES[s][1],"max_errors":numeric,
                          "KL_meanKL_TV_endpointTV_requiredTV":{k:v.mean(0).tolist() for k,v in evidence.items()},
                          "per_query_sha256":checksum(file.with_suffix(".pt"))})
        tables[s]=vals
    means={k:[sum(v[k][i] for v in tables.values())/2 for i in (0,1)] for k in cells}
    chosen=choose(means)
    aud=subset(chosen)
    verified,paired,errors,top={},{},{},{}
    for s,ix in samples.items():
        m,t,_,_,er=evaluate(w,"sketchy",s,cfg,inputs,src,tgt,scale,view,ix,aud,(da,db),kd,workers)
        cached=torch.load(root/("pilot-"+s+".pt"),mmap=True,weights_only=True)
        main=chosen["nominee"]
        maxerr=max(float((v-cached["metrics"][k]).abs().max()) for k,v in m.items())
        if maxerr>2e-6:
            raise ValueError(f"KD replay changed previously evaluated queries {maxerr}")
        verified[s]={k:v.mean(0).tolist() for k,v in m.items()}
        paired[s]={k:paired_delta(m[main],v,cached["query_labels"]) for k,v in m.items()}
        errors[s]={"original_queries_per_condition":len(ix),"max_error":max(er.values()),"replay_error":maxerr}
        top[s]={"fixed":int((~t["regional"]&t[main]).sum()),"broken":int((t["regional"]&~t[main]).sum())}
    atomic_json(root/"pilot.json",chosen|{"means":means,"audited":verified,"paired":paired,"KD":errors,"top1":top,
                                         "seconds":time.monotonic()-start,"GPU":torch.cuda.get_device_name()})
    print("chosen",chosen["nominee"],means[main],"regional",means["regional"],"simple",chosen["best_simple"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--workers",type=int,default=8)
    a=p.parse_args()
    run(a.workspace,a.workers)
