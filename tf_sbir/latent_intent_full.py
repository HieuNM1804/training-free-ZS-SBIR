"""Lock the decision rule on complete Sketchy and replay its exact selected ranks on all three splits."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,checksum,read
from .conditional_agreement_study import Replay
from .crossdomain_benchmark import digest,revision
from .crossdomain_inputs import CHECKPOINT,runtime
from .encoding import save_tensor
from .foundation import load_words
from .intent_action_index import FAMILY
from .intent_decisions import RULES,action_values,plan
from .intent_utility import exposure
from .latent_intent_pilot import action_table,key
from .map_diagnosis import map_values
from .open_name_diagnosis import FAMILY as PUBLIC
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd,paired_delta
from .semantic import read_words
from .verification_inputs import get_inputs
from .word_conditional_verifier import zscore

ALPHAS=(.075,.12,.2,.35,.55,1.)
CAPS=(.12,.2,.35,.55,1.)
SIMPLE=(.025,.075,.12,.2,.35,.55)
SPLITS=("sketchy_1","sketchy_2")
DECISIONS=(*RULES,"substitute_first")


def parse(setting):
    n,t,temp,mix=setting.split("/")[:4]
    return int(n),float(t),float(temp),float(mix)


def rule(k):
    bits=k.split("/")
    return bits[-2] if len(bits)==6 else bits[0]


def best(means,methods,n):
    return sorted((k for k in means if rule(k) in methods),key=means.get,reverse=True)[:n]


def settings_from(pilot):
    selected=set()
    for mode in RULES:
        count=3 if mode=="joint" else 2
        for name in best(pilot,{mode},len(pilot)):
            selected.add((mode,parse(name)))
            if sum(m==mode for m,_ in selected)>=count:break
    nominees={spec for _,spec in selected}
    n,t,temp,_=parse(best(pilot,{"joint"},1)[0])
    return sorted(nominees|{(n,t,temp,.7),(n,t,temp,.95)})


def score_split(w,s,data,y,g,text,settings,rows_wanted,reference,primary,workers):
    book=torch.load(w/"cache"/FAMILY/(s+"-actions.pt"),weights_only=True,mmap=True)
    library={k:book[k].cuda() for k in ("pair","first")}
    library["memberships"]={k:v.cuda() for k,v in book["memberships"].items()}
    groups=book["groups"].cuda()
    photos=data["gallery_native"].cuda().float()
    g_gpu=g.cuda()
    source=torch.load(w/"cache"/PUBLIC/(s+"-proposals.pt"),weights_only=True,mmap=True)["whole"]
    values,diagnostics,error=defaultdict(list),defaultdict(list),{}
    wanted=set(rows_wanted)
    simple=[k for k in rows_wanted if rule(k) in ("first_name","averaged")]
    replay=Replay(reference,g,s) if reference else None
    with (ThreadPoolExecutor(max_workers=workers) if reference else nullcontext()) as pool:
        for start in range(0,len(y),40):
            sl=slice(start,start+40)
            truth=y[sl]
            gy=truth.cuda()
            base=data["regional_score"][sl].cuda()
            norm=zscore(base)
            query_words=tuple(x[sl].cuda() for x in source)
            models=plan(base,library,groups,query_words,settings)
            local=torch.arange(len(base),device=base.device)
            codes=torch.cat([(local[:,None]*len(text)+wds).flatten() for _,wds in models.values()]).unique(sorted=True)
            scored=action_table(codes,norm,photos,text,groups,gy,g_gpu,s,alphas=ALPHAS,retain_ranks=bool(reference))
            pair,first,metrics,hits=scored[:4]
            bmetric,brank,brel,_=map_values(base,gy,g_gpu,s)
            bexp=exposure(groups[brank],int(groups.max())+1)
            brank=brank.int()
            if reference:diagnostics["base_correct"].append(brel[:,0].cpu())
            def record(name,quick,order):
                if reference:
                    official=replay.score(order,truth,pool)
                    e=float((official-quick.cpu()).abs().max())
                    if e>2e-6:raise ValueError("Selected rank replay differs from the official KD evaluator")
                    error[name]=max(error.get(name,0.),e)
                    values[name].append(official)
                else:
                    values[name].append(quick.cpu())
            if "regional" in wanted:record("regional",bmetric,brank)
            for setting,(measure,nominees) in models.items():
                method_keys=[k for k in rows_wanted if rule(k) in DECISIONS and parse(k)==setting]
                caps={float(k.split("/")[-1]) for k in method_keys}
                lookup=torch.searchsorted(codes,(local[:,None]*len(text)+nominees).contiguous())
                chosen=action_values(measure,lookup,(pair,first),bexp,caps,alphas=ALPHAS)
                for name in method_keys:
                    mode=rule(name)
                    chosen_idx,utility_gain=chosen["joint" if mode=="substitute_first" else mode,float(name.split("/")[-1])]
                    adopt=chosen_idx>0
                    i=(chosen_idx-1).clamp_min(0)
                    word_slot=i//len(ALPHAS) if mode!="substitute_first" else torch.zeros_like(i)
                    strength=i%len(ALPHAS)
                    idx=lookup[local,word_slot]
                    fast=torch.where(adopt[:,None],metrics[idx,strength],bmetric)
                    rank=None
                    if reference:
                        rank=torch.where(adopt[:,None],scored[4][idx,strength],brank)
                        if name==primary:
                            diagnostics["top25"].append(rank[:,:25].cpu().int())
                            diagnostics["primary_correct"].append((g_gpu[rank[:,0].long()]==gy).cpu())
                        diagnostics["word/"+name].append(torch.where(adopt,nominees[local,word_slot],-1).cpu().int())
                        diagnostics["alpha/"+name].append(torch.where(adopt,base.new_tensor(ALPHAS)[strength],0).cpu())
                        diagnostics["predicted_gain/"+name].append(utility_gain.cpu())
                    record(name,fast,rank)
            qlog,qids=query_words
            for name in simple:
                kind,n,t,a=name.split("/")
                n,t,a=int(n),float(t),float(a)
                target=text[qids[:,:n]]
                if kind=="first_name":
                    x=zscore(target[:,0]@photos.T)
                else:
                    prob=(qlog[:,:n]*t).softmax(1)
                    x=zscore((target*prob[:,:,None]).sum(1)@photos.T)
                quick,rank,_,_=map_values((1-a)*norm+a*x,gy,g_gpu,s)
                record(name,quick,rank)
    return {k:torch.cat(v) for k,v in values.items()},{k:torch.cat(v) for k,v in diagnostics.items()},error


@torch.inference_mode()
def run(w,phase,workers):
    root=w/"results"/FAMILY
    pilot=read(root/"protocol.json")
    if not pilot["revision"].startswith("9fee0a0"):raise ValueError("Unexpected pilot source")
    rev=revision()
    all_settings=settings_from(read(root/"pilot.json")["means"])
    spec={"revision":rev,"caps":CAPS,"alphas":ALPHAS,"settings":all_settings,"pilot_sha256":checksum(root/"pilot.json"),
          "search":"all four policies share the same actions and can abstain"}
    sig=digest(spec)
    f=root/"full-protocol.json"
    if f.exists() and read(f)["sha256"]!=sig:raise ValueError("Changed full protocol")
    atomic_json(f,spec|{"sha256":sig})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Different evaluator")
    primary=None
    if phase=="lock":
        settings=all_settings
        lookup=["regional",*(key(st,m,cap) for st in settings for m in RULES for cap in CAPS),
                *(f"averaged/{n}/{t:g}/{a:g}" for n in (16,64) for t in (.3,1.) for a in SIMPLE),
                *(f"first_name/16/0.3/{a:g}" for a in SIMPLE)]
    else:
        lock=read(root/"freeze.json")
        if lock["revision"]!=rev or lock["full_sketchy_sha256"]!=checksum(root/"shortlist.json"):raise ValueError("Invalid freeze")
        primary=lock["primary"]
        st=parse(primary);cap=float(primary.split("/")[-1])
        lookup=sorted({primary,"regional",lock["simple"],lock["best_control"],lock["collapsed"],lock["first"],lock["incumbent"],
                       *(key(st,m,cap) for m in DECISIONS)})
        settings=sorted({parse(k) for k in lookup if rule(k) in DECISIONS})
    result={}
    for ds in (("sketchy","tuberlin") if phase=="audit" else ("sketchy",)):
        cfg=runtime(w,ds,batch=384,workers=6)
        words=read_words(cfg.vocabulary)
        if len(words)!=20931:raise ValueError("Unexpected vocabulary")
        target=load_words(cfg,words)[1].cuda()
        for s,(data,y,g) in get_inputs(w,ds).items():
            dest=root/((s if phase=="audit" else "shortlist-"+s)+".json")
            if dest.exists():
                old=read(dest)
                if old["signature"]!=sig or (phase=="audit" and old["per_query_sha256"]!=checksum(dest.with_suffix(".pt"))):
                    raise ValueError("Stale split")
                result[s]=old["values"];continue
            metrics,detail,err=score_split(w,s,data,y,g,target,settings,lookup,kd if phase=="audit" else None,primary,workers)
            means={k:v.mean(0).tolist() for k,v in metrics.items()}
            obj={"signature":sig,"queries":len(y),"gallery":len(g),"values":means}
            if phase=="audit":
                path=dest.with_suffix(".pt")
                save_tensor({"metrics":metrics,"query_labels":y}|detail,path)
                selected=detail["alpha/"+primary]>0
                delta=100*(metrics[primary]-metrics["regional"])
                obj.update({"protocol":{"split":s,"queries":len(y),"gallery":len(g),"classes":len(g.unique()),
                                        "checkpoint":CHECKPOINT,"kd_sha256":KD_FUNCTION_SHA,"revision":rev,"gpu":torch.cuda.get_device_name(),
                                        "vocabulary_sha256":checksum(cfg.vocabulary),"map_k":200 if s=="sketchy_2" else 0,
                                        "p_k":200 if s=="sketchy_2" else 100},
                            "primary":primary,"matched_base":lock["simple"],"best_contrast":lock["best_control"],"baseline":"regional",
                            "scores":{k:dict(zip(("mAP","precision"),v)) for k,v in means.items()},
                            "vs_controls":{k:paired_delta(metrics[primary],r,y) for k,r in metrics.items()},
                            "per_query_sha256":checksum(path),"original_kd_queries_per_condition":len(y),"kd_max_abs_error":err,
                            "action_analysis":{"adopted":int(selected.sum()),"mean_alpha":float(detail["alpha/"+primary].mean()),
                                               "adopted_actual_gain_pp":delta[selected].mean(0).tolist() if selected.any() else None,
                                               "unchanged_max_delta":float(delta[~selected].abs().amax()) if (~selected).any() else None},
                            "top1_changes":{"errors_fixed":int((~detail["base_correct"] & detail["primary_correct"]).sum()),
                                            "hits_broken":int((detail["base_correct"] & ~detail["primary_correct"]).sum())}})
            atomic_json(dest,obj)
            result[s]=means
            chosen=primary or best(means,{"joint"},1)[0]
            print(phase,s,chosen,means[chosen],"best",max(means,key=means.get),means[max(means,key=means.get)],flush=True)
    if phase=="lock":
        mean={k:[sum(result[s][k][i] for s in SPLITS)/2 for i in (0,1)] for k in lookup}
        atomic_json(root/"shortlist.json",{"splits":result,"means":mean})
        lock={"revision":rev,"full_sketchy_sha256":checksum(root/"shortlist.json"),"primary":best(mean,{"joint"},1)[0],
              "simple":best(mean,{"averaged","first_name","regional"},1)[0],
              "best_control":best(mean,{"collapsed","incumbent","first","averaged","first_name","regional"},1)[0]}
        lock.update({m:best(mean,{m},1)[0] for m in ("collapsed","incumbent","first")})
        atomic_json(root/"freeze.json",lock|{"sha256":digest(lock)})
        print("locked",lock,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--phase",choices=("lock","audit"),required=True)
    p.add_argument("--workers",type=int,default=10)
    args=p.parse_args()
    run(args.workspace,args.phase,args.workers)
