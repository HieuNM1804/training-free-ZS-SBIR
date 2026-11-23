"""Diagnose native and counterfactual attention in frozen candidate lists and replay the same full gallery ranks via KD-SBIR."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
import os
from pathlib import Path

import torch

from .attention_exchange_inputs import FAMILY,LAYERS
from .attention_exchange_pairs import K,RANDOM,SEEDS,KINDS
from .concept_benchmark import atomic_json,checksum,read
from .conditional_agreement_study import Replay
from .crossdomain_benchmark import digest,revision
from .crossdomain_inputs import CHECKPOINT
from .encoding import save_tensor
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd,paired_delta


STAGES=("L6","L9","L12","mean")
PREFIX=(64,192,384)
GAINS=(-.35,.12,.35,.8,2.)
PRIMARY="symmetric_frag"
CONTROLS=("symmetric_wrong_keys","symmetric_wrong_observer","symmetric_uniform_observer","euclidean_frag","cosine_frag")
FAMILY_GROUP=("symmetric_frag","symmetric","to_photo_frag","to_sketch_frag","fragment_gain","blank_gain")


def standardized(x):
    return (x-x.mean(-1,keepdim=True))/x.std(-1,keepdim=True,unbiased=False).clamp_min(1e-8)


def split_ap(good,split):
    mask=good[:,:200] if split=="sketchy_2" else good
    pos=torch.arange(1,mask.size(1)+1,device=mask.device)
    return (mask*(mask.cumsum(1)/pos)).sum(1),mask.sum(1)


def from_prefix(truth,base_num,base_den,old_top,k,split):
    limit=200 if split=="sketchy_2" else 100
    m=min(k,200) if split=="sketchy_2" else k
    new_num,new_den=split_ap(truth[:,:m],split)
    old_num,old_den=split_ap(old_top[:,:m],split)
    ap=(base_num+new_num-old_num)/(base_den+new_den-old_den).clamp_min(1)
    pbase=old_top[:,:limit].sum(1)
    pr=(pbase+truth[:,:limit].sum(1)-old_top[:,:min(k,limit)].sum(1))/limit
    return torch.stack((ap,pr),1)


def evidence_stats(values,truth,base_wrong):
    n=len(values)
    r=torch.arange(n,device=values.device)
    ok=truth.any(1)&(~truth).any(1)
    pos=truth.int().argmax(1);neg=(~truth).int().argmax(1)
    favored=values[r,pos]>values[r,neg]
    adj=truth[:,:-1]!=truth[:,1:];diff=(values[:,:-1]-values[:,1:])*(2*truth[:,:-1].float()-1)
    posn=truth.sum(1);negn=truth.size(1)-posn
    order=values.argsort(dim=1,descending=True,stable=True)
    ranked=truth.gather(1,order);under=((~ranked)*ranked.cumsum(1)).sum(1)
    return {"eligible":int(ok.sum()),"errors_with_positive":int((ok&base_wrong).sum()),
            "positive_beats_top_wrong_on_errors":int((favored&ok&base_wrong).sum()),"adjacent_pairs":int(adj.sum()),
            "adjacent_positive_score_wins":int(((diff>0)&adj).sum()),"mean_auc_on_eligible":float((under[ok]/(posn[ok]*negn[ok])).mean()) if ok.any() else None}


def key(seed,depth,kind,k,g):return f"{seed}/{depth}/{kind}/{k}/{g:g}"


def parse(name):
    seed,d,kind,k,g=name.split("/")
    return seed,d,kind,int(k),float(g)


def select(tables):
    a,b=tables
    def best(names):return max(sorted(names),key=lambda x:tuple((a[x][i]+b[x][i])/2 for i in (0,1)))
    allnames=[k for k in a if k.count("/")==4]
    cfg={};audit={"base/"+s for s in SEEDS}
    for s in SEEDS:
        names=[k for k in allnames if k.startswith(s+"/")]
        p=best([k for k in names if parse(k)[2]==PRIMARY and parse(k)[4]>0]);cfg[s+"/primary"]=p
        audit.add(p)
        for mode in (*CONTROLS,"symmetric","fragment_gain","gain_linear_part","gain_log_normalizer","gain_wrong_keys"):
            n=best([k for k in names if parse(k)[2]==mode and parse(k)[4]>0])
            cfg[s+"/"+mode]=n;audit.add(n)
        cfg[s+"/negative"]=best([k for k in names if parse(k)[2]==PRIMARY and parse(k)[4]<0]);audit.add(cfg[s+"/negative"])
        cfg[s+"/best_positive"]=best([k for k in names if parse(k)[4]>0]);audit.add(cfg[s+"/best_positive"])
        _,depth,_,cut,gain=parse(p)
        for m in (*CONTROLS,"symmetric","symmetric_empty","to_photo_frag","to_sketch_frag"):
            audit.add(key(s,depth,m,cut,gain))
    return cfg,sorted(audit)


def decode_scores(arr,st,seed,k,kind):
    si=SEEDS.index(seed);i=KINDS.index(kind)
    sl=slice(si*K,si*K+k)
    if st=="mean":
        return sum(standardized(x[:,sl,i]) for x in arr)/len(arr)
    return arr[STAGES.index(st)][:,sl,i]


@torch.inference_mode()
def run(w,stage):
    root=w/"results"/FAMILY;root.mkdir(parents=True,exist_ok=True)
    cache=w/"cache"/FAMILY/"panel"
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Wrong KD evaluator")
    official=stage=="pilot-official"
    if official:
        desc=read(root/"pilot-selection.json")
        if digest(desc["config"])!=desc["sha256"]:raise ValueError("Changed pilot selection")
        chosen=desc["config"]["audit"];sig=desc["sha256"]
    else:
        chosen=["base/"+s for s in SEEDS]+[key(s,d,c,k,g) for s in SEEDS for d in STAGES for c in KINDS for k in PREFIX for g in GAINS]
        spec={"code":revision(),"model":CHECKPOINT,"kd":KD_FUNCTION_SHA,"grid":chosen,"primary":PRIMARY,"sources":SEEDS,
              "sample":"36 queries per class with full original tail outside each reranked prefix"}
        sig=digest(spec);path=root/"protocol.json"
        if path.is_file() and read(path)["sha256"]!=sig:raise ValueError("Changed protocol")
        atomic_json(path,{"config":spec,"sha256":sig,"gpu":torch.cuda.get_device_name()})
    for s in ("sketchy_1","sketchy_2"):
        dest=root/(stage+"-"+s+".json")
        if dest.exists():
            prev=read(dest)
            if prev["signature"]!=sig or checksum(dest.with_suffix(".pt"))!=prev["per_query_sha256"]:raise ValueError("Changed pilot results")
            continue
        inp=torch.load(cache/("pilot-"+s+".pt"),mmap=True,weights_only=True)
        arr=[];checks={}
        for layer in LAYERS:
            stem=cache/("pilot-"+s+f"-L{layer}")
            meta=read(stem.with_suffix(".json"))
            if checksum(stem.with_suffix(".pt"))!=meta["sha256"]:raise ValueError("Changed attention evidence")
            arr.append(torch.load(stem.with_suffix(".pt"),mmap=True,weights_only=True).cuda())
            checks[str(layer)]={k:meta[k] for k in ("self_kl","decomposition","native_attention_cache_checks")}
        y=inp["labels"];yy=y.cuda();gt=inp["gl"].cuda();ix=inp["rows"]
        m,top,head={}, {}, {};stats={}
        replay=Replay(kd,inp["gl"],s) if official else None
        max_error=0.
        with (ThreadPoolExecutor(max_workers=min(10,len(os.sched_getaffinity(0)))) if official else nullcontext()) as pool:
            for z,seed in enumerate(SEEDS):
                order=inp["plans"][seed].cuda().long()
                truth=gt[order]==yy[:,None];old=truth[:,:K];wrong=~truth[:,0]
                num,den=split_ap(truth,s)
                pcut=200 if s=="sketchy_2" else 100
                raw=torch.stack((num/den.clamp_min(1),truth[:,:pcut].float().mean(1)),1)
                def store(name,v,good,first,prefix_order=None):
                    nonlocal max_error
                    if official:
                        exact=[]
                        for off in range(0,len(y),64):
                            full=order[off:off+64]
                            if prefix_order is not None:
                                full=torch.cat((prefix_order[off:off+64],full[:,prefix_order.size(1):]),1)
                            exact.append(replay.score(full,y[off:off+64],pool))
                        exact=torch.cat(exact);max_error=max(max_error,float((exact-v.cpu()).abs().amax()))
                        if max_error>2e-6:raise ValueError("Prefix accounting differs from original KD")
                        v=exact;head[name]=first.int().cpu()
                    m[name]=v.cpu();top[name]=good.cpu()
                store("base/"+seed,raw,~wrong,order[:,:5])
                relevant=[parse(tag) for tag in chosen if not tag.startswith("base/") and tag.startswith(seed+"/")]
                for d,c,k in sorted({(r[1],r[2],r[3]) for r in relevant}):
                    detail=decode_scores(arr,d,seed,k,c)
                    if not official and k==K:
                        stats[seed+"/"+d+"/"+c]=evidence_stats(detail,old,wrong)
                    x=standardized(detail);initial=standardized(inp["seedvals"][:,z,:k].cuda())
                    original=order[:,:k]
                    for gain in sorted({r[4] for r in relevant if r[1:4]==(d,c,k)}):
                        rank=(initial+gain*x).argsort(dim=1,descending=True,stable=True)
                        good=old[:,:k].gather(1,rank);reordered=original.gather(1,rank)
                        val=from_prefix(good,num,den,old,k,s)
                        store(key(seed,d,c,k,gain),val,good[:,0],reordered[:,:5],reordered)
                print(stage,s,seed,flush=True)
        if not official:
            randomgt=gt[inp["selected"][:,-RANDOM:].long().cuda()]==yy[:,None]
            for j,l in enumerate(LAYERS):
                for i,k in enumerate(KINDS):
                    stats[f"random/L{l}/{k}"]=evidence_stats(arr[j][:,-RANDOM:,i],randomgt,~randomgt[:,0])
        save_tensor({"ids":ix,"labels":y,"metrics":m,"top1":top,"first_five":head},dest.with_suffix(".pt"))
        report={"signature":sig,"code":revision(),"queries":len(ix),"gallery":len(gt),"classes":len(y.unique()),
                "values":{k:v.double().mean(0).tolist() for k,v in m.items()},"pair_diagnosis":stats,"checks":checks,
                "original_evaluator":official,"KD_max_error":max_error if official else None,"per_query_sha256":checksum(dest.with_suffix(".pt"))}
        if official:
            cl=y.unique(return_inverse=True)[1];p=desc["config"]["selected"]
            leads=[p[s0+"/"+x] for s0 in SEEDS for x in ("primary","best_positive")]
            report["paired"]={k:{other:paired_delta(m[k],m[other],cl) for other in chosen if other.startswith((k.split("/")[0]+"/","base/"))} for k in dict.fromkeys(leads)}
            report["top1"]={k:[int((top[k]&~top["base/"+k.split("/")[0]]).sum()),int((~top[k]&top["base/"+k.split("/")[0]]).sum())] for k in leads}
        atomic_json(dest,report);print("saved",stage,s,len(m),flush=True)
        del arr
    if not official:
        tables=[read(root/("pilot-"+s+".json"))["values"] for s in ("sketchy_1","sketchy_2")]
        p,audit=select(tables)
        cfg={"selected":p,"audit":audit,"protocol":sig,"selection":"equal mean pilot mAP then precision on the two unseen Sketchy splits"}
        atomic_json(root/"pilot-selection.json",{"config":cfg,"sha256":digest(cfg)})
        print("selected",p["regional/primary"],len(audit),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    ap.add_argument("--stage",choices=("pilot","pilot-official"),default="pilot")
    a=ap.parse_args();run(a.workspace,a.stage)
