"""Evaluate exact multi-photo semantic cover probabilities against independent clue marginals."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .cluster_mine import cluster_assignments
from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .crossdomain_benchmark import digest, revision
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .response_geometry_views import study_base
from .retrieval_diagnosis import stratified_sample
from .retrieval_evidence import load_kd, paired_delta
from .semantic_witness_covers import DRAW_COUNTS, FAMILY, SIZES, THRESHOLDS, audit, group_counts, scores, word_margins


GROUPS=(24,64)
GAINS=(.1,.5,1.5)
KINDS=("cover","excess","marginal","one","mean","shuffled","shuffled_excess")


def key(g,k,t,r,m,a):
    return f"{g}/{k}/{t:g}/{r}/{m}/{a:g}"


def parse(tag):
    g,k,t,r,m,a=tag.split("/")
    return int(g),int(k),float(t),int(r),m,float(a)


def grid():
    return ["regional",*(key(g,k,t,r,m,a) for g in GROUPS for k in SIZES for t in THRESHOLDS
             for r in DRAW_COUNTS for m in KINDS if m not in ("one","mean") or r==1 for a in GAINS)]


def choose(means):
    def best(modes):
        return max((k for k in means if k!="regional" and parse(k)[4] in modes),key=means.get)
    main=best(("cover","excess"))
    g,k,t,r,m,a=parse(main)
    return {"nominee":main,"matched":[key(g,k,t,(1 if name in ("one","mean") else r),name,a) for name in KINDS],
            "best_simple":best(("marginal","one","mean")),"best_shuffled":best(("shuffled","shuffled_excess")),
            "best_only_dependence":best(("excess",))}


def shuffle_columns(mask,groups,k):
    n=len(groups)
    perm=torch.arange(n,device=groups.device)[:,None].expand(-1,k).clone()
    rng=torch.Generator(device="cpu").manual_seed(1201+k+int(groups.max()))
    for c in groups.unique():
        ix=torch.where(groups==c)[0]
        if len(ix)<2:continue
        offsets=torch.randint(1,len(ix),(k,),generator=rng).to(groups.device)
        local=(torch.arange(len(ix),device=groups.device)[:,None]+offsets)%len(ix)
        perm[ix]=ix[local]
    return mask.gather(1,perm[None].expand(len(mask),-1,-1))


def category_summary(terms,ids,y,incumbent):
    b=len(ids)
    wrong=incumbent!=y
    choice={}
    for tag,lists in terms.items():
        values=torch.cat(lists)
        top=values.argmax(1)
        row=torch.arange(b)
        diff=values[row,y]-values[row,incumbent]
        choice[tag]={"all_top1":float((top==y).float().mean()),
                     "wrong_baseline_queries":int(wrong.sum()),"error_top1":float((top[wrong]==y[wrong]).float().mean()),
                     "error_true_beats_original_rival":float(((diff[wrong]>0).float()+.5*(diff[wrong]==0)).mean())}
    return choice


def eval_split(w,s,inputs,ids,rows,base,kd,workers,oracle):
    _,_,qs,_,gals,raw,praw,src,tgt,scale=inputs
    qi,all_labels=qs[s]
    pi,gl=gals[s]
    photo=praw[pi].cuda()
    y=all_labels[ids]
    truth=gl.cuda()
    branches=defaultdict(list)
    for row in rows:
        if row!="regional":
            g,k,t,r,m,a=parse(row)
            branches[g,k,t].append((row,r,m,a))
    assignments={g:cluster_assignments(F.normalize(photo-photo.mean(0)),g) for g in sorted({key[0] for key in branches})}
    metrics,top,oracle_values=defaultdict(list),defaultdict(list),defaultdict(list)
    errors={}
    evaluator=Replay(kd,gl,s) if kd else None
    with (ThreadPoolExecutor(max_workers=workers) if kd else nullcontext()) as pool:
        for start in range(0,len(ids),32):
            sl=slice(start,start+32)
            z=base[sl].cuda()
            q=raw[qi[ids[sl]]].cuda()
            bits=word_margins(photo,q,src,tgt,max(SIZES),THRESHOLDS)
            def record(name,rank):
                val,order,rel,_=map_values(rank,y[sl].cuda(),truth,s)
                val=val.cpu()
                if evaluator:
                    exact=evaluator.score(order,y[sl],pool)
                    err=float((exact-val).abs().max())
                    if err>2e-6:raise ValueError("Original evaluator changed")
                    errors[name]=max(errors.get(name,0.),err)
                    val=exact
                metrics[name].append(val)
                top[name].append(rel[:,0].cpu())
            record("regional",z)
            scale_row=z.std(1,keepdim=True,unbiased=False)
            for (n,k,t),variants in branches.items():
                slots=assignments[n]
                mask=bits[t][...,:k]
                counts=group_counts(mask,slots,n)
                actual=scores(counts,k)
                perm=None
                if any(m.startswith("shuffled") for _,_,m,_ in variants):
                    shuffled=group_counts(shuffle_columns(mask,slots,k),slots,n)
                    perm=scores(shuffled,k)
                    if float((actual[2]["mean"]-perm[2]["mean"]).abs().max())>1e-8:
                        raise ValueError("Changed group-level clue marginals")
                for tag,r,m,a in variants:
                    if m.startswith("shuffled"):
                        v=perm[r]["cover" if m=="shuffled" else "excess"]
                    else:v=actual[r][m]
                    correction=v[:,slots].float()
                    c=(correction-correction.mean(1,keepdim=True))/correction.std(1,keepdim=True,unbiased=False).clamp_min(1e-7)
                    record(tag,z+a*scale_row*c)
            if oracle:
                for k in SIZES:
                    for t in THRESHOLDS:
                        counts=group_counts(bits[t][...,:k],truth,int(truth.max())+1)
                        values=scores(counts,k)
                        for r in DRAW_COUNTS:
                            for m in ("cover","excess","marginal","one","mean"):
                                oracle_values[f"{k}/{t:g}/{r}/{m}"].append(values[r][m].cpu())
            if start%320==0:print("official" if kd else "scan",s,start,len(ids),flush=True)
    oracle_report=category_summary(oracle_values,ids,y,gl[base.argmax(1)]) if oracle else {}
    return ({k:torch.cat(v) for k,v in metrics.items()},{k:torch.cat(v) for k,v in top.items()},
            {k:torch.cat(v) for k,v in oracle_values.items()},oracle_report,errors)


@torch.inference_mode()
def run(w,workers):
    root=w/"results"/FAMILY
    start=time.monotonic()
    inputs=semantic_inputs(w,"sketchy")
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Wrong evaluator")
    rows=grid()
    spec={"revision":revision(),"grid":rows,"checkpoint_vocab":checksum(inputs[0].vocabulary),"KD":KD_FUNCTION_SHA,
          "sample_seed":63241,"queries_per_class":36,"selection":"equal mAP of both unseen Sketchy pilot splits, then equal precision"}
    sig=digest(spec)
    if (root/"protocol.json").is_file() and read(root/"protocol.json")["sha256"]!=sig:raise ValueError("Different study")
    atomic_json(root/"protocol.json",spec|{"sha256":sig,"algebra":audit(inputs[7].device)})
    tables,used={},{}
    for s,(_,labels) in inputs[2].items():
        ids=stratified_sample(labels,36,63241)
        base=study_base(w,s,inputs,ids)
        used[s]=(ids,base)
        f=root/("pilot-"+s+".json")
        if f.is_file():
            old=read(f)
            if old["signature"]!=sig or checksum(f.with_suffix(".pt"))!=old["per_query_sha256"]:raise ValueError("Incomplete pilot")
            tables[s]=old["values"]
            continue
        met,top,classes,diagnosis,_=eval_split(w,s,inputs,ids,rows,base,None,workers,True)
        save_tensor({"query_ids":ids,"query_labels":labels[ids],"metrics":met,"top1":top,"retrospective_category_scores":classes},
                    f.with_suffix(".pt"))
        tables[s]={k:v.mean(0).tolist() for k,v in met.items()}
        atomic_json(f,{"signature":sig,"queries":len(ids),"gallery":len(inputs[4][s][1]),"values":tables[s],
                       "known_categories_diagnosis":diagnosis,"per_query_sha256":checksum(f.with_suffix(".pt"))})
    means={k:[sum(d[k][j] for d in tables.values())/len(tables) for j in (0,1)] for k in rows}
    lock=choose(means)
    checks=sorted({"regional",*lock["matched"],*(v for k,v in lock.items() if k!="matched")})
    official,paired,kd_checks,changes={},{},{},{}
    for s,(ids,base) in used.items():
        met,top,_,_,er=eval_split(w,s,inputs,ids,checks,base,kd,workers,False)
        old=torch.load(root/("pilot-"+s+".pt"),weights_only=True,mmap=True)
        mismatch=max(float((v-old["metrics"][k]).abs().max()) for k,v in met.items())
        if mismatch>2e-6:raise ValueError("Official order differs")
        main=lock["nominee"]
        official[s]={k:v.mean(0).tolist() for k,v in met.items()}
        paired[s]={k:paired_delta(met[main],v,old["query_labels"]) for k,v in met.items()}
        changes[s]={"fixed":int((~top["regional"]&top[main]).sum()),"broken":int((top["regional"]&~top[main]).sum())}
        kd_checks[s]={"queries":len(ids),"error":max(er.values()),"replay_error":mismatch}
    atomic_json(root/"pilot.json",lock|{"means":means,"audited":official,"paired":paired,"KD":kd_checks,"top1":changes,
                "seconds":time.monotonic()-start,"GPU":torch.cuda.get_device_name()})
    print("selected",main,means[main],means["regional"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--workers",type=int,default=2)
    a=p.parse_args()
    run(a.workspace,a.workers)
