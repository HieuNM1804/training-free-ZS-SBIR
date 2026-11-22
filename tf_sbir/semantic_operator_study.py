"""Screen shared semantic operators with an exact KD-SBIR replay on the complete photo galleries."""

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
from .crossdomain_benchmark import digest, revision
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .response_geometry_views import FAMILY as PHOTO_VIEWS, read_photos, study_base
from .retrieval_diagnosis import stratified_sample
from .retrieval_evidence import load_kd, paired_delta
from .shared_semantic_operators import BLENDS, DIMENSIONS, FAMILY, GAINS, PENALTIES, algebra, make_folds


def unpack(row):
    k,name,s,a=row.split("/")
    return int(k),name,float(s),float(a)


def key(k,name,s,a):
    return f"{k}/{name}/{s:g}/{a:g}"


def grid():
    codes=["regional"]
    for k in DIMENSIONS:
        codes.extend(key(k,m,0,a) for m in ("raw","linear") for a in BLENDS)
        for s in PENALTIES:
            for m in ("shared","matched","rotated"):
                codes.extend(key(k,m,s,a) for a in BLENDS)
                codes.extend(key(k,m+"_gain",s,a) for a in GAINS)
            codes.extend(key(k,"reverse_gain",s,a) for a in GAINS)
    return codes


def choose(means):
    def best(modes):
        return max((k for k in means if k!="regional" and unpack(k)[1] in modes),key=means.get)
    main=best(("shared","shared_gain"))
    k,m,s,a=unpack(main)
    suffix="_gain" if m.endswith("_gain") else ""
    matched=[key(k,kind+suffix,s,a) for kind in ("shared","matched","rotated")]
    if suffix:matched.append(key(k,"reverse_gain",s,a))
    else:matched.extend(key(k,kind,0,a) for kind in ("raw","linear"))
    return {"nominee":main,"best_raw":best(("raw",)),"best_linear":best(("linear",)),
            "best_matched":best(("matched","matched_gain")),"best_rotated":best(("rotated","rotated_gain")),
            "best_reverse":best(("reverse_gain",)), "best_replacement":best(("shared",)),
            "best_gain":best(("shared_gain",)),"matched":matched}


def short(chosen):
    return sorted({"regional",*chosen["matched"],*(v for k,v in chosen.items() if k!="matched")})


def get_folds(w,ds,s,inputs,dims=DIMENSIONS):
    cfg,qp,qs,pp,gs,q,p,src,tgt,scale=inputs
    folder=w/"cache"/PHOTO_VIEWS/ds
    plan=torch.load(folder/"plan.pt",mmap=True,weights_only=True)
    pi,_=gs[s]
    # Every gallery photo is used. Its physical file ID, not its extraction priority, determines the matrix row.
    photos=read_photos(folder,plan,pi)
    native=p[pi].cuda()
    views=photos.to(device=native.device,dtype=torch.float32).mean(2)
    del photos
    feats=make_folds(views,native,tgt,scale,dims)
    return feats


def evaluate(s,ys,gl,base,folds,rows,kd=None,workers=2):
    evaluator=Replay(kd,gl,s) if kd else None
    truth=gl.cuda()
    grouped=defaultdict(list)
    for row in rows:
        if row!="regional":
            k,m,strength,weight=unpack(row)
            grouped[k].append((row,m,strength,weight))
    metrics,top,auxdiag=defaultdict(list),defaultdict(list),defaultdict(list)
    error={}
    with (ThreadPoolExecutor(max_workers=workers) if kd else nullcontext()) as pool:
        for begin in range(0,len(ys),64):
            sl=slice(begin,begin+64)
            original=base[sl].cuda()
            y=ys[sl]
            def record(row,a):
                calc,order,rel,_=map_values(a,y.cuda(),truth,s)
                exact=calc.cpu()
                if evaluator:
                    val=evaluator.score(order,y,pool)
                    err=float((val-exact).abs().max())
                    if err>2e-6:raise ValueError("KD metric mismatch")
                    error[row]=max(err,error.get(row,0.))
                    exact=val
                metrics[row].append(exact)
                top[row].append(rel[:,0].cpu())
            record("regional",original)
            for dim,table in grouped.items():
                strengths=sorted({t for _,m,t,_ in table if m not in ("raw","linear")})
                full={}
                for fold in folds[dim]:
                    pred=fold.read(original,strengths)
                    for k,v in pred.items():
                        if k not in full:full[k]=torch.empty_like(original)
                        full[k][:,fold.target_ids]=v
                for strength in strengths:
                    delta=full["shared",strength]-full["raw"]
                    other=full["rotated",strength]-full["raw"]
                    a=delta-delta.mean(1,keepdim=True)
                    b=other-other.mean(1,keepdim=True)
                    co=(a*b).mean(1)/(a.square().mean(1)*b.square().mean(1)).clamp_min(1e-20).sqrt()
                    auxdiag[f"{dim}/{strength:g}"].append(torch.stack((a.std(1),b.std(1),co),1).cpu())
                for name,m,t,a in table:
                    if m in ("raw","linear"):
                        score=(1-a)*original+a*full[m]
                    elif m.endswith("_gain"):
                        mode=m[:-5]
                        value=full["shared" if mode=="reverse" else mode,t]-full["raw"]
                        score=original+(-a if mode=="reverse" else a)*value
                    else:
                        score=(1-a)*original+a*full[m,t]
                    record(name,score)
            if begin%1024==0:print("KD" if kd else "scan",s,begin,len(ys),flush=True)
    return ({k:torch.cat(v) for k,v in metrics.items()},{k:torch.cat(v) for k,v in top.items()},
            {k:torch.cat(v) for k,v in auxdiag.items()},error)


@torch.inference_mode()
def run(w,workers):
    start=time.monotonic()
    root=w/"results"/FAMILY
    inputs=semantic_inputs(w,"sketchy")
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Different KD source")
    rows=grid()
    spec={"revision":revision(),"arms":rows,"vocab_sha":checksum(inputs[0].vocabulary),"KD":KD_FUNCTION_SHA,
          "queries_per_class":36,"pilot_seed":63241,"gallery_fold_seed":59731,"reference_names":64,"rotation_seed_base":39107,
          "selection":"equal mean of the two exploratory Sketchy mAPs then mean precision"}
    sig=digest(spec)
    if (root/"protocol.json").exists() and read(root/"protocol.json")["sha256"]!=sig:raise ValueError("Pilot differs")
    torch.backends.cuda.matmul.allow_tf32=False
    atomic_json(root/"protocol.json",spec|{"sha256":sig,"checks":algebra(torch.device("cuda"))})
    inputs_by_split={}
    table={}
    for s,(_,labels) in inputs[2].items():
        ids=stratified_sample(labels,36,63241)
        base=study_base(w,s,inputs,ids)
        gl=inputs[4][s][1]
        folds=get_folds(w,"sketchy",s,inputs)
        args=(s,labels[ids],gl,base,folds)
        inputs_by_split[s]=args
        f=root/("pilot-"+s+".json")
        if f.exists():
            prev=read(f)
            if prev["signature"]!=sig or checksum(f.with_suffix(".pt"))!=prev["per_query_sha256"]:
                raise ValueError("Invalid per-query pilot")
            table[s]=prev["values"]
            continue
        metrics,top,diag,_=evaluate(*args,rows)
        save_tensor({"query_ids":ids,"query_labels":labels[ids],"metrics":metrics,"top1_correct":top,"diagnostics":diag},
                    f.with_suffix(".pt"))
        values={r:v.mean(0).tolist() for r,v in metrics.items()}
        info={"signature":sig,"queries":len(ids),"gallery":len(gl),"values":values,
              "conditional_commutation_each_fold":{str(d):[fold.diagnostics for fold in fs] for d,fs in folds.items()},
              "correction_scale_rotated_scale_correlation":{k:v.mean(0).tolist() for k,v in diag.items()},
              "per_query_sha256":checksum(f.with_suffix(".pt"))}
        atomic_json(f,info)
        table[s]=values
    means={k:[sum(r[k][j] for r in table.values())/len(table) for j in (0,1)] for k in rows}
    selected=choose(means)
    audit=short(selected)
    actual,paired,errs,edits={},{},{},{}
    for s,args in inputs_by_split.items():
        m,top,_,errors=evaluate(*args,audit,kd,workers)
        z=torch.load(root/("pilot-"+s+".pt"),mmap=True,weights_only=True)
        err=max(float((v-z["metrics"][k]).abs().max()) for k,v in m.items())
        if err>2e-6:raise ValueError("Original replay differs from pilot")
        main=selected["nominee"]
        actual[s]={k:v.mean(0).tolist() for k,v in m.items()}
        paired[s]={k:paired_delta(m[main],v,args[1]) for k,v in m.items()}
        errs[s]={"max_abs_error":max(errors.values()),"replay_error":err,"original_queries_per_condition":len(args[1])}
        edits[s]={"corrected":int((~top["regional"]&top[main]).sum()),"broken":int((top["regional"]&~top[main]).sum())}
    atomic_json(root/"pilot.json",selected|{"means":means,"audited":actual,"paired":paired,"KD":errs,"top1":edits,
                "GPU":torch.cuda.get_device_name(),"seconds":time.monotonic()-start})
    print("selected",main,means[main],means["regional"],flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    cli.add_argument("--workers",type=int,default=2)
    args=cli.parse_args()
    run(args.workspace,args.workers)
