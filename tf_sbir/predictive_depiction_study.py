"""Evaluate a predictive depiction distance against matched and separately selected controls."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .cross_view_regions import CandidateRegions
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .gallery_panel_resume import region_views, semantic_inputs
from .gallery_region_panels import FAMILY as PANEL, CONFIG, Index
from .map_diagnosis import map_values
from .predictive_depiction_distance import FAMILY, NAMES, RIDGES, GAINS, FIXED, ESTIMATES, algebra, estimates
from .quickdraw_locked import KD_FUNCTION_SHA
from .regional_intervention_study import index as compact_index
from .retrieval_diagnosis import stratified_sample
from .retrieval_evidence import load_kd, paired_delta
from .shared_concept_likelihood import emission


def tag(k,mode,ridge,gain):
    return f"{k}/{mode}/{ridge:g}/{gain:g}"


def parse(row):
    k,mode,ridge,gain=row.split("/")
    return int(k),mode,float(ridge),float(gain)


def grid():
    return ["regional",*(tag(k,m,r,g) for k in NAMES for m in (*FIXED,*ESTIMATES)
                         for r in ((0.,) if m in FIXED else RIDGES) for g in GAINS)]


def baseline(w,split,inputs,ids):
    cfg,qp,qs,pp,gs,q,photos,src,tgt,scale=inputs
    qi,y=qs[split]
    pi,g=gs[split]
    if digest(UNSEEN_CLASSES[split])!=SPLIT_HASHES[split]:
        raise ValueError("Changed evaluation categories")
    regional=w/"cache"/PANEL/(split+"-regional-reference.pt")
    if regional.is_file():
        return torch.load(regional,weights_only=True,mmap=True)[ids]
    prior=w/"cache/paired-prompt-head-intervention/full"/(split+".pt")
    if prior.is_file():
        return torch.load(prior,weights_only=True,mmap=True)["regional_score"][ids]
    dest=w/"cache"/FAMILY/(split+"-regional-"+digest(ids.tolist())[:16]+".pt")
    if dest.is_file():
        return torch.load(dest,weights_only=True,mmap=True)
    photo=photos[pi].cuda()
    qsel=qi[ids]
    pack,_=region_views(w,split,cfg,qp,qsel,q)
    fullhist=w/"cache/pragmatic-gallery-competition"/(split+"-named-rgb16.pt")
    if fullhist.exists():
        historical=torch.load(fullhist,weights_only=True,mmap=True)[ids]
    else:
        graph=Index(SimpleNamespace(p=photo),rgb_only=True)
        prob=torch.empty((len(photo),len(tgt)),device=photo.device)
        for start in range(0,len(photo),192):
            sub=photo[start:start+192]
            prob[start:start+len(sub)]=(scale*(sub@tgt.T)).softmax(1)
        historical=torch.cat([graph.score(CONFIG,emission(q[ix].cuda(),src,scale)@prob.T).cpu() for ix in qsel.split(96)])
        del graph,prob
    stamp=digest({"gallery":[str(pp[i]) for i in pi],"model":CHECKPOINT,"source":"regional-64/RGB16"})
    names,bank=compact_index(w,split,photo,tgt,scale,stamp[:16])
    ranks=[]
    for start in range(0,len(ids),48):
        sl=slice(start,start+48)
        reg=pack["regions"][sl].cuda()
        reader=CandidateRegions(q[qsel[sl]].cuda(),pack["flipped"][sl].cuda(),reg[:,:,0],reg[:,:,1],src,names,bank,
                                src.new_empty((0,len(src))),historical[sl].cuda())
        reader.prepare(scale)
        ranks.append((.65*reader.first+.35*reader.read("self",1.,"gallery")).cpu())
    complete=torch.cat(ranks)
    save_tensor(complete,dest)
    return complete


def candidate_axes(features,photos,src,tgt,scale):
    values=scale*(features@src.T)
    selected=values.topk(max(NAMES),1).indices
    x=values.gather(1,selected)
    delta=tgt[selected]-src[selected]
    u=scale*(features[:,None]*delta).sum(-1)
    b,k=selected.shape
    def read(bank):
        logits=scale*(photos@bank.reshape(b*k,-1).T)
        return logits.reshape(len(photos),b,k).permute(1,0,2).contiguous()
    return x[:,None],read(tgt[selected]),u[:,None],read(-delta)


def measure(split,inputs,ids,base,rows,kd,workers,block_size):
    _,_,qs,_,gs,raw,pr,src,tgt,scale=inputs
    qi,ys=qs[split]
    pi,labels=gs[split]
    y=ys[ids]
    photo=pr[pi].cuda()
    truth=labels.cuda()
    grouped=defaultdict(lambda: defaultdict(list))
    for name in rows:
        if name!="regional":
            k,mode,ridge,gain=parse(name)
            grouped[k,ridge][mode].append((name,gain))
    met,top,diagnostic=defaultdict(list),defaultdict(list),defaultdict(list)
    worst={}
    replay=Replay(kd,labels,split) if kd else None
    with (ThreadPoolExecutor(max_workers=workers) if kd else nullcontext()) as pool:
        for off in range(0,len(ids),block_size):
            sl=slice(off,off+block_size)
            q=raw[qi[ids[sl]]].cuda()
            start=base[sl].cuda()
            x,yphot,u,v=candidate_axes(q,photo,src,tgt,scale)
            target=y[sl].cuda()
            def score(name,val):
                got,order,rel,_=map_values(val,target,truth,split)
                actual=got.cpu()
                if kd:
                    official=replay.score(order,y[sl],pool)
                    err=float((actual-official).abs().max())
                    worst[name]=max(worst.get(name,0.),err)
                    if err>2e-6:
                        raise ValueError("Changed KD ranks or metrics")
                    actual=official
                met[name].append(actual)
                top[name].append(rel[:,0].cpu())
                return order[:,:200],rel[:,:200]
            ranked,hits=score("regional",start)
            boundary=hits[:,:-1]!=hits[:,1:]
            side=2*hits[:,:-1].float()-1
            for (count,ridge),modes in grouped.items():
                values=estimates(*(z[...,:count] for z in (x,yphot,u,v)),ridge,set(modes))
                for mode,e in values.items():
                    local=e.gather(1,ranked)
                    gap=(local[:,:-1]-local[:,1:])*side
                    diag=torch.stack((boundary.sum(1),((gap>0)&boundary).sum(1),((gap==0)&boundary).sum(1),
                                      (gap*boundary).sum(1)),1)
                    diagnostic[f"{count}/{mode}/{ridge:g}"].append(diag.cpu())
                    for name,gain in modes[mode]:
                        score(name,start+gain*e)
            if off%(block_size*20)==0:
                print("original KD" if kd else "scan",split,off,len(ids),flush=True)
    return ({k:torch.cat(v) for k,v in met.items()},{k:torch.cat(v) for k,v in top.items()},
            {k:torch.cat(v) for k,v in diagnostic.items()},worst)


def choose(means):
    def best(modes):
        return max((k for k in means if k!="regional" and parse(k)[1] in modes),key=means.get)
    main=best(("predict","gain"))
    count,_,ridge,gain=parse(main)
    return {"nominee":main,"best_native":best(FIXED),"best_inside":best(("inside","inside_gain")),
            "best_one_direction":best(("source","target")),"best_permuted":best(("wrong",)),
            "matched":[tag(count,m,(0. if m in FIXED else ridge),gain) for m in (*FIXED,*ESTIMATES)]}


@torch.inference_mode()
def run(w,workers,block_size):
    start=time.monotonic()
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Changed official evaluator")
    rows=grid()
    root=w/"results"/FAMILY
    inputs=semantic_inputs(w,"sketchy")
    spec={"revision":revision(),"evaluator":KD_FUNCTION_SHA,"checkpoint":CHECKPOINT,
          "public_vocabulary":checksum(inputs[0].vocabulary),"pilot_seed":63241,"per_class":36,
          "primary_modes":["predict","gain"],"rows":rows,"selection":"equal complete-gallery pilot mAP then precision"}
    sig=digest(spec)
    if (root/"protocol.json").exists() and read(root/"protocol.json")["sha256"]!=sig:
        raise ValueError("Pilot protocol changed")
    atomic_json(root/"protocol.json",spec|{"sha256":sig,"leave_one_out_refit_error":algebra(torch.device("cuda"))})
    samples={s:stratified_sample(ql,36,63241) for s,(_,ql) in inputs[2].items()}
    bases,tables={},{}
    for s,ids in samples.items():
        torch.backends.cuda.matmul.allow_tf32=False
        bases[s]=baseline(w,s,inputs,ids)
        file=root/("pilot-"+s+".json")
        if file.exists():
            old=read(file)
            if old["signature"]!=sig or checksum(file.with_suffix(".pt"))!=old["per_query_sha256"]:
                raise ValueError("Changed pilot observations")
            tables[s]=old["values"]
            continue
        met,top,hard,_=measure(s,inputs,ids,bases[s],rows,None,workers,block_size)
        save_tensor({"metrics":met,"top1":top,"hard_pair_counts":hard,"query_labels":inputs[2][s][1][ids],
                     "query_ids":ids},file.with_suffix(".pt"))
        vals={k:t.mean(0).tolist() for k,t in met.items()}
        h={}
        for k,t in hard.items():
            good=t[:,0]>0
            h[k]={"eligible_queries":int(good.sum()),"adjacent_mixed_pairs":int(t[:,0].sum()),
                  "fraction_favoring_relevant":float(((t[good,1]+.5*t[good,2])/t[good,0]).mean()),
                  "mean_relevance_advantage":float((t[good,3]/t[good,0]).mean())}
        atomic_json(file,{"signature":sig,"queries":len(ids),"gallery":len(inputs[4][s][1]),"values":vals,
                          "hard_adjacent_top200":h,"per_query_sha256":checksum(file.with_suffix(".pt"))})
        tables[s]=vals
    means={k:[sum(d[k][i] for d in tables.values())/len(tables) for i in (0,1)] for k in rows}
    lock=choose(means)
    audit=sorted({"regional",*lock["matched"],*(v for k,v in lock.items() if k!="matched")})
    verified,paired,errors,top1={},{},{},{}
    for s,ids in samples.items():
        met,top,_,err=measure(s,inputs,ids,bases[s],audit,kd,workers,block_size)
        cached=torch.load(root/("pilot-"+s+".pt"),mmap=True,weights_only=True)
        difference=max(float((v-cached["metrics"][k]).abs().max()) for k,v in met.items())
        if difference>2e-6:raise ValueError("Changed official pilot scores")
        verified[s]={k:v.mean(0).tolist() for k,v in met.items()}
        primary=lock["nominee"]
        paired[s]={k:paired_delta(met[primary],v,cached["query_labels"]) for k,v in met.items()}
        errors[s]={"original_queries_per_condition":len(ids),"max_error":max(err.values()),"replay_error":difference}
        top1[s]={"fixed":int((~top["regional"]&top[primary]).sum()),"broken":int((top["regional"]&~top[primary]).sum())}
    atomic_json(root/"pilot.json",lock|{"audited":verified,"paired":paired,"means":means,"KD":errors,"top1":top1,
                     "elapsed_seconds":time.monotonic()-start,"GPU":torch.cuda.get_device_name()})
    print("chosen",primary,means[primary],means["regional"],flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    cli.add_argument("--workers",type=int,default=2)
    cli.add_argument("--batch",type=int,default=12)
    args=cli.parse_args()
    run(args.workspace,args.workers,args.batch)
