"""Screen regional dependence on the original galleries, reranking the same 512 candidates for all arms."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from pathlib import Path
import time

import torch

from .concept_benchmark import atomic_json, checksum, read
from .concept_response_geometry import ARMS, SIZES, checks, descriptors, match
from .conditional_agreement_study import Replay
from .crossdomain_benchmark import digest, revision
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .response_geometry_views import FAMILY, inputs_plan, query_pack, read_photos, study_base
from .retrieval_evidence import load_kd, paired_delta


DEPTH=512
WEIGHTS=(.06,.2,.6,1.5)
METHODS=("interaction","cosine")


def key(size,mode,weight):
    return f"{size}/{mode}/{weight:g}"


def parse(name):
    size,mode,weight=name.split("/")
    return int(size),mode,float(weight)


def evidence(w,s,inputs,folder,plan):
    _,_,qs,_,gs,q,p,src,tgt,scale=inputs
    ids=plan["splits"][s]["ids"]
    qi,y=qs[s]
    pi,g=gs[s]
    base=study_base(w,s,inputs,ids)
    entries=base.argsort(dim=1,descending=True,stable=True)[:,:DEPTH].contiguous()
    dest=folder/("shortlist-evidence-"+s+".pt")
    source=digest({"revision":revision(),"local_ids":ids.tolist(),"shape":[len(pi),len(qi)],"depth":DEPTH})
    if dest.exists() and read(dest.with_suffix(".json"))["signature"]==source:
        saved=torch.load(dest,mmap=True,weights_only=True)
        if not torch.equal(saved["shortlist"],entries):raise ValueError("Shortlist changed")
        return base,entries,saved["aux"]
    actual=pi[entries]
    needed=actual.unique().sort().values
    positions=torch.searchsorted(needed,actual)
    photos=read_photos(folder,plan,needed)
    source_views=query_pack(w,s,inputs,ids)
    aux=torch.empty((len(ids),len(SIZES),len(ARMS),DEPTH),dtype=torch.float32)
    for begin in range(0,len(ids),4):
        sl=slice(begin,begin+4)
        n=len(ids[sl])
        native=q[qi[ids[sl]]].cuda()
        qv=source_views[sl].cuda()
        words=(scale*(native@src.T)).topk(max(SIZES),1).indices
        signatures={size:descriptors(qv,native,src[words[:,:size]],words[:,:size],"source") for size in SIZES}
        for col in range(0,DEPTH,96):
            pc=slice(col,col+96)
            block=positions[sl,pc]
            count=block.shape[1]
            subviews=photos[block.flatten()].cuda()
            subnative=p[needed[block.flatten()]].cuda()
            for i,size in enumerate(SIZES):
                wids=words[:,:size].repeat_interleave(count,0)
                ps=descriptors(subviews,subnative,tgt[wids],wids,"target")
                qsigs={k:v.repeat_interleave(count,0) for k,v in signatures[size].items()}
                vals=match(qsigs,ps)
                aux[sl,i,:,pc]=torch.stack([vals[k].view(n,count) for k in ARMS],1).cpu()
        if begin%160==0:
            print("paired responses",s,begin,len(ids),flush=True)
    save_tensor({"aux":aux,"shortlist":entries},dest)
    atomic_json(dest.with_suffix(".json"),{"signature":source,"sizes":SIZES,"arms":ARMS})
    return base,entries,aux


def compute(rows,base,shortlist,aux,y,g,s,kd=None,workers=2):
    replay=Replay(kd,g,s) if kd else None
    truth=g.cuda()
    all_positions=torch.arange(len(g),device=truth.device,dtype=torch.float32)
    out=defaultdict(list)
    best=defaultdict(list)
    max_errors={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for off in range(0,len(y),64):
            sl=slice(off,off+64)
            frozen=base[sl].cuda()
            native=frozen.argsort(dim=1,descending=True,stable=True)
            first=shortlist[sl].cuda()
            if not torch.equal(native[:,:DEPTH],first):raise ValueError("Incumbent ranks have changed")
            original=frozen.gather(1,first)
            gaps=original.std(1,keepdim=True,unbiased=False)
            v=aux[sl].cuda()
            mu=v.mean(-1,keepdim=True)
            v=(v-mu)/v.std(-1,keepdim=True,unbiased=False).clamp_min(1e-7)
            native_ranks=torch.empty_like(frozen).scatter_(1,native,-all_positions.expand_as(frozen))
            for row in rows:
                if row=="regional":score=native_ranks
                else:
                    k,m,a=parse(row)
                    order=(original+a*gaps*v[:,SIZES.index(k),ARMS.index(m)]).argsort(dim=1,descending=True,stable=True)
                    changed=first.gather(1,order)
                    score=native_ranks.clone().scatter_(1,changed,-all_positions[:DEPTH].expand_as(changed))
                got,rank,rel,_=map_values(score,y[sl].cuda(),truth,s)
                val=got.cpu()
                if replay:
                    exact=replay.score(rank,y[sl],pool)
                    err=float((exact-val).abs().max())
                    max_errors[row]=max(err,max_errors.get(row,0.))
                    if err>2e-6:raise ValueError("Official retrieval metric changed")
                    val=exact
                out[row].append(val)
                best[row].append(rel[:,0].cpu())
    return {k:torch.cat(v) for k,v in out.items()},{k:torch.cat(v) for k,v in best.items()},max_errors


def choose(means):
    def best(modes):
        return max((k for k in means if k!="regional" and parse(k)[1] in modes),key=means.get)
    main=best(METHODS)
    k,_,w=parse(main)
    return {"nominee":main,"best_marginal":best(("marginals","mean","dispersion")),
            "best_ordinary":best(("raw_tau","pearson","without_reference","full_residual")),
            "best_permuted":best(("permuted","permuted_cosine")),"best_additive":best(("additive","word_geometry")),
            "best_raw_input":best(("without_projection",)), "matched":[key(k,m,w) for m in ARMS]}


@torch.inference_mode()
def run(w,workers):
    start=time.monotonic()
    inputs=semantic_inputs(w,"sketchy")
    folder,plan=inputs_plan(w,"sketchy",inputs)
    root=w/"results"/FAMILY
    rows=["regional",*(key(k,m,a) for k in SIZES for m in ARMS for a in WEIGHTS)]
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("KD mismatch")
    exp={"revision":revision(),"depth":DEPTH,"controls":rows,"seed":63241,"queries_per_class":36,
         "scoring":"within top 512, scale each term to baseline score SD per query; all remaining positions retained",
         "vocabulary":checksum(inputs[0].vocabulary),"KD":KD_FUNCTION_SHA,"selection":"equal exploratory Sketchy mAP then precision"}
    sig=digest(exp)
    if (root/"protocol.json").exists() and read(root/"protocol.json")["sha256"]!=sig:raise ValueError("Different study protocol")
    atomic_json(root/"protocol.json",exp|{"sha256":sig,"math_checks":checks(inputs[7].device)})
    packed={}
    values={}
    for s,(_,labels) in inputs[2].items():
        ids=plan["splits"][s]["ids"]
        base,indices,aux=evidence(w,s,inputs,folder,plan)
        data=(base,indices,aux,labels[ids],inputs[4][s][1],s)
        packed[s]=data
        f=root/("pilot-"+s+".json")
        if f.exists():
            old=read(f)
            if old["signature"]!=sig or checksum(f.with_suffix(".pt"))!=old["per_query_sha256"]:raise ValueError("Incomplete saved pilot")
            values[s]=old["values"]
            continue
        m,t,_=compute(rows,*data)
        save_tensor({"query_ids":ids,"query_labels":labels[ids],"metrics":m,"top1_correct":t},f.with_suffix(".pt"))
        values[s]={k:v.mean(0).tolist() for k,v in m.items()}
        atomic_json(f,{"signature":sig,"queries":len(ids),"gallery":len(data[4]),"values":values[s],"per_query_sha256":checksum(f.with_suffix(".pt"))})
    means={k:[sum(d[k][i] for d in values.values())/2 for i in (0,1)] for k in rows}
    selected=choose(means)
    audit=sorted({"regional",*selected["matched"],*(v for k,v in selected.items() if k!="matched")})
    verified,paired,errors,changes={},{},{},{}
    for s,args in packed.items():
        actual,top,err=compute(audit,*args,kd,workers)
        compare=torch.load(root/("pilot-"+s+".pt"),mmap=True,weights_only=True)
        delta=max(float((v-compare["metrics"][k]).abs().max()) for k,v in actual.items())
        if delta>2e-6:raise ValueError("Original and fast metrics disagree")
        main=selected["nominee"]
        paired[s]={k:paired_delta(actual[main],v,args[3]) for k,v in actual.items()}
        errors[s]={"queries":len(args[3]),"max_error":max(err.values()),"cached_replay_error":delta}
        changes[s]={"corrected":int((~top["regional"]&top[main]).sum()),"broken":int((top["regional"]&~top[main]).sum())}
        verified[s]={k:v.mean(0).tolist() for k,v in actual.items()}
    atomic_json(root/"pilot.json",selected|{"means":means,"audited":verified,"paired":paired,"KD":errors,
                "top1":changes,"elapsed_seconds":time.monotonic()-start,"GPU":torch.cuda.get_device_name()})
    print("final",main,means[main],means["regional"],flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    cli.add_argument("--workers",type=int,default=2)
    args=cli.parse_args()
    run(args.workspace,args.workers)
