"""Compare the cross-domain whole/fragment likelihood interaction with separately optimized single-image contrasts."""

import argparse
import hashlib
import inspect
from collections import defaultdict
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,checksum,read
from .conditional_agreement_study import Replay
from .crossdomain_benchmark import digest,revision
from .crossdomain_inputs import CHECKPOINT
from .drawing_evidence_subsets import PARTITIONS,subset_bank
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .map_diagnosis import map_values
from .paired_fragment_inputs import FAMILY
from .paired_fragment_area_control import read_area
from .quickdraw_locked import KD_FUNCTION_SHA
from .recoverable_patch_study import prepare
from .retrieval_evidence import load_kd,paired_delta
from .whole_fragment_likelihood import GAINS,GRAPH,METHODS,PAIRS,SIZES,algebra,catalog,mean_by_group,prob,score_parts,target_tables


def grid():
    return ["regional",*(f"{n}/{ps}/{pt}/{graph}/{m}/{a:g}" for n in SIZES for ps,pt in PAIRS
                         for graph in GRAPH for m in METHODS for a in GAINS)]


def select(means):
    def best(methods,real=True):
        valid=[k for k in means if k!="regional" and k.split("/")[4] in methods and
               (not real or k.split("/")[1:3] in (["grid","grid"],["appearance","appearance"]))]
        return max(valid,key=means.get)
    optimum={m:best((m,)) for m in METHODS}
    main=optimum["interaction"]
    n,qs,ps,graph,_,gain=main.split("/")
    matched={f"{n}/{qs}/{ps}/{graph}/{m}/{gain}" for m in METHODS}
    matched.update(f"{n}/{a}/{b}/{graph}/interaction/{gain}" for a,b in PAIRS)
    return main,optimum,sorted(matched)


def evaluate(w,s,inputs,data,p,rows,kd):
    names=defaultdict(lambda:defaultdict(list))
    for name in rows:
        if name=="regional":continue
        n,l,r,graph,m,g=name.split("/")
        names[int(n)][l,r,graph].append((name,m,float(g)))
    _,_,qs,_,_,raw,_,src,tgt,scale=inputs
    cat=catalog(w,s,inputs)
    qabs=qs[s][0][p["ids"]]
    full=raw[qabs].cuda()
    singleton=2**torch.arange(6)
    zz=data["subsets"][p["positions"]][:,:,singleton].cuda().float()
    group=cat["groups"].cuda()
    perm=cat["permutation"].cuda()
    detail=Replay(kd,p["g"],s) if kd else None
    truth=p["g"].cuda()
    store,top,margin,worst=defaultdict(list),defaultdict(list),defaultdict(list),{}
    with (ThreadPoolExecutor(max_workers=2) if kd else nullcontext()) as pool:
        for j,configs in names.items():
            readout=target_tables(w,s,"sketchy",inputs,cat,j)
            axes=src[cat["order"][:j]]
            source_full=prob(full,axes,scale)
            source_parts={}
            for k in {v[0] for v in configs}:
                views=data["area"][p["positions"]].cuda().float() if k=="area_shuffle" else zz[:,PARTITIONS.index(k)]
                source_parts[k]=prob(views,axes,scale).mean(1)
            fixed={}
            for _,part,g in configs:
                if (part,g) not in fixed:
                    ref=readout[part,g]
                    fixed[part,g]=(ref[perm],mean_by_group(ref,group))
            for start in range(0,len(p["ids"]),64):
                sl=slice(start,start+64)
                base=p["base"][sl].cuda();ys=p["y"][sl]
                size=base.std(1,keepdim=True,unbiased=False)
                correct=truth[None]==ys.cuda()[:,None]
                pos=base.masked_fill(~correct,-torch.inf).argmax(1,keepdim=True)
                neg=base.masked_fill(correct,-torch.inf).argmax(1,keepdim=True)
                def record(k,matrix):
                    val,order,rel,_=map_values(matrix,ys.cuda(),truth,s)
                    obs=val.cpu()
                    if detail:
                        official=detail.score(order,ys,pool)
                        err=float((official-obs).abs().max())
                        if err>2e-6:raise ValueError("KD-SBIR rank replay failed")
                        worst[k]=max(worst.get(k,0.),err)
                        obs=official
                    store[k].append(obs);top[k].append(rel[:,0].cpu())
                if j==next(iter(names)):record("regional",base)
                for (left,right,graph),spec in configs.items():
                    result=score_parts(source_full[sl],source_parts[left][sl],readout["whole",graph],readout[right,graph],
                                       *fixed[right,graph])
                    scaled={}
                    for name,m,gain in spec:
                        if m not in scaled:
                            xx=result[m]
                            if not torch.isfinite(xx).all():raise ValueError("Nonfinite likelihood")
                            delta=(xx-xx.mean(1,keepdim=True))/xx.std(1,keepdim=True,unbiased=False).clamp_min(1e-6)
                            scaled[m]=size*delta
                            margin[f"{j}/{left}/{right}/{graph}/{m}"].append((delta.gather(1,pos)-delta.gather(1,neg)).squeeze(1).cpu())
                        record(name,base+gain*scaled[m])
                    del scaled,result
                if start%384==0:print(s,j,"KD" if kd else "scan",start,len(p["ids"]),flush=True)
            del fixed,readout,source_parts
    return ({k:torch.cat(v) for k,v in store.items()},{k:torch.cat(v) for k,v in top.items()},
            {k:torch.cat(v) for k,v in margin.items()},worst)


@torch.inference_mode()
def run(w):
    inputs=semantic_inputs(w,"sketchy")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data=subset_bank(w,"sketchy","pilot-query")
    data["area"]=read_area(w,"sketchy","pilot-query",data["ids"])
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Unexpected evaluator")
    rows=grid()
    root=w/"results"/FAMILY
    spec={"code":revision(),"model":CHECKPOINT,"KD":KD_FUNCTION_SHA,"vocab":checksum(inputs[0].vocabulary),"rows":rows,
          "queries_per_unseen_class":36,"sample_seed":63241,"selection":"equal mean of unseen Sketchy mAP then precision. Confer the same tuning opportunity on all control families.",
          "primary":"four-cell log interaction with genuine whole and fragment observations drawn from each image",
          "gpu":torch.cuda.get_device_name()}
    sig=digest(spec)
    if (root/"protocol.json").exists() and read(root/"protocol.json")["signature"]!=sig:raise ValueError("Pilot signature differs")
    atomic_json(root/"protocol.json",spec|{"signature":sig,"algebra":algebra()})
    tables={}
    for s in inputs[2]:
        out=root/("pilot-"+s+".json")
        if out.exists():
            previous=read(out)
            if previous["signature"]!=sig or checksum(out.with_suffix(".pt"))!=previous["per_query_sha256"]:raise ValueError("Changed scan")
            tables[s]=previous["values"];continue
        p=prepare(w,s,inputs,data)
        m,t,margin,_=evaluate(w,s,inputs,data,p,rows,None)
        errors=~t["regional"]
        tables[s]={k:v.double().mean(0).tolist() for k,v in m.items()}
        save_tensor({"ids":p["ids"],"labels":p["y"],"metrics":m,"top1":t,"hard_pair":margin},out.with_suffix(".pt"))
        cat=catalog(w,s,inputs)
        coverage={n:float(cat["prior"][cat["order"][:n]].sum()) for n in SIZES}
        atomic_json(out,{"signature":sig,"queries":len(errors),"gallery":len(p["g"]),"values":tables[s],"catalog_probability_mass":coverage,
                         "hard_pair_positive_on_original_top1_errors":{k:float((v[errors]>0).float().mean()) for k,v in margin.items()},
                         "baseline_errors":int(errors.sum()),"per_query_sha256":checksum(out.with_suffix(".pt"))})
        del p
    means={k:[sum(t[k][i] for t in tables.values())/len(tables) for i in (0,1)] for k in rows}
    main,opts,matched=select(means)
    checks=sorted({"regional",*opts.values(),*matched})
    audit,paired,repairs,errors={},{},{},{}
    for s in inputs[2]:
        p=prepare(w,s,inputs,data)
        m,t,_,err=evaluate(w,s,inputs,data,p,checks,kd)
        previous=torch.load(root/("pilot-"+s+".pt"),mmap=True,weights_only=True)
        change=max(float((v-previous["metrics"][k]).abs().max()) for k,v in m.items())
        if change>2e-6:raise ValueError("Final ranking changed")
        audit[s]={k:v.double().mean(0).tolist() for k,v in m.items()}
        y=p["y"].unique(return_inverse=True)[1]
        paired[s]={k:paired_delta(m[main],v,y) for k,v in m.items()}
        repairs[s]={"fixed":int((~t["regional"]&t[main]).sum()),"broken":int((t["regional"]&~t[main]).sum())}
        errors[s]={"queries":len(y),"KD_max_error":max(err.values()),"repeat_max":change}
        del p
    atomic_json(root/"pilot.json",{"main":main,"best":opts,"matched":matched,"means":means,"audited":audit,"paired":paired,
                "top1":repairs,"KD":errors,"signature":sig})
    print("four-way whole fragment",main,means[main],means["regional"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    a=p.parse_args()
    run(a.workspace)
