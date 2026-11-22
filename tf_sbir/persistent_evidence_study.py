"""Test whether the subset-inclusion relation conveys class evidence beyond the scores of the individual partial observations."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .drawing_evidence_subsets import FAMILY, GROUP_COUNT, PARTITIONS, subset_bank
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .recoverable_patch_study import locate, prepare, word_log
from .retrieval_evidence import load_kd, paired_delta


SIZES=(1,2,3)
METHODS=("excess","persist","one_more","max","mean","shuffled","null_difference","loss")
GAINS=(-.18,.06,.18,.5)
CHANNELS=("words","rgb")


def lattice(device):
    size=torch.tensor([i.bit_count() for i in range(1<<GROUP_COUNT)],device=device)
    sets={k:torch.where(size==k)[0] for k in range(GROUP_COUNT+1)}
    gen=torch.Generator().manual_seed(92761)
    perms=[]
    for _ in range(6):
        perm=torch.arange(len(size),device=device)
        for idx in sets.values():
            perm[idx]=idx[torch.randperm(len(idx),generator=gen).to(device)]
        if not torch.equal(size[perm],size):raise ValueError("Changed the distribution of mask sizes")
        perms.append(perm)
    return sets,perms


def lower_envelope(values,one_only=False):
    # Dynamic programming computes exactly min_{T superset S} v(T). One-step control reuses unpropagated v.
    b,n,m=values.shape
    out=values.contiguous().clone()
    original=values.contiguous() if one_only else out
    for i in range(GROUP_COUNT):
        stride=1<<i
        destination=out.view(b,n//(2*stride),2,stride,m)
        source=original.view_as(destination)
        destination[:,:,0]=torch.minimum(destination[:,:,0],source[:,:,1])
    return out


def profile(v,sets,perms):
    env=lower_envelope(v)
    one=lower_envelope(v,True)
    nul={k:[] for k in SIZES}
    for perm in perms:
        rotated=lower_envelope(v[:,perm])
        for k in SIZES:nul[k].append(rotated[:,sets[k]].amax(1))
    scores={}
    for k in SIZES:
        actual=env[:,sets[k]].amax(1)
        worst=one[:,sets[k]].amax(1)
        best=v[:,sets[k]].amax(1)
        null=torch.stack(nul[k])
        z=null.mean(0)
        options={"excess":actual-z,"persist":actual,"one_more":worst,"max":best,"mean":v[:,sets[k]].mean(1),
                 "shuffled":z,"null_difference":.5*(null[::2].mean(0)-null[1::2].mean(0)),"loss":actual-best}
        for name,a in options.items():scores[k,name]=a
    return scores


def algebra(device):
    allm=torch.arange(64,device=device)
    gen=torch.Generator().manual_seed(1307)
    x=torch.randn(2,64,9,generator=gen).to(device)
    v=lower_envelope(x)
    max_error=0.
    for i in range(64):
        proper=allm[(allm&i)==i]
        max_error=max(max_error,float((v[:,i]-x[:,proper].amin(1)).abs().max()))
    sets,perms=lattice(device)
    real=profile(x,sets,perms)
    error=max(float((real[k,"persist"]-real[k+1,"persist"]).clamp_min(0).max()) for k in SIZES[:-1])
    monotone=x[:,0:1]+torch.stack([allm.bitwise_right_shift(i).bitwise_and(1)[:,None]*torch.rand(1,9,generator=gen).to(device)
                                    for i in range(6)]).sum(0)
    mon=profile(monotone,sets,perms)
    exact=max(float((mon[k,"persist"]-mon[k,"max"]).abs().max()) for k in SIZES)
    if max_error>1e-6 or error>1e-6 or exact>1e-6:raise ValueError("Superset guarantee failed")
    return {"exhaustive_min_error":max_error,"size_monotonicity_error":error,"monotone_additive_control":exact}


def grid():
    return ["regional",*(f"{p}/{ch}/{k}/{m}/{g:g}" for p in PARTITIONS for ch in CHANNELS for k in SIZES for m in METHODS for g in GAINS)]


def select(means):
    def best(modes,parts):
        return max((k for k in means if k!="regional" and k.split("/")[0] in parts and k.split("/")[3] in modes),key=means.get)
    real=("grid","appearance")
    options={m:best((m,),real) for m in METHODS}
    options["shuffled_regions"]=best(("excess",),("shuffled",))
    main=options["excess"]
    p,ch,n,_,g=main.split("/")
    matched=[f"{p}/{ch}/{n}/{m}/{g}" for m in METHODS]+[f"shuffled/{ch}/{n}/excess/{g}"]
    return main,options,matched


def scores_by_subset(e,channel,p):
    b,n,d=e.shape
    z=e.reshape(b*n,d)
    if channel=="words":values=word_log(z,p["axes"],p["bank"],p["scale"])
    else:values=p["scale"]*(z@p["rgb"].T)
    values=values.view(b,n,-1)
    values=values-values.mean(2,keepdim=True)
    return values-values[:,:1]


def assess(s,p,data,rows,kd,workers):
    configs=defaultdict(lambda:defaultdict(list))
    for tag in rows:
        if tag=="regional":continue
        part,ch,n,method,g=tag.split("/")
        configs[part,ch][int(n),method].append((tag,float(g)))
    sets,perms=lattice("cuda")
    replay=Replay(kd,p["g"],s) if kd else None
    gl=p["g"].cuda()
    metrics,tops,errors,margins=defaultdict(list),defaultdict(list),{},defaultdict(list)
    with (ThreadPoolExecutor(max_workers=workers) if kd else nullcontext()) as pool:
        for start in range(0,len(p["ids"]),16):
            sl=slice(start,start+16)
            labels=p["y"][sl]
            source=data["subsets"][p["positions"][sl]].cuda().float()
            base=p["base"][sl].cuda()
            std=base.std(1,keepdim=True,unbiased=False)
            candidate=gl[None]==labels.cuda()[:,None]
            pos=base.masked_fill(~candidate,-torch.inf).argmax(1,keepdim=True)
            neg=base.masked_fill(candidate,-torch.inf).argmax(1,keepdim=True)
            def score(tag,matrix):
                v,order,rel,_=map_values(matrix,labels.cuda(),gl,s)
                values=v.cpu()
                if replay:
                    exact=replay.score(order,labels,pool)
                    error=float((values-exact).abs().max())
                    if error>2e-6:raise ValueError("Original KD evaluator disagrees")
                    errors[tag]=max(errors.get(tag,0.),error)
                    values=exact
                metrics[tag].append(values);tops[tag].append(rel[:,0].cpu())
            score("regional",base)
            for (part,ch),variants in configs.items():
                val=scores_by_subset(source[:,PARTITIONS.index(part)],ch,p)
                pieces=defaultdict(list)
                for chunk in val.split(768,dim=2):
                    eff=profile(chunk,sets,perms)
                    for key in variants:pieces[key].append(eff[key])
                for key,arms in variants.items():
                    a=torch.cat(pieces[key],1)
                    scaled=(a-a.mean(1,keepdim=True))/a.std(1,keepdim=True,unbiased=False).clamp_min(1e-6)
                    name=f"{part}/{ch}/{key[0]}/{key[1]}"
                    margins[name].append((scaled.gather(1,pos)-scaled.gather(1,neg)).squeeze(1).cpu())
                    for tag,gain in arms:score(tag,base+gain*std*scaled)
                del val,pieces
            if start%256==0:print(s,"official" if kd else "scan",start,len(p["y"]),flush=True)
    return {k:torch.cat(v) for k,v in metrics.items()},{k:torch.cat(v) for k,v in tops.items()},{k:torch.cat(v) for k,v in margins.items()},errors


@torch.inference_mode()
def run(w):
    inputs=semantic_inputs(w,"sketchy")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Different evaluator")
    data=subset_bank(w,"sketchy","pilot-query")
    rows=grid()
    root=w/"results"/FAMILY
    spec={"revision":revision(),"model":CHECKPOINT,"KD":KD_FUNCTION_SHA,"grid":rows,"gallery_vocabulary":checksum(inputs[0].vocabulary),
          "queries_per_class":36,"seed":63241,"device":torch.cuda.get_device_name(),"calibration":"center each partial-view score over the unlabeled gallery, subtract the empty observation; standardize the final correction on the same gallery",
          "null":"six independent permutations of subset identity within each cardinality, fixed for all images and datasets",
          "choice":"equal mAP on unseen Sketchy splits, then precision; main is the effect of the correct inclusion structure beyond shuffled scores"}
    sig=digest(spec)
    if (root/"protocol.json").exists() and read(root/"protocol.json")["signature"]!=sig:raise ValueError("Changed pilot")
    atomic_json(root/"protocol.json",spec|{"signature":sig,"algebra":algebra("cuda")})
    tables={}
    for s in inputs[2]:
        f=root/("pilot-"+s+".json")
        if f.is_file():
            rec=read(f)
            if rec["signature"]!=sig or checksum(f.with_suffix(".pt"))!=rec["per_query_sha256"]:raise ValueError("Changed pilot files")
            tables[s]=rec["values"];continue
        p=prepare(w,s,inputs,data)
        m,top,margin,_=assess(s,p,data,rows,None,2)
        tables[s]={k:v.double().mean(0).tolist() for k,v in m.items()}
        misses=~top["regional"]
        summary={k:{"all_margin_positive":float((x>0).float().mean()),"initial_top1_wrong_positive":float((x[misses]>0).float().mean()),
                    "median_scaled_margin_on_errors":float(x[misses].median())} for k,x in margin.items()}
        save_tensor({"ids":p["ids"],"labels":p["y"],"metrics":m,"top1":top,"hard_pair_margin":margin},f.with_suffix(".pt"))
        atomic_json(f,{"signature":sig,"values":tables[s],"hard_errors":summary,"top1_errors":int(misses.sum()),"queries":len(misses),
                       "gallery":len(p["g"]),"full_empty_numerical_max":data["checks"].amax(0).tolist(),"per_query_sha256":checksum(f.with_suffix(".pt"))})
        del p
    means={k:[sum(t[k][i] for t in tables.values())/len(tables) for i in (0,1)] for k in rows}
    main,opts,matched=select(means)
    keep=sorted({"regional",*opts.values(),*matched})
    audit,paired,moves,checks={},{},{},{}
    for s in inputs[2]:
        p=prepare(w,s,inputs,data)
        m,top,_,err=assess(s,p,data,keep,kd,2)
        prev=torch.load(root/("pilot-"+s+".pt"),mmap=True,weights_only=True)
        disagreement=max(float((prev["metrics"][k]-v).abs().amax()) for k,v in m.items())
        if disagreement>2e-6:raise ValueError("Shortlist replay changed")
        audit[s]={k:v.double().mean(0).tolist() for k,v in m.items()}
        labs=p["y"].unique(return_inverse=True)[1]
        paired[s]={k:paired_delta(m[main],v,labs) for k,v in m.items()}
        moves[s]={"fixed":int((~top["regional"]&top[main]).sum()),"broken":int((top["regional"]&~top[main]).sum())}
        checks[s]={"queries":len(labs),"KD_max":max(err.values()),"repeat":disagreement}
        del p
    atomic_json(root/"pilot.json",{"main":main,"selected":opts,"matched":matched,"means":means,"audited":audit,"paired":paired,
                                  "top1":moves,"KD":checks})
    print("persistent visual evidence",main,means[main],means["regional"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
