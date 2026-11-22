"""Evaluate a purely visual whole/fragment kernel separately from its use as a correction to the existing lexical regional scorer."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
import os
from pathlib import Path

import torch

from .benchmark import merge_inputs
from .concept_benchmark import atomic_json,checksum,read
from .conditional_agreement_study import Replay
from .crossdomain_benchmark import digest,revision
from .crossdomain_inputs import CHECKPOINT
from .data import SKETCHY_SPLITS
from .drawing_evidence_subsets import PARTITIONS,subset_bank
from .encoding import load_global_inputs,save_tensor
from .fragment_rgb_kernel import GAINS,PARTITION_PAIRS,READOUTS,TEMPERATURES,audit,four_inputs,readout
from .gallery_panel_resume import runtime
from .map_diagnosis import map_values
from .paired_fragment_area_control import read_area
from .paired_fragment_inputs import FAMILY,PARTS,read_features
from .quickdraw_locked import KD_FUNCTION_SHA
from .recoverable_patch_study import locate
from .retrieval_diagnosis import stratified_sample
from .retrieval_evidence import load_kd,paired_delta


BASES=("visual","regional")


def grid():
    return [*BASES,*(f"{kind}/{l}/{r}/{a:g}/{mode}/{g:g}" for kind in BASES for l,r in PARTITION_PAIRS
                           for a in TEMPERATURES for mode in READOUTS for g in GAINS)]


def choose(means,kind):
    def best(mode):
        return max((k for k in means if k.split("/")[0]==kind and len(k.split("/"))==6 and k.split("/")[4]==mode
                    and k.split("/")[1:3] in (["grid","grid"],["appearance","appearance"])),key=means.get)
    options={m:best(m) for m in READOUTS}
    main=options["interaction"]
    prefix,l,r,temp,_,gain=main.split("/")
    matched={f"{prefix}/{l}/{r}/{temp}/{m}/{gain}" for m in READOUTS}
    matched.update(f"{prefix}/{left}/{right}/{temp}/interaction/{gain}" for left,right in PARTITION_PAIRS)
    return options,sorted(matched)


def dataset_inputs(w):
    cfg=runtime(w,"sketchy",batch=256,workers=min(4,len(os.sched_getaffinity(0))))
    qp,qs=merge_inputs(cfg.root,SKETCHY_SPLITS,"sketch")
    pp,gs=merge_inputs(cfg.root,SKETCHY_SPLITS,"photo")
    q,p,_=load_global_inputs(cfg,qp,pp,False)
    return qs,gs,q,p


def condition(w,s,inputs,features):
    qs,gs,rawq,rawp=inputs
    qi,yl=qs[s];pi,gl=gs[s]
    ids=stratified_sample(yl,36,63241)
    qid=qi[ids];where=locate(features["ids"],qid)
    single=2**torch.arange(6)
    val=features["subsets"][where][:,:,single].cuda()
    q={part:val[:,PARTITIONS.index(part)] for part in PARTS}
    q["area_shuffle"]=read_area(w,"sketchy","pilot-query",qid).cuda()
    gallery,_=read_features(w,"sketchy","photo",pi)
    p={part:gallery[:,PARTS.index(part)].contiguous().cuda() for part in PARTS}
    p["area_shuffle"]=read_area(w,"sketchy","photo",pi).cuda()
    regional=torch.load(w/"cache/gallery-stable-region-evidence"/(s+"-regional-reference.pt"),mmap=True,weights_only=True)[ids]
    return {"query":rawq[qid].cuda(),"photos":rawp[pi].cuda(),"q":q,"p":p,"ids":ids,"yl":yl[ids],"gl":gl,"regional":regional}


def score(s,p,entries,kd):
    configs=defaultdict(lambda:defaultdict(list))
    for tag in entries:
        if tag in BASES:continue
        kind,l,r,t,mode,gain=tag.split("/")
        configs[l,r][float(t)].append((tag,kind,mode,float(gain)))
    gallery=p["photos"]
    truth=p["gl"].cuda()
    replay=Replay(kd,p["gl"],s) if kd else None
    m,tops,errs=defaultdict(list),defaultdict(list),{}
    with (ThreadPoolExecutor(max_workers=2) if kd else nullcontext()) as pool:
        for start in range(0,len(p["ids"]),24):
            sl=slice(start,start+24)
            q=p["query"][sl];y=p["yl"][sl]
            base={"visual":q@gallery.T,"regional":p["regional"][sl].cuda()}
            sigma={b:v.std(1,keepdim=True,unbiased=False) for b,v in base.items()}
            def store(k,score):
                val,order,rel,_=map_values(score,y.cuda(),truth,s)
                real=val.cpu()
                if replay:
                    actual=replay.score(order,y,pool)
                    e=float((real-actual).abs().max())
                    if e>2e-6:raise ValueError("KD ranking disagrees")
                    errs[k]=max(errs.get(k,0.),e);real=actual
                m[k].append(real);tops[k].append(rel[:,0].cpu())
            for tag,baseval in base.items():store(tag,baseval)
            for (left,right),temps in configs.items():
                cache=defaultdict(list)
                for j in range(0,len(gallery),640):
                    ix=slice(j,j+640)
                    sim=four_inputs(q,p["q"][left][sl],gallery[ix],p["p"][right][ix])
                    for temp,active in temps.items():
                        study=readout(sim,temp)
                        for feature in {k[2] for k in active}:cache[temp,feature].append(study[feature])
                for temp,active in temps.items():
                    scaled={}
                    for nam,kind,feature,g in active:
                        if feature not in scaled:
                            x=torch.cat(cache.pop((temp,feature)),1)
                            scaled[feature]=(x-x.mean(1,keepdim=True))/x.std(1,keepdim=True,unbiased=False).clamp_min(1e-5)
                        store(nam,base[kind]+g*sigma[kind]*scaled[feature])
            if start%240==0:print(s,"KD" if kd else "grid",start,len(p["ids"]),flush=True)
    return {k:torch.cat(v) for k,v in m.items()},{k:torch.cat(v) for k,v in tops.items()},errs


@torch.inference_mode()
def run(w):
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    inputs=dataset_inputs(w)
    features=subset_bank(w,"sketchy","pilot-query")
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Wrong evaluator")
    root=w/"results"/FAMILY/"rgb-kernel"
    entries=grid()
    meta={"revision":revision(),"model":CHECKPOINT,"KD":KD_FUNCTION_SHA,"rows":entries,"seed":63241,"per_class":36,
          "scope":"visual baseline and visual kernel use no text or vocabulary. Regional rows explicitly add the very same correction to the frozen public-word regional scores.",
          "selection":"separate equal-weight two-split unseen mAP optima for dictionary-free and pre-existing regional baselines, precision as a tiebreaker"}
    sig=digest(meta)
    if (root/"protocol.json").exists() and read(root/"protocol.json")["signature"]!=sig:raise ValueError("Changed pilot")
    atomic_json(root/"protocol.json",meta|{"signature":sig,"algebra":audit(),"gpu":torch.cuda.get_device_name()})
    tables={}
    for s in inputs[0]:
        f=root/("pilot-"+s+".json")
        if f.exists():
            old=read(f)
            if old["signature"]!=sig or checksum(f.with_suffix(".pt"))!=old["per_query_sha256"]:raise ValueError("Changed data")
            tables[s]=old["values"];continue
        p=condition(w,s,inputs,features)
        m,t,_=score(s,p,entries,None)
        tables[s]={k:v.double().mean(0).tolist() for k,v in m.items()}
        save_tensor({"ids":p["ids"],"labels":p["yl"],"metrics":m,"top1":t},f.with_suffix(".pt"))
        atomic_json(f,{"signature":sig,"values":tables[s],"queries":len(p["ids"]),"gallery":len(p["gl"]),
                       "per_query_sha256":checksum(f.with_suffix(".pt"))});del p
    means={k:[sum(t[k][i] for t in tables.values())/len(tables) for i in (0,1)] for k in entries}
    controls={b:choose(means,b) for b in BASES}
    finalist=sorted({*BASES,*[name for opt,matches in controls.values() for name in (*opt.values(),*matches)]})
    report,paired,change,errs={},{},{},{}
    for s in inputs[0]:
        p=condition(w,s,inputs,features)
        m,top,errors=score(s,p,finalist,kd)
        old=torch.load(root/("pilot-"+s+".pt"),mmap=True,weights_only=True)
        discrepancy=max(float((v-old["metrics"][k]).abs().max()) for k,v in m.items())
        if discrepancy>2e-6:raise ValueError("Finalist changed")
        report[s]={k:v.double().mean(0).tolist() for k,v in m.items()}
        lab=p["yl"].unique(return_inverse=True)[1]
        paired[s]={};change[s]={}
        for base in BASES:
            main=controls[base][0]["interaction"]
            paired[s][base]={k:paired_delta(m[main],v,lab) for k,v in m.items() if k==base or k.startswith(base+"/")}
            change[s][base]={"fixed":int((~top[base]&top[main]).sum()),"broken":int((top[base]&~top[main]).sum())}
        errs[s]={"KD_max":max(errors.values()),"repeat":discrepancy,"queries":len(lab)};del p
    atomic_json(root/"pilot.json",{"selected":controls,"means":means,"audited":report,"paired":paired,"top1":change,"KD":errs})
    for b in BASES:
        k=controls[b][0]["interaction"]
        print("visual kernel",b,k,means[k],means[b],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
