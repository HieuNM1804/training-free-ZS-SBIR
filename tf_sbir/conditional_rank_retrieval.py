"""Benchmark order-preserving conditional transport on both complete Sketchy galleries, including pure histogram transport and wrong spatial ranks."""

import argparse
import hashlib
import inspect
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_patch_rank_transport import FAMILY,GROUPS,K,LAYERS,METHODS,N,measure,prepare
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .encoding import load_frozen_clip, save_tensor
from .gallery_panel_resume import semantic_inputs
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import stratified_sample
from .retrieval_evidence import load_kd, paired_delta
from .semantic_neuron_counterfactuals import embeddings, code
from .semantic_neuron_gallery import FAMILY as REF, batches, native, to_gpu
from .semantic_neuron_retrieval import evaluate, setup


ARMS=[(l,g,N,m) for l in LAYERS for g in GROUPS for m in METHODS if m!="identity"]
CHANNELS=("words","rgb")
GAINS=(.12,.45,1.)


def extract(w,s,inputs,v):
    cfg,qp,qs,_,_,native_queries,*_=inputs
    qi,y=qs[s]
    rows=qi[stratified_sample(y,36,63241)]
    record=w/"cache"/FAMILY/(s+"-pilot-inputs.pt")
    stamp=digest({"images":digest([str(qp[int(i)]) for i in rows]),"arms":list(map(code,ARMS)),"K":K,"model":CHECKPOINT,
                  "memory":read(w/"cache"/REF/(s+"-memory-v2.json"))["signature"]})
    if record.exists():
        if read(record.with_suffix(".json"))["signature"]!=stamp:raise ValueError("Feature signature changed")
        return torch.load(record,weights_only=True,mmap=True)
    descriptors=torch.load(w/"cache"/REF/(s+"-memory-v2.pt"),mmap=True,weights_only=True)
    bank=to_gpu(descriptors["table"])
    units={l:{g:ix.cuda() for g,ix in sub.items()} for l,sub in descriptors["units"].items()}
    memory=prepare(bank)
    reference,variants,tail=[],[],[]
    for j,x in enumerate(batches(qp,rows,cfg.workers,16)):
        z=native(v,x.cuda(non_blocking=True),units)
        options,ctx=measure(z,bank,memory)
        delta=embeddings(v,z,units,options,ARMS)
        reference.append(z["scene"].cpu());variants.append(delta.half().cpu())
        tail.append(torch.stack([ctx[l,g]["near_saturation"] for l in LAYERS for g in GROUPS],1).cpu())
        if j%16==0:print("inference",s,j*16,len(rows),flush=True)
    raw=torch.cat(reference)
    err=(1-(raw*native_queries[rows]).sum(1)).abs().max()
    if err>.002:raise ValueError("Unchanged CLIP differs")
    data={"ids":rows,"native":raw,"change":torch.cat(variants),"conditional_tail_rate":torch.cat(tail)}
    save_tensor(data,record)
    atomic_json(record.with_suffix(".json"),{"signature":stamp,"max_native_cos_error":float(err)})
    return data


def choose(means,groups,methods):
    return max((k for k in means if k!="regional" and k.split("/")[1] in groups and k.split("/")[3] in methods),key=means.get)


@torch.inference_mode()
def run(w):
    inputs=semantic_inputs(w,"sketchy")
    cfg=inputs[0]
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    v=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("KD evaluator changed")
    names=list(map(code,ARMS))
    rows=["regional",*(f"{r}/{c}/{g:g}" for r in names for c in CHANNELS for g in GAINS)]
    root=w/"results"/FAMILY
    spec={"revision":revision(),"checkpoint":CHECKPOINT,"KD":KD_FUNCTION_SHA,"rows":rows,"sample_seed":63241,
          "pilot_queries_per_class":36,"conditional_pool":K,"units":N,"primary":"scene-conditioned transfer of within-image conditional ranks",
          "selection":"equal average unseen Sketchy mAP, then precision; all alternative methods also tune their own settings"}
    sig=digest(spec)
    if (root/"protocol.json").exists() and read(root/"protocol.json")["signature"]!=sig:raise ValueError("Pilot protocol changed")
    atomic_json(root/"protocol.json",spec|{"signature":sig,"device":torch.cuda.get_device_name()})
    tables={}
    for s in inputs[2]:
        data=extract(w,s,inputs,v)
        context=setup(w,s,inputs,data,"pilot-query")
        dest=root/("pilot-"+s+".json")
        if dest.exists():
            old=read(dest)
            if old["signature"]!=sig or checksum(dest.with_suffix(".pt"))!=old["per_query_sha256"]:raise ValueError("Pilot changed")
            tables[s]=old["values"]
        else:
            m,top,_=evaluate(s,data,{"arms":names},context,rows,None,2)
            tables[s]={k:x.double().mean(0).tolist() for k,x in m.items()}
            save_tensor({"ids":context["ids"],"labels":context["y"],"metrics":m,"top1":top,
                         "conditional_tail_rate":data["conditional_tail_rate"]},dest.with_suffix(".pt"))
            atomic_json(dest,{"signature":sig,"values":tables[s],"queries":len(context["ids"]),"gallery":len(context["g"]),
                             "one_sided_finite_sample_saturation":data["conditional_tail_rate"].mean(0).tolist(),
                             "per_query_sha256":checksum(dest.with_suffix(".pt"))})
        del data,context
    means={r:[sum(t[r][i] for t in tables.values())/len(tables) for i in (0,1)] for r in rows}
    groups=("sensor","curvature")
    best={m:choose(means,groups,(m,)) for m in METHODS if m!="identity"}
    main=best["conditional_scene"]
    l,group,n,_,ch,g=main.split("/")
    match=[f"{l}/{group}/{n}/{m}/{ch}/{g}" for m in METHODS if m!="identity"]
    fixed="standalone:"+main.rsplit("/",2)[0]
    random_units=choose(means,("sensor_matched",),("conditional_scene",))
    keep=sorted({"regional","native_rgb",fixed,random_units,*best.values(),*match})
    summary,paired,errors,fix={},{},{},{}
    for s in inputs[2]:
        data=extract(w,s,inputs,v)
        p=setup(w,s,inputs,data,"pilot-query")
        m,top,er=evaluate(s,data,{"arms":names},p,keep,kd,2)
        previous=torch.load(root/("pilot-"+s+".pt"),weights_only=True,mmap=True)
        mismatch=max(float((previous["metrics"][k]-vv).abs().max()) for k,vv in m.items() if k in previous["metrics"])
        if mismatch>2e-6:raise ValueError("Finalists changed")
        y=p["y"].unique(return_inverse=True)[1]
        summary[s]={k:vv.double().mean(0).tolist() for k,vv in m.items()}
        paired[s]={k:paired_delta(m[main],vv,y) for k,vv in m.items()}
        errors[s]={"official":max(er.values()),"pilot_selection_replay":mismatch,"queries":len(y)}
        fix[s]={"fixed":int((top[main]&~top["regional"]).sum()),"broken":int((~top[main]&top["regional"]).sum())}
        del data,p
    atomic_json(root/"pilot.json",{"nominee":main,"best":best,"matched":match,"separately_matched_units":random_units,
                  "audited":summary,"paired":paired,"KD":errors,"top1":fix,"means":means})
    print("conditional rank method",main,means[main],means["regional"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
