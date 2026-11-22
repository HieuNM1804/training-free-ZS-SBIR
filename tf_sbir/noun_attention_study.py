"""Test separate, training-free ranking changes for the amount and content of noun attention."""

import argparse
import hashlib
import inspect
from pathlib import Path
import time

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import SIZES, bare_words
from .cross_view_gallery import run as build_gallery
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .depiction_token_study import READ, evaluate, key, parse
from .encoding import save_tensor
from .foundation import load_words
from .gallery_depiction_study import views
from .noun_attention_paths import FAMILY
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs, stratified_sample
from .retrieval_evidence import load_kd, paired_delta
from .semantic import read_words


LAYER=11
PAIRS=((-4.,0.),(0.,-4.),(-4.,-8.),(-8.,-4.),(-4.,4.),(4.,-4.),(-8.,0.),(0.,-8.),
       (-8.,-2.),(-2.,-8.),(-2.,0.),(0.,-2.),(2.,0.),(0.,2.),(6.,0.),(0.,6.),(-10.,-5.),(-5.,-10.))
KINDS=("mass","collinear","common","head_swap","source_mass")
UNSPLIT=(-10.,-8.,-6.,-4.,-2.,-1.,1.,2.,4.,6.,8.)
ENDPOINT=(-1.,-.5,-.2,.2,.5,1.,1.6)


def grid():
    return ["regional",*(key(kind,LAYER,a,b,r) for kind in KINDS for a,b in PAIRS for r in READ),
            *(key(kind,LAYER,a,a,r) for kind in ("raw","raw_free") for a in UNSPLIT for r in READ),
            *(key("endpoint",LAYER,a,0.,r) for a in ENDPOINT for r in READ)]


def load_paths(w,cfg,src,tgt):
    path=w/"cache"/FAMILY/"interventions.pt"
    desc=read(path.with_suffix(".json"))
    if desc["vocabulary"]!=checksum(cfg.vocabulary) or desc["checkpoint"]!=CHECKPOINT:
        raise ValueError("Frozen text bank changed")
    vals=torch.load(path,weights_only=True,mmap=True)
    mass,message,common,swap,whole,one_side,_=vals.unbind(1)
    total=whole.cuda().float()
    routing=mass.cuda().float()
    projected=((routing*total).sum(1,keepdim=True)/total.square().sum(1,keepdim=True).clamp_min(1e-12))*total
    bank={"mass":routing,"common":common.cuda().float(),"head_swap":swap.cuda().float(),"collinear":projected,
          "source_mass":one_side.cuda().float(),"endpoint":src-tgt}
    closure=((routing+message.cuda().float()-total).square().mean()/total.square().mean()).sqrt().item()
    if closure>.002:
        raise ValueError("Relative storage precision changed the split")
    return {(k,LAYER):vec for k,vec in bank.items()},total,desc


def choose(means):
    def best(pred):
        return max((k for k in means if pred(k)),key=means.get)
    main=best(lambda k:k.startswith("mass/"))
    _,l,a,b,s=parse(main)
    out={"nominee":main,"best_unsplit":best(lambda k:k=="regional" or k.startswith(("raw/","raw_free/"))),
         "best_endpoint":best(lambda k:k.startswith("endpoint/")),
         "matched":[key(k,l,a,b,s) for k in KINDS[1:]],"ablation":[key("mass",l,a,b,r) for r in READ],
         "mass_only":best(lambda k:k.startswith("mass/") and parse(k)[3]==0),
         "message_only":best(lambda k:k.startswith("mass/") and parse(k)[2]==0)}
    out.update({"best_"+kind:best(lambda k:k.startswith(kind+"/")) for kind in KINDS[1:]})
    return out


def representatives(choices):
    return sorted({"regional",choices["nominee"],choices["mass_only"],choices["message_only"],
                   *(v for k,v in choices.items() if k.startswith("best_")),*choices["matched"],*choices["ablation"]})


@torch.inference_mode()
def run(w,workers):
    started=time.monotonic()
    cfg,*inputs=global_inputs(w,"sketchy")
    terms=read_words(cfg.vocabulary)
    if len(terms)!=20931:
        raise ValueError("Changed public text inputs")
    src,tgt=(v.cuda() for v in load_words(cfg,terms))
    _,scale=bare_words(cfg,terms)
    bank,total,desc=load_paths(w,cfg,src,tgt)
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Unexpected KD function")
    cells=grid()
    spec={"revision":revision(),"checkpoint":CHECKPOINT,"vocabulary":checksum(cfg.vocabulary),"token_index":desc["signature"],
          "source_layer":LAYER,"pairs":PAIRS,"unsplit":UNSPLIT,"prompt":ENDPOINT,"alternatives":KINDS,
          "selection":"equal mean full-gallery AP for the two balanced Sketchy pilots then precision",
          "seed":63241,"per_class":36,"KD":KD_FUNCTION_SHA}
    sig=digest(spec)
    root=w/"results"/FAMILY
    if (root/"protocol.json").is_file() and read(root/"protocol.json")["sha256"]!=sig:
        raise ValueError("Incompatible pilot")
    atomic_json(root/"protocol.json",spec|{"sha256":sig,"text_audit":desc})
    build_gallery(w,"sketchy")
    reader=views(w,"sketchy",inputs[0])
    samples={s:stratified_sample(y,36,63241) for s,(_,y) in inputs[1].items()}
    torch.backends.cuda.matmul.allow_tf32=True
    tables={}
    for s,ids in samples.items():
        path=root/("pilot-"+s+".json")
        if path.is_file():
            row=read(path)
            if row["signature"]!=sig or row["per_query_sha256"]!=checksum(path.with_suffix(".pt")):
                raise ValueError("Stale condition")
            tables[s]=row["values"]
            continue
        m,t,diag,_,base_err=evaluate(w,"sketchy",s,cfg,inputs,src,tgt,scale,reader,ids,cells,bank,None,workers,total)
        vals={k:v.mean(0).tolist() for k,v in m.items()}
        save_tensor({"metrics":m,"top1_correct":t,"query_labels":inputs[1][s][1][ids],"query_ids":ids,"routing_diagnostics":diag},path.with_suffix(".pt"))
        atomic_json(path,{"signature":sig,"queries":len(ids),"gallery":SIZES[s][1],"values":vals,
                          "mass_relative_rms_correlation_distinctive_rms_global_and_local":diag[str(LAYER)].mean(0).tolist(),
                          "baseline_error":base_err,"per_query_sha256":checksum(path.with_suffix(".pt"))})
        tables[s]=vals
    means={k:[sum(t[k][j] for t in tables.values())/2 for j in (0,1)] for k in cells}
    choices=choose(means)
    audit=representatives(choices)
    per_split,errors,contrast,top={},{},{},{}
    for s,ids in samples.items():
        m,t,_,er,_=evaluate(w,"sketchy",s,cfg,inputs,src,tgt,scale,reader,ids,audit,bank,kd,workers,total)
        cached=torch.load(root/("pilot-"+s+".pt"),mmap=True,weights_only=True)
        worst=max(float((v-cached["metrics"][k]).abs().max()) for k,v in m.items())
        if worst>2e-6:
            raise ValueError(f"K-D replay changed the measured list {worst}")
        k=choices["nominee"]
        per_split[s]={n:v.mean(0).tolist() for n,v in m.items()}
        errors[s]={"KD_queries_per_condition":len(ids),"max_error":max(er.values()),"replay":worst}
        contrast[s]={n:paired_delta(m[k],v,cached["query_labels"]) for n,v in m.items()}
        top[s]={"errors_fixed":int((~t["regional"]&t[k]).sum()),"hits_broken":int((t["regional"]&~t[k]).sum())}
    atomic_json(root/"pilot.json",choices|{"audited":per_split,"means":means,"KD":errors,"top1":top,
                         "paired":contrast,"seconds":time.monotonic()-started,"device":torch.cuda.get_device_name()})
    print("selected",choices["nominee"],means[choices["nominee"]],"unsplit",choices["best_unsplit"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--workers",type=int,default=8)
    a=p.parse_args()
    run(a.workspace,a.workers)
