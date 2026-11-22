"""Lock the noun-state hypothesis on complete Sketchy and measure original KD metrics on transfer."""

import argparse
import hashlib
import inspect
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import bare_words
from .cross_view_gallery import run as build_gallery
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .depiction_token_paths import FAMILY
from .depiction_token_study import READ, TYPES, banks, choose, evaluate, key, parse
from .encoding import save_tensor
from .foundation import load_words
from .gallery_depiction_study import views
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import load_kd, paired_delta
from .semantic import read_words


def full_grid(pilot):
    means = pilot["means"]
    params = set()
    for kind, number in (("noun",3),("collinear",2),("exchanged",2),("single",2)):
        best = []
        for k in sorted((k for k in means if k.startswith(kind+"/")), key=means.get, reverse=True):
            choice = parse(k)[1:4]
            if choice not in best:
                best.append(choice)
            if len(best) == number:
                break
        params.update(best)
    _,layer,a,b,_ = parse(pilot["nominee"])
    params.update((layer,a*ratio,b*ratio) for ratio in (.5,1.5,2.))
    params.add((layer,-a,-b))
    # Both optimized raw variants see their own signed strength range.
    raw = (-1.,-.5,-.2,.1,.2,.35,.5,1.,1.5)
    return sorted({"regional", *(key(k,l,a,b,r) for k in TYPES for l,a,b in params for r in READ),
                   *(key(k,0,a,a,r) for k in ("raw","raw_free") for a in raw for r in READ)})


@torch.inference_mode()
def run(w,phase,workers):
    root=w/"results"/FAMILY
    ref=read(root/"pilot.json")
    cells=full_grid(ref)
    sp={"revision":revision(),"pilot_sha256":checksum(root/"pilot.json"),"checkpoint":CHECKPOINT,
        "cells":cells,"evaluator":KD_FUNCTION_SHA,"decision":"full unseen Sketchy equal mAP then precision"}
    sig=digest(sp)
    path=root/"full-protocol.json"
    if path.is_file() and read(path)["sha256"]!=sig:
        raise ValueError("This is a different source or full protocol")
    atomic_json(path,sp|{"sha256":sig})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Changed original evaluator")
    if phase=="audit":
        frozen=read(root/"freeze.json")
        if frozen["revision"]!=revision() or frozen["full_sketchy_sha256"]!=checksum(root/"shortlist.json"):
            raise ValueError("Full Sketchy lock does not match the ranking study")
        method=frozen["nominee"]
        cells=sorted({"regional",method,frozen["best_raw"],frozen["best_single"],frozen["best_collinear"],
                      frozen["best_exchanged"],*frozen["matched"],*frozen["ablation"]})
    for ds in (("sketchy",) if phase=="lock" else ("sketchy","tuberlin")):
        cfg,*inputs=global_inputs(w,ds)
        terms=read_words(cfg.vocabulary)
        if len(terms)!=20931:
            raise ValueError("Changed vocabulary")
        src,tgt=(v.cuda() for v in load_words(cfg,terms))
        _,scale=bare_words(cfg,terms)
        text,meta=banks(w,terms,src,tgt,{parse(k)[:2] for k in cells if k!="regional"})
        if meta["vocabulary"]!=checksum(cfg.vocabulary):
            raise ValueError("Mismatched text cache")
        build_gallery(w,ds)
        reader=views(w,ds,inputs[0])
        torch.backends.cuda.matmul.allow_tf32=True
        for s,(_,y) in inputs[1].items():
            dest=root/((s if phase=="audit" else "shortlist-"+s)+".json")
            if dest.is_file():
                old=read(dest)
                if old["signature"]!=sig or (phase=="audit" and old["per_query_sha256"]!=checksum(dest.with_suffix(".pt"))):
                    raise ValueError("Incompatible completed split")
                continue
            ix=torch.arange(len(y))
            m,t,measure,er,base_err=evaluate(w,ds,s,cfg,inputs,src,tgt,scale,reader,ix,cells,text,
                                            kd if phase=="audit" else None,workers)
            vals={k:v.mean(0).tolist() for k,v in m.items()}
            result={"signature":sig,"queries":len(y),"gallery":len(inputs[3][s][0]),"values":vals,"reference_error":base_err}
            if phase=="audit":
                file=dest.with_suffix(".pt")
                save_tensor({"metrics":m,"query_labels":y,"top1_correct":t,"intervention_by_layer":measure},file)
                result.update({"primary":method,"baseline":"regional","matched_base":frozen["best_raw"],"best_contrast":frozen["best_control"],
                               "protocol":{"split":s,"gpu":torch.cuda.get_device_name(),"checkpoint":CHECKPOINT,"revision":revision(),
                                           "evaluator":KD_FUNCTION_SHA,"vocabulary_sha256":checksum(cfg.vocabulary),"classes":len(y.unique())},
                               "scores":{k:dict(zip(("mAP","precision"),v)) for k,v in vals.items()},
                               "vs_controls":{k:paired_delta(m[method],v,y) for k,v in m.items()},
                               "top1_changes":{"errors_fixed":int((~t["regional"]&t[method]).sum()),"hits_broken":int((t["regional"]&~t[method]).sum())},
                               "mean_rms_correlation_residual_per_layer":{k:v.mean(0).tolist() for k,v in measure.items()},
                               "per_query_sha256":checksum(file),"original_kd_queries_per_condition":len(y),"kd_max_abs_error":er})
            atomic_json(dest,result)
            print("stored",phase,s,max(vals,key=vals.get),"base",vals["regional"],flush=True)
    if phase=="lock":
        tables={s:read(root/("shortlist-"+s+".json"))["values"] for s in ("sketchy_1","sketchy_2")}
        means={k:[sum(v[k][i] for v in tables.values())/2 for i in (0,1)] for k in cells}
        chosen=choose(means)
        chosen["best_control"]=max((chosen[k] for k in ("best_raw","best_collinear","best_exchanged","best_single")),key=means.get)
        atomic_json(root/"shortlist.json",{"means":means,"splits":tables})
        chosen.update({"primary":chosen["nominee"],"revision":revision(),"full_sketchy_sha256":checksum(root/"shortlist.json")})
        atomic_json(root/"freeze.json",chosen|{"sha256":digest(chosen)})
        print("frozen",chosen["primary"],means[chosen["primary"]],"control",chosen["best_control"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--phase",choices=("lock","audit"),required=True)
    p.add_argument("--workers",type=int,default=8)
    a=p.parse_args()
    run(a.workspace,a.phase,a.workers)
