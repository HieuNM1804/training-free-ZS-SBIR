"""Choose observable-word interventions on complete Sketchy, then audit the frozen transfer."""

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
from .depiction_channel import algebra_checks
from .encoding import save_tensor
from .foundation import load_words
from .gallery_depiction_study import FAMILY, KINDS, READ, describe, eval_split, key, parse, views, choose
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import load_kd
from .semantic import read_words


def followup(pilot):
    vals=pilot["means"]
    pairs=set()
    for name,take in (("visible",4),("scalar",2),("shuffled",2)):
        ordered=sorted((k for k in vals if k.startswith(name+"/")),key=vals.get,reverse=True)
        best=[]
        for k in ordered:
            param=parse(k)[1:3]
            if param not in best:
                best.append(param)
            if len(best)==take:
                break
        pairs.update(best)
    # Include nearby signed contrasts to discriminate the two coefficients.
    _,v,b,_=parse(pilot["nominee"])
    pairs.update(((v*.5,b*.5),(v*1.5,b*1.5),(v,0.),(0.,b)))
    gains=(-1.,-.5,-.2,.1,.2,.35,.5,1.,1.6)
    return sorted({"regional",*(key(k,a,b,m) for k in KINDS for a,b in pairs for m in READ),
                   *(key(k,a,a,m) for k in ("raw","raw_free") for a in gains for m in READ)})


@torch.inference_mode()
def run(w,phase,workers):
    root=w/"results"/FAMILY
    pilot=read(root/"pilot.json")
    cells=followup(pilot)
    sp={"revision":revision(),"pilot":checksum(root/"pilot.json"),"cells":cells,"source":CHECKPOINT,
        "selection":"mean full unseen Sketchy mAP then mean precision","nulls":algebra_checks(torch.device("cuda"))}
    # Invariant values depend on floating-point arithmetic and are diagnostic,
    # not a condition for reusing a previously recorded result.
    sig=digest({k:v for k,v in sp.items() if k!="nulls"})
    doc=root/"full-protocol.json"
    if doc.exists() and read(doc)["sha256"]!=sig:
        raise ValueError("Full study revision changed")
    atomic_json(doc,sp|{"sha256":sig})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Changed reference evaluator")
    if phase=="audit":
        freeze=read(root/"freeze.json")
        if freeze["full_sketchy_sha256"]!=checksum(root/"shortlist.json") or freeze["revision"]!=revision():
            raise ValueError("Invalid full Sketchy lock")
        primary=freeze["nominee"]
        cells=sorted({"regional",primary,freeze["best_raw"],freeze["best_scalar"],freeze["best_shuffled"],
                      *freeze["matched"],*freeze["ablation"]})
    for ds in (("sketchy",) if phase=="lock" else ("sketchy","tuberlin")):
        cfg,*inputs=global_inputs(w,ds)
        terms=read_words(cfg.vocabulary)
        if len(terms)!=20931:
            raise ValueError("Changed public vocabulary")
        src,tgt=(u.cuda() for u in load_words(cfg,terms))
        _,scale=bare_words(cfg,terms)
        build_gallery(w,ds)
        reader=views(w,ds,inputs[0])
        torch.backends.cuda.matmul.allow_tf32=True
        for s,(_,y) in inputs[1].items():
            dest=root/((s if phase=="audit" else "shortlist-"+s)+".json")
            if dest.is_file():
                prev=read(dest)
                if prev["signature"]!=sig or (phase=="audit" and prev["per_query_sha256"]!=checksum(dest.with_suffix(".pt"))):
                    raise ValueError("Changed full score record")
                continue
            m,t,d,checks,errs=eval_split(w,ds,s,cfg,inputs,src,tgt,scale,reader,torch.arange(len(y)),cells,
                                        kd if phase=="audit" else None,workers)
            vals={k:v.mean(0).tolist() for k,v in m.items()}
            row={"signature":sig,"queries":len(y),"gallery":len(inputs[3][s][0]),"values":vals,"max_errors":checks}
            if phase=="audit":
                diag,paired=describe(m,t,d,freeze,y)
                file=dest.with_suffix(".pt")
                save_tensor({"metrics":m,"top1_correct":t,"query_labels":y,"depiction":d},file)
                row.update({"primary":primary,"baseline":"regional","matched_base":freeze["best_raw"],
                            "best_contrast":freeze["best_control"],"protocol":{"split":s,"gpu":torch.cuda.get_device_name(),
                            "revision":revision(),"checkpoint":CHECKPOINT,"visible_words":len(terms),"evaluator":KD_FUNCTION_SHA,
                            "vocabulary_sha256":checksum(cfg.vocabulary),"classes":len(y.unique())},
                            "scores":{k:dict(zip(("mAP","precision"),v)) for k,v in vals.items()},"vs_controls":paired,
                            "per_query_sha256":checksum(file),"original_kd_queries_per_condition":len(y),"kd_max_abs_error":errs,
                            "top1_changes":{"errors_fixed":diag["top1_fixed"],"hits_broken":diag["top1_broken"]},"diagnosis":diag})
            atomic_json(dest,row)
            print("stored",phase,s,vals["regional"],max(vals,key=vals.get),flush=True)
    if phase=="lock":
        tables={s:read(root/("shortlist-"+s+".json"))["values"] for s in ("sketchy_1","sketchy_2")}
        mean={k:[sum(v[k][i] for v in tables.values())/2 for i in (0,1)] for k in cells}
        chosen=choose(mean)
        chosen["best_control"]=max((chosen["best_raw"],chosen["best_scalar"],chosen["best_shuffled"]),key=mean.get)
        atomic_json(root/"shortlist.json",{"means":mean,"splits":tables})
        chosen.update({"revision":revision(),"primary":chosen["nominee"],"full_sketchy_sha256":checksum(root/"shortlist.json")})
        atomic_json(root/"freeze.json",chosen|{"sha256":digest(chosen)})
        print("locked",chosen["primary"],mean[chosen["primary"]],"control",chosen["best_control"],flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    ap.add_argument("--phase",choices=("lock","audit"),required=True)
    ap.add_argument("--workers",type=int,default=8)
    a=ap.parse_args()
    run(a.workspace,a.phase,a.workers)
