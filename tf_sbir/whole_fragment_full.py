"""Lock the selected four-state comparison on Sketchy and run the unchanged original evaluator on complete galleries."""

import argparse
import hashlib
import inspect
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,checksum,read
from .crossdomain_benchmark import revision,digest
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .paired_fragment_area_control import read_area
from .paired_fragment_inputs import FAMILY,read_features
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd,paired_delta
from .semantic_neuron_retrieval import setup
from .whole_fragment_study import evaluate


def lock(w):
    root=w/"results"/FAMILY
    pilot=read(root/"pilot.json")
    main=pilot["main"]
    n,left,right,graph,_,gain=main.split("/")
    null="area_shuffle" if left=="appearance" else "shuffled"
    spatial=[f"{n}/{a}/{b}/{graph}/interaction/{gain}" for a,b in ((null,null),)]
    rows=sorted({"regional",main,pilot["best"]["source"],pilot["best"]["target"],pilot["best"]["additive"],
                 pilot["best"]["cluster_shuffled"],pilot["best"]["excess"],pilot["best"]["parts"],
                 *spatial,*[f"{n}/{left}/{right}/{graph}/{m}/{gain}" for m in ("source","target","cluster_shuffled","cluster_mean")]})
    meta={"main":main,"rows":rows,"two_pilot_splits":list(pilot["audited"]),"pilot_sha256":checksum(root/"pilot.json"),
          "definition":"source and target observations; same photo partitions and same control grid on both Sketchy splits and TU; only reindex the current unlabeled target gallery",
          "code":revision()}
    sig=digest(meta)
    path=root/"freeze.json"
    if path.exists() and read(path)["sha256"]!=sig:raise ValueError("A different lock already exists")
    atomic_json(path,{"sha256":sig,"protocol":meta})
    print("locked",sig,main,len(rows),flush=True)


@torch.inference_mode()
def run(w,ds):
    root=w/"results"/FAMILY
    frozen=read(root/"freeze.json")
    cfg=frozen["protocol"]
    if digest(cfg)!=frozen["sha256"]:raise ValueError("Different lock")
    inputs=semantic_inputs(w,ds)
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("KD-SBIR changed")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    for s,(qi,labels) in inputs[2].items():
        f=root/(s+".json")
        if f.is_file():
            old=read(f)
            if old["lock"]!=frozen["sha256"] or checksum(f.with_suffix(".pt"))!=old["per_query_sha256"]:raise ValueError("Invalid full result")
            continue
        features,_=read_features(w,ds,"sketch",qi)
        extra=read_area(w,ds,"all-query",qi)
        data={"ids":qi,"singletons":features,"area":extra}
        p=setup(w,s,inputs,data,"all-query")
        p["positions"]=torch.arange(len(qi))
        met,top,margins,errs=evaluate(w,s,inputs,data,p,cfg["rows"],kd)
        main=cfg["main"];y=labels.unique(return_inverse=True)[1]
        archive=f.with_suffix(".pt")
        save_tensor({"ids":p["ids"],"labels":labels,"metrics":met,"top1":top,"hard_pairs":margins},archive)
        delta={k:paired_delta(met[main],val,y) for k,val in met.items()}
        atomic_json(f,{"lock":frozen["sha256"],"code":revision(),"main":main,"rows":cfg["rows"],
                       "queries":len(qi),"gallery":len(p["g"]),"classes":len(y.unique()),"device":torch.cuda.get_device_name(),
                       "values":{k:a.double().mean(0).tolist() for k,a in met.items()},"paired_main_vs_each":delta,
                       "top1":{"fixed":int((~top["regional"]&top[main]).sum()),"broken":int((top["regional"]&~top[main]).sum())},
                       "KD_max_error":max(errs.values()),"per_query_sha256":checksum(archive)})
        print("full",s,met[main].mean(0).tolist(),met["regional"].mean(0).tolist(),flush=True)
        del data,features,extra,p


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--stage",choices=("lock","full"),required=True)
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    a=p.parse_args()
    if a.stage=="lock":lock(a.workspace)
    else:run(a.workspace,a.dataset)
