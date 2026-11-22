"""Quantify how far perfect per-query decisions within the tested intervention family could move AP."""

import argparse
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,checksum,read
from .crossdomain_benchmark import revision
from .encoding import save_tensor
from .retrieval_evidence import paired_delta
from .semantic_neuron_gallery import FAMILY


def upper(metrics,choices,labels,base):
    stack=torch.stack([metrics[key] for key in choices])
    rank=stack[:,:,0].argmax(0)
    rows=torch.arange(len(labels))
    chosen=stack[rank,rows]
    gain=100*(chosen[:,0]-base[:,0])
    summary={"configurations":len(choices),"mean_metrics":chosen.double().mean(0).tolist(),"paired":paired_delta(chosen,base,labels),
             "gain_AP_pp_median":float(gain.median()),"gain_AP_pp_p90":float(gain.quantile(.9)),
             "fraction_strictly_improved":float((gain>1e-6).float().mean()),"fraction_gain_above_3pp":float((gain>3).float().mean())}
    return chosen,rank,summary


def run(w,stage):
    root=w/"results"/FAMILY
    lock=read(root/"freeze.json")
    primary=lock["configuration"]["nominee"]
    curve,donor,dim,mode,channel,gain=primary.split("/")
    fullfamily=lock["full_rows"]
    final=[k for k in fullfamily if k not in ("regional","native_rgb") and not k.startswith("standalone:")]
    splits=("sketchy_1","sketchy_2","tuberlin") if stage=="full" else ("sketchy_1","sketchy_2")
    evidence,results={},{}
    for s in splits:
        fname=s if stage=="full" else "pilot-"+s
        f=root/(fname+".pt")
        if not f.is_file():continue
        record=torch.load(f,mmap=True,weights_only=True)
        labels=record["labels"].unique(return_inverse=True)[1]
        score=record["metrics"]
        baseline=score["regional"]
        families={"native_or_frozen_primary":["regional",primary],
                  "all_frozen_regional_controls":["regional",*final]}
        if stage=="pilot":
            families["all_candidates_of_full_semantic_donor_family"]=["regional",*[k for k in score if k!="regional" and k.split("/")[1] in ("sensor","curvature") and k.split("/")[3] in ("late4","late32")]]
            families["all_scanned_interventions"]=["regional",*[k for k in score if k!="regional"]]
        detailed={}
        summary={}
        for name,keys in families.items():
            val,idx,summary[name]=upper(score,keys,labels,baseline)
            detailed[name]={"metrics":val,"choice":idx.to(torch.int16),"names":keys}
        results[s]={"queries":len(labels),"native":baseline.double().mean(0).tolist(),"oracles":summary}
        evidence[s]={"ids":record["ids"],"labels":labels,"ceiling":detailed}
    dest=root/("ceiling-"+stage+".pt")
    save_tensor(evidence,dest)
    atomic_json(dest.with_suffix(".json"),{"revision":revision(),"freeze":lock["sha256"],"per_query_sha256":checksum(dest),
                "splits":results,"scope":"retrospective ceiling of precisely the listed fixed scorers, choosing by each query's true AP after scoring. These choices cannot be used at inference or to claim a bound for untested methods."})
    print({s:{k:round(z["paired"]["mAP"]["delta_pp"],3) for k,z in r["oracles"].items()} for s,r in results.items()},flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    cli.add_argument("--stage",choices=("pilot","full"),default="pilot")
    a=cli.parse_args()
    run(a.workspace,a.stage)
