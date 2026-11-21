"""Measure post-retrieval variation across independently scored sketches of one photographed object."""

import argparse
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,read
from .regional_interventions import FAMILY
from .retrieval_diagnosis import global_inputs


def aggregates(values, group, count):
    return torch.zeros(count,dtype=torch.float64).index_add_(0,group,values.double())


def decompose(ap, group, cat, n_group, n_class):
    ap=ap.double()
    gc=len(n_group)
    nc=len(n_class)
    o=aggregates(ap,group,gc)/n_group
    c=aggregates(ap,cat,nc)/n_class
    terms={"within_object":(ap-o[group]).square().sum(),
           "between_objects_within_class":(o[group]-c[cat]).square().sum(),
           "between_classes":(c[cat]-ap.mean()).square().sum()}
    total=(ap-ap.mean()).square().sum()
    if abs(sum(terms.values())-total)>1e-8:raise ValueError("Variance decomposition does not close")
    eligible=n_group[group]>1
    ss_o=aggregates(ap.square(),group,gc)
    s_o=aggregates(ap,group,gc)
    ss_c=aggregates(ap.square(),cat,nc)
    s_c=aggregates(ap,cat,nc)
    q=ap[eligible]
    g=group[eligible];cls=cat[eligible]
    within=(q.square()*n_group[g]-2*q*s_o[g]+ss_o[g])/(n_group[g]-1)
    unlike=n_class[cls]-n_group[g]
    across=q.square()-2*q*(s_c[cls]-s_o[g])/unlike+(ss_c[cls]-ss_o[g])/unlike
    return {"descriptive_ap_variance_share":{k:float(v/total) for k,v in terms.items()},
            "mean_squared_ap_distance_other_sketch_same_object":float(within.mean()),
            "mean_squared_ap_distance_same_class_different_object":float(across.mean())}


def run(w):
    cfg,qp,qs,pp,gs,_,_=global_inputs(w,"sketchy")
    report={}
    for split,(qi,y) in qs.items():
        pi,gallery=gs[split]
        names={(Path(pp[idx]).parent.name,Path(pp[idx]).stem) for idx in pi.tolist()}
        groups={}
        use=[];keys=[]
        uncertain=[]
        for i,idx in enumerate(qi.tolist()):
            path=Path(qp[idx]);stem,sep,last=path.stem.rpartition("-")
            obj=(path.parent.name,stem)
            if not sep or not last.isdigit() or obj not in names:
                uncertain.append(path.name);continue
            if obj not in groups:groups[obj]=len(groups)
            use.append(i);keys.append(groups[obj])
        if not use:raise ValueError("No confirmed photo IDs")
        take=torch.tensor(use); group=torch.tensor(keys)
        blob=torch.load(w/"results"/FAMILY/(split+".pt"),weights_only=True)
        meta=read(w/"results"/FAMILY/(split+".json"))
        if not torch.equal(blob["query_labels"],y):raise ValueError("Different query order")
        cat=y[take]
        no=group.bincount().double()
        nc=cat.bincount().double()
        hits=(gallery[blob["top25"][take,0].long()]==cat).double()
        ordered=aggregates(hits,group,len(no))
        hc=aggregates(hits,cat,len(nc))
        eligible=(no[group]>1)&(hits==0)
        unlike=nc[cat]-no[group]
        same=ordered[group]/(no[group]-1).clamp_min(1)
        other=(hc[cat]-ordered[group])/unlike.clamp_min(1)
        if not eligible.any():raise ValueError("No missed retrieval with another verified drawing")
        out={"queries":len(qi),"queries_with_verified_source_photo":len(take),"unverified":len(uncertain),
             "object_instances":len(no),"objects_with_multiple_sketches":int((no>1).sum()),
             "top1_mistakes_with_sibling":int(eligible.sum()),
             "a_sibling_gets_category_top1_given_current_miss":float((ordered[group][eligible]>0).double().mean()),
             "random_sibling_top1_given_current_miss":float(same[eligible].mean()),
             "other_photo_random_sketch_top1_given_current_miss_within_same_class":float(other[eligible].mean())}
        out["method_breakdown"]={}
        for k in ("baseline","matched_base","best_contrast","primary"):
            ap=blob["metrics"][meta[k]][take,0]
            out["method_breakdown"][k]=decompose(ap,group,cat,no,nc)
        report[split]=out
    atomic_json(w/"results"/FAMILY/"within-object-diagnosis.json",report)
    for k,d in report.items():
        print(k,d["queries_with_verified_source_photo"],d["object_instances"],d["a_sibling_gets_category_top1_given_current_miss"],
              d["method_breakdown"]["primary"],flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(a.parse_args().workspace)
