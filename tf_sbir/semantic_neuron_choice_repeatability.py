"""After independent ranking, test whether a beneficial intervention recurs on another drawing of the exact same photographed object."""

import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from .concept_benchmark import atomic_json, checksum, read
from .crossdomain_benchmark import revision
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .retrieval_evidence import paired_delta
from .semantic_neuron_gallery import FAMILY


def verified_sources(qpaths,ppaths,qids,pids):
    counts=defaultdict(int)
    for i in pids.tolist():
        f=Path(ppaths[i]);counts[f.parent.name,f.stem]+=1
    sources=[]
    for i in qids.tolist():
        f=Path(qpaths[i])
        stem,sep,suffix=f.stem.rpartition("-")
        key=f.parent.name,stem
        sources.append(key if sep and suffix.isdigit() and counts[key]==1 else None)
    return sources


def reference_panels(sources,label,base):
    groups=defaultdict(list)
    for q,source in enumerate(sources):
        if source is not None:groups[source].append(q)
    ids=list(groups)
    cats=np.array([label[groups[k][0]] for k in ids])
    size=np.array([len(groups[k]) for k in ids])
    level=np.array([float(np.median(base[groups[k]])) for k in ids])
    record=[]
    for i,source in enumerate(ids):
        members=groups[source]
        if len(members)<2:continue
        k=len(members)-1
        eligible=(size>=k)&(np.arange(len(ids))!=i)
        have_same=np.where(eligible&(cats==cats[i]))[0]
        have_other=np.where(eligible&(cats!=cats[i]))[0]
        if not len(have_same) or not len(have_other):continue
        for q in members:
            siblings=[j for j in members if j!=q]
            med=np.median(base[siblings])
            def pick(choices):
                j=choices[np.abs(level[choices]-med).argmin()]
                remaining=list(groups[ids[j]])
                selected=[]
                for val in sorted(base[siblings]):
                    idx=int(np.argmin(np.abs(base[remaining]-val)))
                    selected.append(remaining.pop(idx))
                return selected
            record.append((q,siblings,pick(have_same),pick(have_other)))
    return record


def study(metrics,keys,labels,panels,fixed):
    vals=torch.stack([metrics[k] for k in keys],1)
    true=vals[:,:,0]
    choice={"self":[],"siblings":[],"other_object_same_class":[],"other_class":[],"native":[],"fixed":[]}
    base=keys.index("regional")
    fix=keys.index(fixed)
    for j,one,two,three in panels:
        choice["self"].append(int(true[j].argmax()))
        for label,ref in (("siblings",one),("other_object_same_class",two),("other_class",three)):
            choice[label].append(int(true[ref].mean(0).argmax()))
        choice["native"].append(base);choice["fixed"].append(fix)
    qs=torch.tensor([j for j,*_ in panels])
    codes={k:torch.tensor(c) for k,c in choice.items()}
    outcomes={k:vals[qs,c] for k,c in codes.items()}
    y=labels[qs].unique(return_inverse=True)[1]
    report={"queries_with_verified_independent_drawings":len(qs),"classes":int(y.max()+1),"number_of_available_scorers":len(keys),
            "mean_metrics":{k:v.double().mean(0).tolist() for k,v in outcomes.items()},
            "versus_untouched":{k:paired_delta(v,outcomes["native"],y) for k,v in outcomes.items()},
            "sibling_minus_different_subject_same_class":paired_delta(outcomes["siblings"],outcomes["other_object_same_class"],y),
            "sibling_selects_exact_same_arm_as_self":float((codes["siblings"]==codes["self"]).float().mean()),
            "alternative_same_class_selects_same_arm_as_self":float((codes["other_object_same_class"]==codes["self"]).float().mean())}
    return report,{"queries":qs,"choices":codes,"metrics":outcomes,"scorers":keys}


def run(w,stage):
    inputs=semantic_inputs(w,"sketchy")
    _,qp,qs,pp,gs,*_=inputs
    root=w/"results"/FAMILY
    frozen=read(root/"freeze.json")
    primary=frozen["configuration"]["nominee"]
    final=[k for k in frozen["full_rows"] if k not in ("regional","native_rgb") and not k.startswith("standalone:")]
    report,stored={},{}
    for s,(qi,yl) in qs.items():
        name=s if stage=="full" else "pilot-"+s
        pack=torch.load(root/(name+".pt"),mmap=True,weights_only=True)
        ids=pack["ids"]
        y=yl[ids]
        metrics=pack["metrics"]
        base=metrics["regional"][:,0].numpy()
        sources=verified_sources(qp,pp,qi[ids],gs[s][0])
        panels=reference_panels(sources,y.numpy(),base)
        sets={"binary":["regional",primary],"frozen":["regional",*final]}
        if stage=="pilot":
            sets["all_conditional_candidates"]=["regional",*[k for k in metrics if k!="regional" and k.split("/")[1] in ("sensor","curvature") and k.split("/")[3] in ("late4","late32")]]
            sets["all_candidates"]=["regional",*[k for k in metrics if k!="regional"]]
        rb,db={},{}
        for n,keys in sets.items():
            rb[n],db[n]=study(metrics,keys,y,panels,primary)
        dists=[abs(np.median(base[ix])-np.median(base[control])) for _,ix,control,_ in panels]
        report[s]={"sample":len(ids),"verified_filenames":sum(k is not None for k in sources),
                   "used":len(panels),"mean_number_other_drawings":float(np.mean([len(i) for _,i,*_ in panels])),
                   "median_difference_in_donor_baseline_AP_for_same_class_control":float(np.median(dists)),"families":rb}
        stored[s]={"ids":ids,"panels":panels,"comparisons":db}
        print(s,len(panels),[(k,round(v["versus_untouched"]["siblings"]["mAP"]["delta_pp"],3),
                              round(v["sibling_minus_different_subject_same_class"]["mAP"]["delta_pp"],3)) for k,v in rb.items()],flush=True)
    path=root/("choice-repeatability-"+stage+".pt")
    save_tensor(stored,path)
    atomic_json(path.with_suffix(".json"),{"revision":revision(),"scope":"retrospective audit. Every retrieval score had already been computed for its query in isolation; selectors use AP of other labeled queries solely to estimate transfer ceilings, never as a deployable retrieval method.",
                "splits":report,"per_query_sha256":checksum(path),"controls":"same number of drawings of a single other object of the same or different class, matched to the reference drawings by original AP; exact photo identity verified by unique filename"})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--stage",choices=("pilot","full"),default="pilot")
    args=parser.parse_args()
    run(args.workspace,args.stage)
