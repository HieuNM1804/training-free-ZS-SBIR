"""Summarize retrieval utility, paired controls, and complete blockwise score accounting."""

import argparse
import json

import numpy as np
import torch

from .concept_benchmark import atomic_json,checksum
from .description_choice_blocks import OUT,PARTS
from .native_block_probe import SEED
from .wordnet_renaming_pilot import SPLITS


class Population:
    def __init__(self,labels,sources):
        self.labels=labels
        self.sources=sources
        ids,self.group,counts=sources.unique(return_inverse=True,return_counts=True)
        self.counts=counts.double()
        self.source_classes=torch.empty(len(ids),dtype=labels.dtype)
        self.source_classes[self.group]=labels
        if not torch.equal(self.source_classes[self.group],labels):raise ValueError("A photo has two classes")
        self.classes=labels.unique()
        self.boot=np.random.default_rng(SEED).integers(len(self.classes),size=(10000,len(self.classes)))

    def effect(self,d):
        d=d.double()
        means=torch.zeros((len(self.counts),*d.shape[1:]),dtype=torch.float64)
        means.index_add_(0,self.group,d)
        shape=(-1,)+(1,)*(d.ndim-1)
        means/=self.counts.reshape(shape)
        clas=torch.stack([means[self.source_classes==c].mean(0) for c in self.classes]).numpy()
        return {"query_points":(100*d.mean(0)).tolist(),"subject_class_points":(100*clas.mean(0)).tolist(),
                "interval95":(100*np.quantile(clas[self.boot].mean(1),(.025,.975),axis=0)).tolist()}


def selected(m,u):
    return torch.where((m>=0).unsqueeze(-1),u[:,0,None,:],u[:,1,None,:])


def accuracy(m,u):
    diff=u[:,0,0]-u[:,1,0]
    keep=diff!=0
    agreement=m[keep]*diff[keep,None]
    return (100*((agreement>0).double()+.5*(agreement==0)).mean(0)).tolist()


def fixed_panel(scores,u,pop,has_redraw):
    selected_end={}
    readings={}
    for role,values in scores.items():
        mask=has_redraw if role=="redraw" else torch.ones_like(has_redraw)
        v=u[mask]
        rp=Population(pop.labels[mask],pop.sources[mask]) if role=="redraw" else pop
        y=values["ledger"][mask].double()
        loc=values["local"][mask].double()
        actual=values["native"][mask].double()[:,None]
        results=selected(y,v)
        local=selected(loc,v)
        end=selected(actual,v)[:,0]
        selected_end[role]=end
        signed_utility=(v[:,0,0]-v[:,1,0])[:,None]
        writes=torch.cat((y[:,:1],y.diff(dim=1),actual-y[:,-1:]),1)*signed_utility
        if (writes.sum(1)-actual[:,0]*signed_utility[:,0]).abs().max()>1e-10:raise ValueError("Unaccounted signed margin")
        old=results[:,8,0]
        readings[role]={"queries":len(v),"ledger_AP_P":(100*results.mean(0)).tolist(),"local_AP_P":(100*local.mean(0)).tolist(),
                        "native_AP_P":(100*end.mean(0)).tolist(),"ledger_better_choice_percent":accuracy(y,v),
                        "local_better_choice_percent":accuracy(loc,v),"native_better_choice_percent":accuracy(actual,v)[0],
                        "signed_utility_writes":(100*writes.mean(0)).tolist(),
                        "native_minus_ledger8":rp.effect(end-results[:,8]),
                        "native_minus_local8":rp.effect(end-local[:,1]),
                        "last_four_AP_gain_loss_tie":[int((end[:,0]>old).sum()),int((end[:,0]<old).sum()),int((end[:,0]==old).sum())]}
    effects={}
    for a,b in (("source","sketch"),("source","sketch_two"),("source","same_class_photo"),
                ("same_class_photo","other_class_photo"),("redraw","sketch")):
        if a=="redraw":
            effects[a+"_minus_"+b]=Population(pop.labels[has_redraw],pop.sources[has_redraw]).effect(selected_end[a]-selected_end[b][has_redraw])
        else:
            effects[a+"_minus_"+b]=pop.effect(selected_end[a]-selected_end[b])
    return {"unequal_AP":int((u[:,0,0]!=u[:,1,0]).sum()),"two_oracle_AP_P":(100*u.amax(1).mean(0)).tolist(),
            "readers":readings,"paired_endpoint_effects":effects}


def free_panel(blob,pop):
    labels=blob["classes"]
    utility=blob["class_utility"]
    src=blob["source"]
    choices={"sketch":blob["sketch"],"sketch_two":blob["sketch_two"],"source":src,
             "same_class_photo":src[blob["peer"]],"other_class_photo":src[blob["other"]],"redraw":blob["sketch"][blob["redraw"]]}
    out,all_end={},{}
    has_redraw=blob["redraw"]!=torch.arange(len(labels))
    for name,ids in choices.items():
        u=utility[ids.long(),labels[:,None,None]]
        rp=pop
        if name=="redraw":
            u=u[has_redraw]
            rp=Population(pop.labels[has_redraw],pop.sources[has_redraw])
        measures={"first":u[:,:,0,:],"best_two_oracle":u.amax(2)}
        out[name]={}
        all_end[name]={}
        for key,t in measures.items():
            end=t[:,-1]
            out[name][key]={"queries":len(u),"AP_P_curves":(100*t.mean(0)).tolist(),"native_minus_ledger8":rp.effect(end-t[:,1]),
                            "native_minus_local8":rp.effect(end-t[:,4])}
            all_end[name][key]=end
    effects={}
    for a,b in (("source","sketch"),("source","sketch_two"),("source","same_class_photo"),("redraw","sketch")):
        if a=="redraw":
            rp=Population(pop.labels[has_redraw],pop.sources[has_redraw])
            effects[a+"_minus_"+b]={k:rp.effect(all_end[a][k]-all_end[b][k][has_redraw]) for k in all_end[a]}
        else:
            effects[a+"_minus_"+b]={k:pop.effect(all_end[a][k]-all_end[b][k]) for k in all_end[a]}
    for a,b in (("source","same_class_photo"),("sketch","redraw")):
        for key in out[a]:
            baseline=np.array(out[a][key]["AP_P_curves"])
            if b=="redraw":
                v=utility[choices[a].long(),labels[:,None,None]][has_redraw]
                baseline=(100*(v[:,:,0,:] if key=="first" else v.amax(2)).mean(0)).numpy()
            control=np.array(out[b][key]["AP_P_curves"])
            if abs(baseline-control).max()>1e-8:raise ValueError("Within-class permutation changed free-nomination utility")
    return {"readers":out,"paired_endpoint_effects":effects,
            "columns":["ledger_4","ledger_8","ledger_12","local_4","local_8","local_12","native"]}


def run(name,revision,measure_revision,native_revision):
    data={}
    checks={}
    for phase in ("fixed","free"):
        path=OUT/f"{name}-{phase}.pt"
        m=json.loads(path.with_suffix(".json").read_bytes())
        if checksum(path)!=m["sha256"] or m["revision"]!=measure_revision or m["native_revision"]!=native_revision:
            raise ValueError("Unverified study inputs")
        data[phase]=torch.load(path,mmap=True,weights_only=True)
        checks[phase]=m["checks"]
    f,full=data["fixed"],data["free"]
    for k in ("query_ids","gallery_ids","source_ids","pair","peer","redraw","other","classes"):
        if not torch.equal(f[k],full[k]):raise ValueError("Unequal panels")
    pop=Population(f["classes"],f["source_ids"])
    u=f["candidate_metrics"].double()
    has_redraw=f["redraw"]!=torch.arange(len(u))
    tables={k:fixed_panel(f[k],u[f["peer"]] if k=="independent" else u,pop,has_redraw) for k in ("self","independent")}
    free=free_panel(full,pop)
    native={}
    for mode in ("sketch","photo"):
        for i in range(PARTS):
            m=json.loads((OUT/f"native-{mode}-{i}.json").read_bytes())
            if m["revision"]!=native_revision:raise ValueError("Stale features")
            native[f"{mode}_{i}"]={k:m[k] for k in ("sha256","rows","maximum_l2_error")}
    atomic_json(OUT/f"summary-{name}.json",{"revision":revision,"measure_revision":measure_revision,
                "native_revision":native_revision,"split":name,"seed":SEED,"sketches":len(u),"gallery":len(f["gallery_ids"]),
                "sources":len(pop.counts),"classes":len(pop.classes),"fixed":tables,"free":free,"checks":checks,
                "native_shards":native,"fixed_output_sha256":checksum(OUT/f"{name}-fixed.pt"),
                "free_output_sha256":checksum(OUT/f"{name}-free.pt")})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",choices=SPLITS,required=True)
    p.add_argument("--revision",required=True)
    p.add_argument("--measure-revision",required=True)
    p.add_argument("--native-revision",required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    run(a.split,a.revision,a.measure_revision,a.native_revision)
