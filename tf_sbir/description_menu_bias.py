"""Separate common sentence preferences from cross-subject changes in sentence discrimination."""

import argparse
import hashlib

import numpy as np
import torch

from .description_block_results import Population
from .description_choice_blocks import inputs
from .description_gallery_menu import observers, source_bank
from .description_gallery_transfer import OUT
from .native_block_probe import SEED, deterministic
from .photo_name_interaction import record, verified
from .wordnet_renaming_pilot import SPLITS


def assignment(split,group,subjects,labels):
    subject_class=torch.empty(len(subjects),dtype=labels.dtype)
    subject_class[group]=labels
    if not torch.equal(subject_class[group],labels):raise ValueError("Source reused by multiple classes")
    fold=torch.empty_like(subject_class)
    for c in labels.unique():
        members=torch.where(subject_class==c)[0].tolist()
        members.sort(key=lambda j:hashlib.sha256(f"{SEED}/menu-fold/{split}/{int(subjects[j])}".encode()).digest())
        fold[members]=torch.arange(len(members))%2
    return subject_class,fold


def weighted_auc(positive,negative,wp,wn):
    order=np.argsort(negative,kind="stable")
    values=negative[order]
    mass=np.r_[0.,np.cumsum(wn[order],dtype=np.float64)]
    lo=np.searchsorted(values,positive,side="left")
    hi=np.searchsorted(values,positive,side="right")
    return float(wp@((mass[lo]+mass[hi])/2)/(wp.sum()*mass[-1]))


def class_estimate(values):
    a=np.asarray(values,dtype=np.float64)
    gen=np.random.default_rng(SEED)
    draws=gen.integers(0,len(a),size=(10000,len(a)))
    return {"class_points":100*float(a.mean()),
            "interval95":(100*np.quantile(a[draws].mean(axis=1),(.025,.975))).tolist()}


def pair_diagnostics(q,photo,labels,fold,group,counts,winner):
    nclasses=len(winner)
    w=counts[group].double().reciprocal().numpy()
    z=q.numpy(); p=photo.numpy(); y=labels.numpy(); f=fold.numpy()
    win=winner.numpy()
    rows=[]
    for held in (0,1):
        for c in range(nclasses):
            others=np.where(win!=win[c])[0]
            if not len(others):continue
            train=(f!=held)&(y==c)
            # Duplicate class winners receive one vote as a sentence.
            alternative=np.unique(win[others])
            top=alternative[z[train][:,alternative].argmax(1)]
            mass=np.bincount(top,weights=w[train],minlength=z.shape[1])
            rival_word=alternative[mass[alternative].argmax()]
            hard=int(np.flatnonzero(win==rival_word)[0])
            for d in others:
                pos=(f==held)&(y==c)
                neg=(f==held)&(y==d)
                vals=[]
                for score in (z,p):
                    delta=score[:,win[c]]-score[:,win[d]]
                    vals.append(weighted_auc(delta[pos],delta[neg],w[pos],w[neg]))
                rows.append((held,c,int(d),int(d==hard),*vals))
    rec=np.asarray(rows,dtype=np.float64)
    result={}
    for key,take in (("hard",rec[:,3]==1),("all",np.ones(len(rec),dtype=bool))):
        pick=rec[take]
        by_class=np.stack([pick[pick[:,1]==c,4:].mean(0) for c in range(nclasses)])
        result[key]={"directed_fold_pairs":len(pick),"sketch":class_estimate(by_class[:,0]),
                     "source":class_estimate(by_class[:,1]),"source_minus_sketch":class_estimate(by_class[:,1]-by_class[:,0])}
    return torch.from_numpy(rec),result


def subject_sum(value,group,n):
    out=torch.zeros((n,value.shape[1]),dtype=value.dtype)
    out.index_add_(0,group,value)
    return out


@torch.inference_mode()
def run(split,revision,table_revision,menu_revision,name_revision,native_revision):
    meta,idx,_=inputs()
    prior=verified(OUT/f"{split}-source_membership.pt",table_revision)
    menu=verified(OUT/f"menu-{split}.pt",menu_revision)
    qi,labels,sources=(prior[k] for k in ("query_ids","class_labels","source_ids"))
    if not torch.equal(menu["query_ids"],qi):raise ValueError("Reference menu query order differs")
    all_vectors=source_bank(meta["checkpoint"],prior["sentence_bank"])
    seen=observers(qi,sources,len(idx["photo"]["paths"]),native_revision,name_revision)
    ids,group,counts=sources.unique(return_inverse=True,return_counts=True)
    subject_class,subject_fold=assignment(split,group,ids,labels)
    fold=subject_fold[group]
    pop=Population(labels,sources)
    table=prior["sentence_table"][1]
    label_count=len(labels.unique())
    summaries,artifacts={},{}
    for family,v in menu["banks"].items():
        options=v["menu"]
        im=(seen["sketch_two"]@all_vectors[options].T).cpu().double()
        photo=(seen["source"]@all_vectors[options].T).cpu().double()
        winner=torch.searchsorted(options,v["winning_candidate_by_class"])
        if not torch.equal(options[winner],v["winning_candidate_by_class"]):raise ValueError("Winner not in menu")
        # Centering the shift removes an immaterial additive constant per subject.
        shift=photo-im
        shift-=shift.mean(1,keepdim=True)
        sums=subject_sum(shift,group,len(ids))
        mean=sums/counts[:,None]
        corrected={k:torch.empty_like(im) for k in ("shared","true_class","other_class")}
        for held in (0,1):
            reference=subject_fold!=held
            class_shifts=torch.stack([mean[reference&(subject_class==c)].mean(0) for c in range(label_count)])
            dest=fold==held
            corrected["shared"][dest]=im[dest]+class_shifts.mean(0)
            corrected["true_class"][dest]=im[dest]+class_shifts[labels[dest]]
            corrected["other_class"][dest]=im[dest]+class_shifts[(labels[dest]+1)%label_count]
        available=counts[group]>1
        redraw=im[available]+(sums[group[available]]-shift[available])/(counts[group[available],None]-1)
        raw={"sketch":im,"source":photo,**corrected,"same_subject_other_drawings":redraw}
        choices={};metrics={};summary={}
        for k,z in raw.items():
            mask=available if k=="same_subject_other_drawings" else torch.ones_like(available)
            choice=options[z.argmax(1)]
            choices[k]=choice
            values=table[choice,labels[mask]]
            metrics[k]=values
            qpop=pop if mask.all() else Population(labels[mask],sources[mask])
            summary[k]={"queries":len(values),"mean_percent":(100*values.mean(0)).tolist(),
                        "delta_retrieval_vs_sketch":qpop.effect(values-v["metrics"]["sketch_two_menu"][mask])}
        if not torch.equal(choices["sketch"],v["decisions"]["sketch_two"]["menu"]):
            # A different GEMM width from the full bank may switch exact ties. Check score regret.
            original=torch.searchsorted(options,v["decisions"]["sketch_two"]["menu"])
            regret=(im.max(1).values-im.gather(1,original[:,None]).flatten()).max().item()
            if regret>1e-5:raise ValueError("Menu replay differs materially")
        else:regret=0.
        records,auc=pair_diagnostics(im,photo,labels,fold,group,counts,winner)
        summaries[family]={"retrieval":summary,"auc":auc,"maximum_sketch_replay_regret":regret,
                           "duplicate_class_winners":len(winner)-len(options)}
        artifacts[family]={"menu":options,"winning_candidate_by_class":v["winning_candidate_by_class"],
                           "choices":choices,"metrics":metrics,"pair_auc":records}
    record(OUT/f"bias-{split}.pt",{"query_ids":qi,"source_ids":sources,"classes":labels,"fold":fold,
           "has_other_drawing":counts[group]>1,"families":artifacts},revision,
           {"split":split,"queries":len(qi),"sources":len(ids),"test_photos":len(prior["part_positions"][1]),
            "families":summaries,"menu_revision":menu_revision,"table_revision":table_revision})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",choices=SPLITS,required=True)
    for key in ("revision","table-revision","menu-revision","name-revision","native-revision"):
        p.add_argument("--"+key,required=True)
    a=p.parse_args()
    deterministic()
    run(a.split,a.revision,a.table_revision,a.menu_revision,a.name_revision,a.native_revision)
