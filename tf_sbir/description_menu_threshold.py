"""Test where to place the decision threshold for two class-associated sentences."""

import argparse
from itertools import combinations

import numpy as np
import torch

from .description_block_results import Population
from .description_choice_blocks import inputs
from .description_gallery_menu import observers, source_bank
from .description_gallery_transfer import OUT
from .description_menu_bias import assignment, class_estimate
from .native_block_probe import deterministic
from .photo_name_interaction import record, verified
from .wordnet_renaming_pilot import SPLITS


def threshold(a,b,wa,wb):
    # Decide A when the margin exceeds the operating point. Group exact ties before locating cuts.
    z=np.r_[a,b]
    weight=np.r_[wa/wa.sum(),-wb/wb.sum()]
    ids=z.argsort(kind="stable")
    z,w=z[ids],weight[ids]
    vals,idx=np.unique(z,return_index=True)
    net=np.add.reduceat(w,idx)
    loss=np.r_[0.,np.cumsum(net)]
    cuts=np.r_[np.nextafter(vals[0],-np.inf),vals[:-1]+.5*np.diff(vals),np.nextafter(vals[-1],np.inf)]
    best=np.flatnonzero(loss<=loss.min()+1e-12)
    return float(cuts[best[np.abs(cuts[best]).argmin()]])


def decisions(delta,tau):
    return (delta>tau).astype(np.float64)+.5*(delta==tau)


def operating_points(q,photo,y,fold,weights,held,hard):
    csize=q.shape[1]
    fit=fold!=held
    test=fold==held
    source_matrices={"photo":np.zeros((csize,csize)), "sketch":np.zeros((csize,csize))}
    for i,j in combinations(range(csize),2):
        ia,ib=fit&(y==i),fit&(y==j)
        for reader,features in (("photo",photo),("sketch",q)):
            delta=features[:,i]-features[:,j]
            t=threshold(delta[ia],delta[ib],weights[ia],weights[ib])
            source_matrices[reader][i,j]=t
            source_matrices[reader][j,i]=-t
    all_tau={"zero":np.zeros((csize,csize))}
    energies={}
    for name,t in source_matrices.items():
        b=t.mean(axis=1);projected=b[:,None]-b[None,:]
        if not np.allclose((t-projected).sum(axis=1),0.,atol=1e-10):
            raise ValueError("Threshold projection is not orthogonal")
        energies[name]=float(np.square(t-projected).sum()/max(np.square(t).sum(),1e-30))
        all_tau[name]=t;all_tau[name+"_projected"]=projected
    qt=q[test]; labels=y[test]; w=weights[test]
    pairs=[(i,j) for i in range(csize) for j in range(csize) if j!=i]
    trials={}
    for name,tau in all_tau.items():
        margins=qt[:,:,None]-qt[:,None,:]
        answer=decisions(margins,tau)
        win=(answer==1).sum(2)==csize-1
        if (win.sum(1)>1).any():raise ValueError("Two strict all-rival winners")
        score=qt-tau.mean(axis=1)
        nominated=score.argmax(1)
        # Orientations are paired with the same folds, true labels, and balancing weights.
        by_pair=[]
        for i,j in pairs:
            a,b=labels==i,labels==j
            acc=.5*(np.average(answer[a,i,j],weights=w[a])+np.average(answer[b,j,i],weights=w[b]))
            by_pair.append((i,j,int(hard[i]==j),acc))
        # Count cycles of length three in strict tournament edges (ties give no directed edge).
        edges=(answer==1).astype(np.int16)
        cycles=(np.einsum("bij,bji->b",edges@edges,edges)//3).astype(np.int32)
        if (name=="zero" or name.endswith("_projected")) and cycles.any():
            raise ValueError("An additive ordering contained a strict cycle")
        trials[name]={"pairs":np.asarray(by_pair),"winner":win.argmax(1),"unique":win.any(1),
                      "cycles":cycles,"projected_choice":nominated}
    return source_matrices,energies,trials


@torch.inference_mode()
def run(split,revision,table_revision,menu_revision,bias_revision,native_revision,name_revision):
    meta,idx,_=inputs()
    source=verified(OUT/f"{split}-source_membership.pt",table_revision)
    menu=verified(OUT/f"menu-{split}.pt",menu_revision)
    past=verified(OUT/f"bias-{split}.pt",bias_revision)
    qi,labels,ids=(source[k] for k in ("query_ids","class_labels","source_ids"))
    if not torch.equal(past["query_ids"],qi) or not torch.equal(menu["query_ids"],qi):
        raise ValueError("Mismatched menus or selectors")
    su,group,counts=ids.unique(return_inverse=True,return_counts=True)
    _,sf=assignment(split,group,su,labels);fold=sf[group]
    if not torch.equal(fold,past["fold"]):raise ValueError("Source folds changed")
    pop=Population(labels,ids)
    vocab=source_bank(meta["checkpoint"],source["sentence_bank"])
    obs=observers(qi,ids,len(idx["photo"]["paths"]),native_revision,name_revision)
    held_table=source["sentence_table"][1]
    summaries,records={},{}
    y=labels.numpy();f=fold.numpy();wt=(1/counts[group].double()).numpy()
    for fam,v in menu["banks"].items():
        cls=v["winning_candidate_by_class"]
        if len(cls.unique())!=len(cls):raise ValueError("This paired threshold experiment requires distinct class sentences")
        vectors=vocab[cls]
        q=(obs["sketch_two"]@vectors.T).cpu().double().numpy()
        pg=(obs["source"]@vectors.T).cpu().double().numpy()
        paired=past["families"][fam]["pair_auc"].numpy()
        matrices={};energy={};all_trials={}
        for held in (0,1):
            h=paired[(paired[:,0]==held)&(paired[:,3]==1)]
            hard={int(i):int(j) for i,j in h[:,1:3]}
            m,e,trial=operating_points(q,pg,y,f,wt,held,hard)
            matrices[held]={k:torch.from_numpy(z) for k,z in m.items()}
            energy[held]=e;all_trials[held]=trial
        result={};raw={}
        for name in all_trials[0]:
            pos=np.r_[np.flatnonzero(f==0),np.flatnonzero(f==1)]
            order=np.argsort(pos)
            def collect(k):return torch.from_numpy(np.concatenate([all_trials[i][name][k] for i in (0,1)])[order])
            cn,win=collect("unique"),collect("winner")
            pred=collect("projected_choice")
            metric=held_table[cls[pred],labels]
            success=(win==labels)&cn
            allpairs=np.concatenate([all_trials[i][name]["pairs"] for i in (0,1)])
            by={}
            for g,select in (("all",np.ones(len(allpairs),bool)),("hard",allpairs[:,2]==1)):
                z=allpairs[select]
                by[g]=class_estimate([z[z[:,0]==c,3].mean() for c in range(len(cls))])
            summ={"binary_balanced_accuracy":by,"all_rival_coverage":pop.effect(cn),
                  "correct_all_rival_full_denominator":pop.effect(success),
                  "percent_queries_with_three_cycle":pop.effect(collect("cycles")>0)}
            if name=="zero" or name.endswith("_projected"):
                summ["retrieval_mean_percent"]=(100*metric.mean(0)).tolist()
                summ["retrieval_minus_baseline"]=pop.effect(metric-v["metrics"]["semantic"])
                summ["retrieval_minus_raw_sketch"]=pop.effect(metric-v["metrics"]["sketch_two_menu"])
            result[name]=summ
            raw[name]={"unique_winner":cn,"winner":win,"potential_choice":pred,"cycles":collect("cycles"),
                       "pair_accuracy":torch.from_numpy(allpairs)}
        records[fam]={"thresholds":matrices,"arms":raw}
        summaries[fam]={"arms":result,"nonadditive_energy":energy}
    record(OUT/f"threshold-{split}.pt",{"query_ids":qi,"sources":ids,"labels":labels,"fold":fold,
           "families":records},revision,{"split":split,"queries":len(qi),"held_photos":len(source["part_positions"][1]),
           "families":summaries,"table_revision":table_revision,"bias_revision":bias_revision})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",choices=SPLITS,required=True)
    for k in ("revision","table-revision","menu-revision","bias-revision","native-revision","name-revision"):
        p.add_argument("--"+k,required=True)
    a=p.parse_args()
    deterministic()
    run(a.split,a.revision,a.table_revision,a.menu_revision,a.bias_revision,a.native_revision,a.name_revision)
