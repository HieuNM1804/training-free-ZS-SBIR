"""Estimate the class-conditional witness counterfactual after scoring every Sketchy query."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import revision
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs


FAMILY="category-conditional-evidence"
FIELDS=("correct_class_wins","reciprocal_class_rank","correct_beats_fixed_wrong","true_minus_fixed_wrong",
        "correct_covariance_advantage","switch_favors_true","true_category_score","wrong_category_score")
KINDS=("regional_global","regional_native","whole_native","base_native")


def midrank(x):
    value,index=x.sort(dim=1,stable=True)
    # Half the tied mass includes the observation itself. Values always have full-gallery mean one half.
    left=torch.searchsorted(value,value,right=False)
    right=torch.searchsorted(value,value,right=True)
    return torch.empty_like(x).scatter_(1,index,(left+right).to(x.dtype)/(2*x.size(1)))


def measures(a,b,member,count):
    fa=(a@member)/count
    fb=(b@member)/count
    same=(a*b)@member/count
    product=fa*fb
    distinct=(count*product-same)/(count-1)
    return {"first":fa,"second":fb,"arithmetic":.5*(fa+fb),"same":same,"independent":product,"distinct":distinct,
            "opposite":2*same-distinct}


def run_split(s,qs,gs,qraw,praw,w):
    qi,y=qs[s]
    pi,g=gs[s]
    base_file=w/"cache/gallery-stable-region-evidence"/(s+"-regional-reference.pt")
    base=torch.load(base_file,mmap=True,weights_only=True)
    hist=torch.load(w/"cache/pragmatic-gallery-competition"/(s+"-named-rgb16.pt"),mmap=True,weights_only=True)
    member=torch.nn.functional.one_hot(g).cuda().float()
    counts=member.sum(0)
    if (counts<2).any():raise ValueError("A category has no pairs of distinct photos")
    photo=praw[pi].cuda()
    results=defaultdict(list)
    category_values=defaultdict(list)
    miss=[]
    rivals=[]
    err=0.
    for offset in range(0,len(qi),96):
        sl=slice(offset,offset+96)
        locked=base[sl].cuda()
        h=hist[sl].cuda().clamp_min(1e-30).log()
        native=qraw[qi[sl]].cuda()@photo.T
        regional=(locked-.65*h)/.35
        inc=midrank(locked)
        whole=midrank(h)
        local=midrank(regional)
        vis=midrank(native)
        gold=y[sl].cuda()
        r=torch.arange(len(gold),device=gold.device)
        top=g.cuda()[locked.argmax(1)]
        wrong=top.clone()
        ok=top==gold
        alt=locked.masked_fill(g.cuda()[None,:]==gold[:,None],-torch.inf).argmax(1)
        wrong[ok]=g.cuda()[alt[ok]]
        miss.append((~ok).cpu())
        rivals.append(wrong.cpu())
        for name,(a,b) in zip(KINDS,((local,whole),(local,vis),(whole,vis),(inc,vis)),strict=True):
            scores=measures(a,b,member,counts)
            covariance=scores["distinct"]-scores["same"]
            advantage=covariance[r,gold]-covariance[r,wrong]
            for mode,value in scores.items():
                signal=value[r,gold]-value[r,wrong]
                rank=1+(value>value[r,gold,None]).sum(1)
                compare=(signal>0).float()+.5*(signal==0)
                same_margin=scores["same"][r,gold]-scores["same"][r,wrong]
                changed=(signal>0).float()-(same_margin>0).float()
                record=torch.stack((value.argmax(1)==gold,1/rank,compare,signal,advantage,changed,
                                    value[r,gold],value[r,wrong]),1).cpu()
                results[name+"/"+mode].append(record)
            # Store independently interpretable class quantities for subsequent analysis.
            category_values[name].append(torch.stack([scores[k] for k in ("first","second","same","distinct")],2).cpu())
        err=max(err,max(float((x.mean(1)-.5).abs().max()) for x in (whole,local,inc,vis)))
    if err>2e-6:raise ValueError("Exact tie-aware ranks failed calibration")
    return {k:torch.cat(v) for k,v in results.items()},{k:torch.cat(v) for k,v in category_values.items()},torch.cat(miss),torch.cat(rivals),err


def summarize(observations,miss,labels):
    report={}
    for group,keep in (("all",torch.ones_like(miss)),("actual_top1_misses",miss),("actual_top1_hits",~miss)):
        sums={}
        for k,v in observations.items():
            t=v[keep].double()
            classid=labels[keep]
            nums=classid.bincount(minlength=int(labels.max())+1).double()
            drawn=torch.randint(len(nums),(4000,len(nums)),generator=torch.Generator().manual_seed(7401))
            mass=nums[drawn].sum(1).clamp_min(1)
            superiority=(t[:,4]>0).double()+.5*(t[:,4]==0)
            total=torch.zeros_like(nums).index_add_(0,classid,superiority)
            ci=(total[drawn].sum(1)/mass).quantile(torch.tensor([.025,.975],dtype=mass.dtype))
            sums[k]={"n":len(t),**dict(zip(FIELDS,t.mean(0).tolist())),
                     "P_distinct_helps_relative_to_same":float(superiority.mean()),"class_bootstrap_CI":ci.tolist(),
                     "inversions_repaired_vs_same":int((t[:,5]>0).sum()),"correct_lost_vs_same":int((t[:,5]<0).sum())}
        report[group]=sums
    return report


@torch.inference_mode()
def run(w):
    _,_,qs,_,gs,q,p,*_=semantic_inputs(w,"sketchy")
    data,report={},{}
    for s in qs:
        rows,cats,miss,rivals,err=run_split(s,qs,gs,q,p,w)
        y=qs[s][1]
        data[s]={"query_ids":torch.arange(len(y)),"query_labels":y,"fixed_rival":rivals,"actual_errors":miss,
                 "outcomes":rows,"oracle_by_category":cats}
        report[s]={"queries":len(y),"gallery":len(gs[s][1]),"actual_errors":int(miss.sum()),
                   "mean_rank_error":err,"groups":summarize(rows,miss,y)}
        print(s,int(miss.sum()),{k:round(v["P_distinct_helps_relative_to_same"],3) for k,v in report[s]["groups"]["actual_top1_misses"].items()
                                   if k.endswith("/distinct")},flush=True)
    dest=w/"results"/FAMILY/"oracle-witness-test.pt"
    save_tensor(data,dest)
    atomic_json(dest.with_suffix(".json"),{"revision":revision(),"scope":"posthoc only, true gallery category used, no resulting score deployable",
                  "fields":FIELDS,"class_score_columns":["first","second","same","distinct"],"splits":report,
                  "per_query_sha256":checksum(dest)})


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(cli.parse_args().workspace)
