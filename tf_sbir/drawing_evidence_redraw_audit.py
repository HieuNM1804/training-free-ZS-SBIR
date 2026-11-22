"""Do margins from all completions of a single query predict how independently drawn sketches will rank the same two photos?"""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import revision
from .drawing_evidence_subsets import FAMILY,PARTITIONS,subset_bank
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .persistent_evidence_study import CHANNELS,SIZES,lattice,profile
from .recoverable_patch_study import prepare
from .semantic_neuron_gallery import FAMILY as PARENT


RANKS=(0,2,7,15,31,63,99,199,499,999,1999)
TERMS=("persist","excess","one_more","max","mean","shuffled","null_difference")


def panels(w,s,p):
    sibling_info=torch.load(w/"results"/PARENT/"choice-repeatability-full.pt",mmap=True,weights_only=True)[s]
    byid={q:(a,b) for q,a,b,_ in sibling_info["panels"]}
    use=[];pairs=[];same=[];other=[]
    pairtype=("relevant_vs_strongest_wrong","first_wrong_above_relevant",*[f"adjacent_to_{j+1}" for j in RANKS])
    original=torch.load(w/"cache/gallery-stable-region-evidence"/(s+"-regional-reference.pt"),mmap=True,weights_only=True)
    gl=p["g"]
    correct=[]
    for j,idx in enumerate(p["ids"].tolist()):
        if idx not in byid:continue
        siblings,controls=byid[idx]
        ord=p["base"][j].argsort(descending=True,stable=True)
        hit=(gl[ord]==p["y"][j])
        first_positive=int(torch.where(hit)[0][0]);first_negative=int(torch.where(~hit)[0][0])
        right=torch.where((~hit[:199])&hit[1:200])[0]
        ids=[[int(ord[first_positive]),int(ord[first_negative])],
             [int(ord[int(right[0])+1]),int(ord[int(right[0])])] if len(right) else [-1,-1],
             *[[int(ord[k]),int(ord[k+1])] for k in RANKS]]
        ps=torch.tensor(ids)
        for members,dest in ((siblings,same),(controls,other)):
            # Post-hoc only. Each reference preference is the empirical frequency of its original frozen ordering.
            a=original[members][:,ps[:,0].clamp_min(0)]
            b=original[members][:,ps[:,1].clamp_min(0)]
            prob=((a>b).float()+.5*(a==b)).mean(0)
            prob[ps[:,0]<0]=float("nan")
            dest.append(prob)
        use.append(j);pairs.append(ps);correct.append(bool(hit[0]))
    return (torch.tensor(use),torch.stack(pairs),torch.stack(same),torch.stack(other),torch.tensor(correct),pairtype)


def secondary_score(features,photos,wordbank,axes,scale,channel):
    # Per-query photo candidate indices are fixed before the intervention.
    if channel=="rgb":return scale*torch.einsum("bsd,bjd->bsj",features,photos)
    logits=scale*(features@axes.T)
    mass=(logits-logits.amax(2,keepdim=True)).exp()
    return torch.bmm(mass,wordbank.transpose(1,2)).clamp_min(1e-30).log()


def agreement(v,target):
    choice=(v>0).float()+.5*(v==0)
    return choice*target+(1-choice)*(1-target)


def summary_for(v,ref,other,mask,base):
    if not mask.any():return {}
    a=agreement(v,ref)[mask]
    b=agreement(base,ref)[mask]
    c=agreement(v,other)[mask]
    d=agreement(base,other)[mask]
    return {"n":int(mask.sum()),"sibling_rank_agreement":float(a.mean()),"control_object_rank_agreement":float(c.mean()),
            "change_from_full_token_similarity":float((a-b).mean()),"sibling_specific_excess":float(((a-b)-(c-d)).mean()),
            "fraction_positive_margin":float(((v[mask]>0).float()+.5*(v[mask]==0)).mean())}


@torch.inference_mode()
def run(w):
    inputs=semantic_inputs(w,"sketchy")
    torch.backends.cuda.matmul.allow_tf32=False
    d=subset_bank(w,"sketchy","pilot-query")
    sets,perms=lattice("cuda")
    root=w/"results"/FAMILY
    data,report={},{}
    for s in inputs[2]:
        p=prepare(w,s,inputs,d)
        use,pairs,sibling,unrelated,hit,names=panels(w,s,p)
        records=defaultdict(list)
        for start in range(0,len(use),24):
            sl=slice(start,start+24)
            j=use[sl]
            observed=d["subsets"][p["positions"][j]].cuda().float()
            ids=pairs[sl].clamp_min(0).flatten(1).cuda()
            rgb=p["rgb"][ids];words=p["bank"][ids]
            for ix,part in enumerate(PARTITIONS):
                for ch in CHANNELS:
                    val=secondary_score(observed[:,ix],rgb,words,p["axes"],p["scale"],ch)
                    val=val-val[:,:1]
                    delta=val[:,:,::2]-val[:,:,1::2]
                    route=f"{part}/{ch}"
                    records[route+"/full"].append(delta[:,-1].cpu())
                    pos=profile(delta,sets,perms)
                    neg=profile(-delta,sets,perms)
                    # Opposing comparisons share exactly the same history of revealed image patches.
                    # Pairing each candidate with an unrelated subset of equal size is a stronger null.
                    perm=perms[0]
                    independent=val[:,:,::2]-val[:,perm,1::2]
                    reversed_independent=val[:,:,1::2]-val[:,perm,::2]
                    alt1=profile(independent,sets,perms);alt2=profile(reversed_independent,sets,perms)
                    single=profile(val,sets,perms)
                    for size in SIZES:
                        for m in TERMS:
                            desc=f"{route}/{size}/{m}"
                            records[desc+"/paired"].append((.5*(pos[size,m]-neg[size,m])).cpu())
                            sv=single[size,m]
                            records[desc+"/subtract_scalars"].append((sv[:,::2]-sv[:,1::2]).cpu())
                            if m in ("persist","excess"):
                                records[desc+"/wrong_pairing"].append((.5*(alt1[size,m]-alt2[size,m])).cpu())
        vals={k:torch.cat(v) for k,v in records.items()}
        r={}
        for key,value in vals.items():
            channel="/".join(key.split("/")[:2])
            full=vals[channel+"/full"]
            first=torch.zeros_like(pairs[:,:,0],dtype=torch.bool);first[:,0]=~hit
            adjacent=torch.zeros_like(first);adjacent[:,1]=pairs[:,1,0]>=0
            nearby=torch.zeros_like(first);nearby[:,2:]=True
            r[key]={m:summary_for(value,sibling,unrelated,flag,full) for m,flag in
                    (("actual_top1_errors",first),("first_adjacent_retrieval_error",adjacent),("unlabelled_nearby_pairs",nearby))}
        data[s]={"local_queries":use,"photos":pairs,"reference_branch":"regional","same_object":sibling,"matched_other_same_class":unrelated,
                 "base_top1_correct":hit,"pair_names":names,"candidate_margins":vals}
        report[s]={"queries":len(use),"base_errors":int((~hit).sum()),"results":r,"pair_names":names}
        print("independent drawings",s,len(use),flush=True)
        del p
    file=root/"independent-drawing-margins.pt"
    save_tensor(data,file)
    atomic_json(file.with_suffix(".json"),{"revision":revision(),"per_query_sha256":checksum(file),"splits":report,
                "design":"Original score uses one sketch; multiple drawings of the same photographed object provide only retrospective test targets. Alternative source objects match class, number of reference drawings and baseline AP. Margins use the same frozen whole-image scorer for every partial observation.",
                "device":torch.cuda.get_device_name()})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
