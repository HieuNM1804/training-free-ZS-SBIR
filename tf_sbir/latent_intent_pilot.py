"""Measure a closed-form ranking decision against controls with the same possible actions."""

import argparse
from collections import defaultdict
from itertools import product
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,checksum,read
from .crossdomain_benchmark import digest,revision
from .crossdomain_inputs import CHECKPOINT,runtime
from .encoding import save_tensor
from .foundation import load_words
from .intent_action_index import FAMILY,TEMPS
from .intent_decisions import ALPHAS,RULES,action_values,plan
from .intent_utility import exposure,verify_identity
from .map_diagnosis import map_values
from .open_name_diagnosis import FAMILY as PUBLIC
from .retrieval_diagnosis import stratified_sample
from .semantic import read_words
from .verification_inputs import get_inputs
from .word_conditional_verifier import zscore

SETTINGS=list(product((16,64),(.3,1.),TEMPS,(0.,.5,.85)))


def key(setting,rule,cap):
    return "/".join([*(f"{x:g}" for x in setting),rule,f"{cap:g}"])


def action_table(codes,base,photos,text,groups,y,g,split,alphas=ALPHAS,retain_ranks=False):
    n_words=len(text)
    results=[[],[],[],[]]
    if retain_ranks:
        results.append([])
    for block in codes.split(40):
        local=block//n_words
        word=block%n_words
        native=zscore(text[word]@photos.T)
        base_row=base[local]
        ranks=[]; margins=[]; outcomes=[]; hits=[]; retained=[]
        for a in alphas:
            score=(1-a)*base_row+a*native
            actual,ranking,correct,_=map_values(score,y[local],g,split)
            pair,first=exposure(groups[ranking],int(groups.max())+1)
            ranks.append(pair);margins.append(first);outcomes.append(actual);hits.append(correct[:,0])
            if retain_ranks:
                retained.append(ranking.int())
        tensors=(ranks,margins,outcomes,hits,retained) if retain_ranks else (ranks,margins,outcomes,hits)
        for pieces,tensor in zip(results,tensors):
            pieces.append(torch.stack(tensor,1))
    return [torch.cat(parts) for parts in results]


@torch.inference_mode()
def evaluate(split,data,qlabels,glabels,w,text,ids):
    root=w/"cache"/FAMILY
    lib=torch.load(root/(split+"-actions.pt"),weights_only=True,mmap=True)
    library={k:lib[k].cuda() for k in ("first","pair")}
    library["memberships"]={k:v.cuda() for k,v in lib["memberships"].items()}
    group=lib["groups"].cuda()
    photos=data["gallery_native"].cuda().float()
    gallery_labels=glabels.cuda()
    pre=torch.load(w/"cache"/PUBLIC/(split+"-proposals.pt"),weights_only=True,mmap=True)["whole"]
    rows,decision_records=defaultdict(list),defaultdict(list)
    for ix in ids.split(24):
        truth=qlabels[ix].cuda()
        base=data["regional_score"][ix].cuda()
        normalized=zscore(base)
        query_words=tuple(x[ix].cuda() for x in pre)
        models=plan(base,library,group,query_words,SETTINGS)
        all_candidates=[]
        local=torch.arange(len(base),device=base.device)
        for _,nominees in models.values():
            all_candidates.append((local[:,None]*len(text)+nominees).flatten())
        codes=torch.cat(all_candidates).unique(sorted=True)
        action_pair,action_first,actual,hits=action_table(codes,normalized,photos,text,group,truth,gallery_labels,split)
        base_metric,base_rank,_,_=map_values(base,truth,gallery_labels,split)
        base_exposure=exposure(group[base_rank],int(group.max())+1)
        rows["regional"].append(base_metric.cpu())
        for setting,(model,nominees) in models.items():
            lookup=torch.searchsorted(codes,(local[:,None]*len(text)+nominees).contiguous())
            decisions=action_values(model,lookup,(action_pair,action_first),base_exposure,ALPHAS)
            nm=nominees.size(1)
            # Only the observed outcomes of choices made above are selected here.
            observed=torch.cat((base_metric[:,None],actual[lookup].reshape(len(lookup),-1,2)),1)
            best=observed[:,:,0].argmax(1)
            rows[key(setting,"oracle",1.)].append(observed[local,best].cpu())
            w_ids=nominees[:,:,None].expand(-1,-1,len(ALPHAS)).reshape(len(base),-1)
            mix_values=base.new_tensor(ALPHAS).repeat(nm)
            for (rule,cap),(chosen,utility_gain) in decisions.items():
                label=key(setting,rule,cap)
                rows[label].append(observed[local,chosen].cpu())
                idx=(chosen-1).clamp_min(0)
                word=torch.where(chosen>0,w_ids[local,idx],-1)
                strength=torch.where(chosen>0,mix_values[idx],0)
                decision_records[label].append(torch.stack((word.float(),strength,utility_gain),1).cpu())
        logits,word_ids=query_words
        for n,t in ((n,t) for n in (16,64) for t in (.3,1.)):
            target=text[word_ids[:,:n]]
            weight=(logits[:,:n]*t).softmax(1)
            averaged=zscore((target*weight[:,:,None]).sum(1)@photos.T)
            first=zscore(target[:,0]@photos.T)
            for name,x in (("averaged",averaged),("first_name",first)):
                if name=="first_name" and (n,t)!=(16,.3):continue
                for a in ALPHAS:
                    k=f"{name}/{n}/{t:g}/{a:g}"
                    rows[k].append(map_values((1-a)*normalized+a*x,truth,gallery_labels,split)[0].cpu())
    return {k:torch.cat(v) for k,v in rows.items()},{k:torch.cat(v) for k,v in decision_records.items()}


@torch.inference_mode()
def run(w):
    root=w/"results"/FAMILY
    cfg=runtime(w,"sketchy",batch=384,workers=6)
    words=read_words(cfg.vocabulary)
    if len(words)!=20931:raise ValueError("Changed public vocabulary")
    text=load_words(cfg,words)[1].cuda()
    rev=revision()
    spec={"revision":rev,"checkpoint":CHECKPOINT,"settings":SETTINGS,"actions":ALPHAS,"decision_rules":RULES,
          "balanced_seed":63241,"per_class":36,"oracle":"only after selecting actions without labels",
          "utility":"expected full-gallery AP at any evaluation cutoff"}
    sig=digest(spec)
    if (root/"protocol.json").exists() and read(root/"protocol.json")["sha256"]!=sig:raise ValueError("Changed pilot")
    atomic_json(root/"protocol.json",spec|{"sha256":sig})
    table={}
    identity=verify_identity(text.device)
    for s,(data,y,g) in get_inputs(w,"sketchy").items():
        file=root/("pilot-"+s+".json")
        if file.exists():
            prev=read(file)
            if prev["signature"]!=sig or prev["per_query_sha256"]!=checksum(file.with_suffix(".pt")):raise ValueError("Invalid pilot artifact")
            table[s]=prev["values"];continue
        ids=stratified_sample(y,36,63241)
        rows,choices=evaluate(s,data,y,g,w,text,ids)
        vals={k:v.mean(0).tolist() for k,v in rows.items()}
        diags={}
        for k,d in choices.items():
            use=d[:,0]>=0
            diff=100*(rows[k]-rows["regional"])
            diags[k]={"adoption_share":float(use.float().mean()),"mean_alpha":float(d[:,1].mean()),
                      "mean_predicted_utility_gain":float(d[:,2].mean()),"adopted_change_pp":diff[use].mean(0).tolist() if use.any() else None}
        tensor=file.with_suffix(".pt")
        save_tensor({"metrics":rows,"choices":choices,"query_labels":y[ids],"split_ids":ids},tensor)
        atomic_json(file,{"signature":sig,"queries":len(ids),"gallery":len(g),"values":vals,"diagnostics":diags,
                          "finite_enumeration_error":identity,"per_query_sha256":checksum(tensor)})
        table[s]=vals
        primary=max((k for k in vals if "/joint/" in k),key=vals.get)
        print(s,primary,vals[primary],vals["regional"],flush=True)
    means={k:[sum(table[s][k][i] for s in table)/2 for i in (0,1)] for k in table["sketchy_1"]}
    atomic_json(root/"pilot.json",{"splits":table,"means":means})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
