"""Screen a reconstruction-based correction on complete candidate galleries."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .cluster_mine import cluster_assignments
from .concept_benchmark import atomic_json,checksum,read
from .conditional_semantic_index import SIZES,bare_words
from .crossdomain_benchmark import digest,revision,SPLIT_HASHES
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import load_frozen_clip,save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .regional_intervention_study import mixscore,index as compact_index
from .reciprocal_reassembly import local,matching_features,encode,correspond,reassemble,read_bank,adjust
from .retrieval_diagnosis import global_inputs,stratified_sample
from .semantic import read_words

FAMILY="reciprocal-layout-reassembly"
WEIGHTS=(.2,.7,2.,6.)


def input_bank(w,ds,role,paths):
    d=w/"cache/candidate-style-counterfactual"/ds
    info=read(d/(role+"-conv.json"))
    if info["checkpoint"]!=CHECKPOINT or info["paths_sha256"]!=digest([str(p) for p in paths]):
        raise ValueError("Patch cache differs from the benchmark order")
    return np.load(d/(role+"-conv.npy"),mmap_mode="r")


def layout_groups(photos):
    centered=F.normalize(photos-photos.mean(0))
    groups=cluster_assignments(centered,64)
    _,groups=groups.unique(sorted=True,return_inverse=True)
    count=int(groups.max())+1
    total=photos.new_zeros((count,photos.shape[1])).index_add_(0,groups,centered)
    response=centered@F.normalize(total).T
    hom=groups[:,None]==torch.arange(count,device=photos.device)
    representatives=response.masked_fill(~hom,-torch.inf).argmax(0)
    return groups,representatives


def choice(scores,groups,count=6):
    mass=scores.new_zeros((len(scores),int(groups.max())+1))
    score=(scores-scores.amax(1,keepdim=True)).exp()
    mass.index_add_(1,groups,score)
    mass=mass/groups.bincount().clamp_min(1)
    return mass.topk(count,dim=1).indices


def variant_scores(corners,cost):
    c={"cost":cost*100}
    for name in ("all","compact","dot"):
        a,b,d=[corners[name,j,"interaction"] for j in range(3)]
        c[name+"-reassembled"]=a-.5*(b+d)
        c[name+"-raw"]=a
        c[name+"-wrong"]=.5*(b+d)
        c[name+"-one"]=corners[name,0,"one"]-.5*(corners[name,1,"one"]+corners[name,2,"one"])
        c[name+"-aligned"]=corners[name,0,"aligned"]-.5*(corners[name,1,"aligned"]+corners[name,2,"aligned"])
    c["self-recovery"]=corners["self",0,"recovery"]-.5*(corners["self",1,"recovery"]+corners["self",2,"recovery"])
    return c


@torch.inference_mode()
def study(w):
    root=w/"results"/FAMILY
    spec={"revision":revision(),"checkpoint":CHECKPOINT,"seed":63241,"per_class":36,"groups":64,"candidates":6,"scales":WEIGHTS,
          "main":["all-reassembled","compact-reassembled","dot-reassembled"],"nomination":"mean regional retrieval mass per gallery group"}
    atomic_json(root/"protocol.json",spec|{"sha256":digest(spec)})
    cfg,paths,qs,pp,gs,qall,pall=global_inputs(w,"sketchy")
    words=read_words(cfg.vocabulary)
    if len(words)!=20931:raise ValueError("Unexpected dictionary")
    src,tgt=(x.cuda() for x in load_words(cfg,words));_,scale=bare_words(cfg,words)
    rq=input_bank(w,"sketchy","query",paths);rp=input_bank(w,"sketchy","gallery",pp)
    qzoom=torch.load(w/"cache/regional-evidence-interventions/sketchy/regions.pt",weights_only=True,mmap=True)
    v=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    table={}
    for s,(qi,y) in qs.items():
        pi,g=gs[s]
        if (len(qi),len(pi),len(g.unique()))!=SIZES[s] or digest(UNSEEN_CLASSES[s])!=SPLIT_HASHES[s]:raise ValueError("Wrong split")
        ids=stratified_sample(y,36,spec["seed"])
        dest=root/("pilot-"+s+".json")
        if dest.is_file():
            old=read(dest)
            if old["revision"]!=revision():raise ValueError("Different code")
            table[s]=old["values"];continue
        p=pall[pi].cuda()
        index_id=digest({"protocol":spec,"paths":digest([str(pp[i]) for i in pi.tolist()]),"words":checksum(cfg.vocabulary)})[:16]
        heads,b=compact_index(w,s,p,tgt,scale,index_id)
        groups,medoids=layout_groups(p)
        gen=torch.Generator().manual_seed(79121)
        ref=pi[torch.randperm(len(pi),generator=gen)[:256]]
        refmu=torch.stack([local(v,read_bank(rp,z)).sum(0) for z in ref.split(32)]).sum(0)/len(ref)
        med=read_bank(rp,pi[medoids.cpu()])
        meddesc=matching_features(local(v,med),refmu)
        medfeat=encode(v,med)
        native_err=float((1-(medfeat*p[medoids]).sum(1)).abs().max())
        base=torch.load(w/"cache/pragmatic-gallery-competition"/(s+"-named-rgb16.pt"),mmap=True,weights_only=True)
        result=defaultdict(list)
        availability=[]
        max_check=0.
        records=defaultdict(list)
        for it,group in enumerate(ids.split(12)):
            labels=y[group].cuda()
            query=qall[qi[group]].cuda()
            qraw=read_bank(rq,qi[group])
            qdesc=matching_features(local(v,qraw),refmu)
            trueq=encode(v,qraw)
            native_err=max(native_err,float((1-(trueq*query).sum(1)).abs().max()))
            if native_err>.002:raise ValueError("Original RGB embeddings differ from either token stream")
            zix=torch.searchsorted(qzoom["ids"],qi[group])
            if not torch.equal(qzoom["ids"][zix],qi[group]):raise ValueError("Missing regional base features")
            crops=qzoom["features"][zix,:,2].cuda().float()
            z=(crops@src[heads].T)*scale
            zoom=torch.logaddexp(z[:,:9].logsumexp(1)-z.new_tensor(9.).log(),z[:,9:].logsumexp(1)-z.new_tensor(4.).log())-z.new_tensor(2.).log()
            scores=mixscore(zoom,b,base[group].cuda().clamp_min(1e-30).log(),.4)
            nominations=choice(scores,groups)
            selected=nominations.flatten()
            qrep=torch.arange(len(group),device=qdesc.device).repeat_interleave(nominations.size(1))
            perm,cost=correspond(meddesc[selected],qdesc[qrep])
            corners,check=reassemble(v,qraw[qrep],med[selected],trueq[qrep],medfeat[selected],perm,src,tgt,scale,heads)
            max_check=max(max_check,check)
            vals=variant_scores(corners,cost)
            baseline=map_values(scores,labels,g.cuda(),s)[0]
            result["regional"].append(baseline.cpu())
            availability.append((g[medoids[nominations].cpu()]==labels.cpu()[:,None]).any(1))
            for name,d in vals.items():
                correction=d.view_as(nominations)
                records[name].append(correction.cpu())
                for weight in WEIGHTS:
                    score=adjust(scores,groups,nominations,correction,weight)
                    result[name+"@"+str(weight)].append(map_values(score,labels,g.cuda(),s)[0].cpu())
            records["nominated_groups"].append(nominations.cpu())
            if it%20==0:print(s,"scored",min((it+1)*12,len(ids)),"/",len(ids),flush=True)
        rows={k:torch.cat(vv) for k,vv in result.items()}
        per=root/("pilot-"+s+".pt")
        save_tensor({"metrics":rows,"labels":y[ids],"ids":ids,"medoid_indices":medoids.cpu(),"nominations":{k:torch.cat(val) for k,val in records.items()}},per)
        summary={k:v.mean(0).tolist() for k,v in rows.items()}
        note={"revision":revision(),"split":s,"queries":len(ids),"gallery":len(g),"values":summary,"actual_medoid_same_class_in_shortlist":float(torch.cat(availability).float().mean()),
              "native_feature_max_one_minus_cosine":native_err,"dot_four_term_formula_absolute_error":max_check,"per_query_sha256":checksum(per)}
        atomic_json(dest,note);table[s]=summary
        winner=max(summary,key=summary.get);print(s,winner,summary[winner],flush=True)
    means={k:[(table["sketchy_1"][k][j]+table["sketchy_2"][k][j])/2 for j in (0,1)] for k in table["sketchy_1"]}
    atomic_json(root/"pilot.json",{"splits":table,"means":means})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    study(p.parse_args().workspace)
