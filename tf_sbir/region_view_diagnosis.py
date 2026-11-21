"""Diagnose which view-level evidence predicts useful reframing on the Sketchy pilot."""

import json
from pathlib import Path

import torch
from torch.nn.functional import log_softmax

from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import digest
from .foundation import load_words
from .map_diagnosis import map_values
from .regional_intervention_study import index, mixscore
from .regional_interventions import FAMILY, obtain
from .retrieval_diagnosis import global_inputs, stratified_sample
from .semantic import read_words


def ranks(x):
    below=(x[:,:,None]>x[:,None,:]).sum(-1)
    tied=(x[:,:,None]==x[:,None,:]).sum(-1)
    return below.float()+.5*(tied-1)


def logshare(a,b,logprior):
    return (a+b-logprior).logsumexp(-1)


@torch.inference_mode()
def run(w):
    cfg,paths,qs,pp,gs,qraw,praw=global_inputs(w,"sketchy")
    plan={s:stratified_sample(y,36,63241) for s,(_,y) in qs.items()}
    combined=torch.cat([qi[plan[s]] for s,(qi,_) in qs.items()]).unique(sorted=True)
    bank=obtain(w,"sketchy",cfg,paths,qraw,combined)
    words=read_words(cfg.vocabulary)
    _,scale=bare_words(cfg,words)
    src,tgt=(x.cuda() for x in load_words(cfg,words))
    folder=w/"results"/FAMILY
    signature=read(folder/"protocol.json")["sha256"]
    result={}
    for s,(qi,y) in qs.items():
        pi,g=gs[s]
        info=digest({"scorer":signature,"paths":digest([str(pp[i]) for i in pi.tolist()]),"words":checksum(cfg.vocabulary)})[:16]
        heads,b=index(w,s,praw[pi].cuda(),tgt,scale,info)
        text=src[heads]
        logm=b.mean(0).clamp_min(1e-30).log()
        ids=plan[s]
        base=torch.load(w/"cache/pragmatic-gallery-competition"/(s+"-named-rgb16.pt"),weights_only=True,mmap=True)
        measures={}
        correlations={}
        placeprior=torch.tensor([1/18]*9+[1/8]*4,device="cuda").log()
        for group in ids.split(56):
            qq=qraw[qi[group]].cuda()
            x=bank[torch.searchsorted(combined,qi[group])].float().cuda()
            z=x@text.T*scale
            keep,drop,zoom=z.unbind(2)
            full=qq@text.T*scale
            f,k,d,v=[log_softmax(t+logm,-1) for t in (full[:,None],keep,drop,zoom)]
            aligned=logshare(v,d,logm)
            agreements=logshare(v[:, :, None],d[:,None,:],logm)
            self_affinity=logshare(v,f,logm)
            gates={"equal":torch.zeros_like(aligned),"context":aligned,"kept":logshare(v,k,logm),
                   "full":self_affinity,"entropy":(v.exp()*v).sum(-1),
                   "drop_penalty":self_affinity-aligned,
                   "paired_context":aligned-(agreements.logsumexp(2)-agreements.new_tensor(13.).log()),
                   "peak":zoom.amax(-1)}
            labels=y[group].cuda()
            hist=base[group].cuda().clamp_min(1e-30).log()
            ap0=map_values(mixscore(full,b,hist,.4),labels,g.cuda(),s)[0]
            per=[]
            for i in range(13):
                per.append(map_values(mixscore(zoom[:,i],b,hist,.4),labels,g.cuda(),s)[0])
            per=torch.stack(per,1)
            measures.setdefault("native",[]).append(ap0.cpu())
            measures.setdefault("center_only",[]).append(per[:,4].cpu())
            winner=per[:,:,0].argmax(1)
            row=torch.arange(len(group),device="cuda")
            measures.setdefault("oracle_best_zoom_by_AP",[]).append(per[row,winner].cpu())
            ar=ranks(per[:,:,0])
            for name,raw in gates.items():
                if name!="equal":
                    gr=ranks(raw)
                    centered_g=gr-gr.mean(1,keepdim=True)
                    centered_a=ar-ar.mean(1,keepdim=True)
                    den=centered_g.norm(dim=1)*centered_a.norm(dim=1)
                    valid=den>0
                    spearman=(centered_g*centered_a).sum(1)[valid]/den[valid]
                    correlations.setdefault(name,[]).append(spearman.cpu())
                    measures.setdefault("hard_"+name,[]).append(per[row,raw.argmax(1)].cpu())
                for level in (1.,3.):
                    sw=log_softmax(level*raw+placeprior,1)
                    pooled=(zoom+sw[:,:,None]).logsumexp(1)
                    obs=map_values(mixscore(pooled,b,hist,.4),labels,g.cuda(),s)[0]
                    measures.setdefault(name+"_"+str(level),[]).append(obs.cpu())
        result[s]={"queries":len(ids),"values":{k:torch.cat(v).mean(0).tolist() for k,v in measures.items()},
                   "average_within_query_rank_correlation":{k:float(torch.cat(v).mean()) for k,v in correlations.items()},
                   "sample":"36/class seed 63241, same complete galleries, diagnostic top-K values from vectorized evaluator"}
        atomic_json(folder/"view-diagnosis.json",result)
        print(s,sorted(result[s]["values"].items(),key=lambda x:-x[1][0])[:8],flush=True)


if __name__=="__main__":
    run(Path("/content/sbir"))
