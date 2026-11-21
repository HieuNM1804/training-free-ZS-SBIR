"""Measure query-conditioned gallery rendering on two complete Sketchy galleries."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,checksum,read
from .conditional_semantic_index import SIZES,bare_words
from .crossdomain_benchmark import SPLIT_HASHES,digest,revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import load_frozen_clip,save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .query_observation import helmert,transport,diagonal,read_layouts,affinities,changes,check_moments
from .reciprocal_reassembly import read_bank,encode,adjust
from .reciprocal_reassembly_study import layout_groups,choice,input_bank
from .regional_intervention_study import index as compact_index,mixscore
from .retrieval_diagnosis import global_inputs,stratified_sample
from .semantic import read_words

FAMILY="query-conditioned-observation"
STRENGTH=(.04,.12,.35,1.)


@torch.inference_mode()
def run(w):
    root=w/"results"/FAMILY
    protocol={"revision":revision(),"checkpoint":CHECKPOINT,"pilot":36,"seed":63241,
              "groups":64,"candidates":6,"weights":STRENGTH,"primary":"exact covariance transfer, layout difference-in-differences",
              "ablation":"native, shared gallery medoid donor, channelwise variance, midpoint of the moment path, raw transferred affinity"}
    atomic_json(root/"protocol.json",protocol|{"sha256":digest(protocol)})
    cfg,paths,qs,pp,gs,qall,pall=global_inputs(w,"sketchy")
    qpatch=input_bank(w,"sketchy","query",paths)
    ppatch=input_bank(w,"sketchy","gallery",pp)
    raw_zoom=torch.load(w/"cache/regional-evidence-interventions/sketchy/regions.pt",mmap=True,weights_only=True)
    words=read_words(cfg.vocabulary)
    if len(words)!=20931:raise ValueError("Unexpected word bank")
    source,target=(v.cuda() for v in load_words(cfg,words));_,scale=bare_words(cfg,words)
    v=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    h=helmert(49,"cuda")
    gen=torch.Generator().manual_seed(22709)
    permutations=[torch.randperm(49,generator=gen).cuda() for _ in range(2)]
    table={}
    for s,(qi,y) in qs.items():
        pi,g=gs[s]
        if (len(qi),len(pi),len(g.unique()))!=SIZES[s] or digest(UNSEEN_CLASSES[s])!=SPLIT_HASHES[s]:
            raise ValueError("Unexpected split")
        dest=root/("pilot-"+s+".json")
        if dest.exists():
            old=read(dest)
            if old["revision"]!=revision():raise ValueError("Previous screen used different code")
            table[s]=old["values"];continue
        ids=stratified_sample(y,36,63241)
        p=pall[pi].cuda()
        code=digest({"protocol":protocol,"gallery":digest([str(pp[i]) for i in pi.tolist()]),"vocabulary":checksum(cfg.vocabulary)})[:16]
        heads,compact=compact_index(w,s,p,target,scale,code)
        group_id,reps=layout_groups(p)
        med=read_bank(ppatch,pi[reps.cpu()])
        base_views=read_layouts(v,med,permutations)
        native_error=float((1-(base_views[0]*p[reps]).sum(1)).abs().max())
        central=(p@p.mean(0)).argmax().item()
        commonref=read_bank(ppatch,pi[central:central+1]).expand_as(med)
        common,_=transport(commonref,med,h)
        common_views=read_layouts(v,common,permutations)
        base=torch.load(w/"cache/pragmatic-gallery-competition"/(s+"-named-rgb16.pt"),mmap=True,weights_only=True)
        buffer=defaultdict(list)
        notes={}
        record=defaultdict(list)
        for step,part in enumerate(ids.split(12)):
            labels=y[part].cuda()
            q=qall[qi[part]].cuda()
            qr=read_bank(qpatch,qi[part])
            trueq=encode(v,qr)
            native_error=max(native_error,float((1-(q*trueq).sum(1)).abs().max()))
            if native_error>.002:raise ValueError("Native patch reconstruction does not match the frozen encoder")
            loc=torch.searchsorted(raw_zoom["ids"],qi[part])
            if not torch.equal(raw_zoom["ids"][loc],qi[part]):raise ValueError("Unknown query")
            zv=raw_zoom["features"][loc,:,2].cuda().float()
            z=(zv@source[heads].T)*scale
            likelihood=torch.logaddexp(z[:,:9].logsumexp(1)-z.new_tensor(9.).log(),z[:,9:].logsumexp(1)-z.new_tensor(4.).log())-z.new_tensor(2.).log()
            original=mixscore(likelihood,compact,base[part].cuda().clamp_min(1e-30).log(),.4)
            nominated=choice(original,group_id)
            pairs=nominated.flatten(); n=nominated.size(1)
            repeated=torch.arange(len(part),device=p.device).repeat_interleave(n)
            xx=qr[repeated]; yy=med[pairs]
            exact,shared=transport(xx,yy,h)
            if step%15==0:
                validation=check_moments(xx,yy,exact,h)
                notes={k:max(notes.get(k,0.),v) for k,v in validation.items()}
            native=affinities(trueq[repeated],[view[pairs] for view in base_views],source,target,scale,heads)
            shared_score=affinities(trueq[repeated],[view[pairs] for view in common_views],source,source,scale,heads)
            cams={"covariance-affinity":shared*scale}
            for name,readout in (("exact",read_layouts(v,exact,permutations)),
                                  ("midpoint",read_layouts(v,(exact+yy)*.5,permutations)),
                                  ("diagonal",read_layouts(v,diagonal(xx,yy),permutations)),
                                  ("common",None)):
                obs=shared_score if name=="common" else affinities(trueq[repeated],readout,source,source,scale,heads)
                for term,values in changes(native,obs).items():
                    if term.endswith("-native"):
                        cams[term]=values
                    else:
                        cams[name+"-"+term]=values
            buffer["regional"].append(map_values(original,labels,g.cuda(),s)[0].cpu())
            for name,values in cams.items():
                delta=values.view_as(nominated)
                record[name].append(delta.cpu())
                for k in STRENGTH:
                    pred=adjust(original,group_id,nominated,delta,k)
                    result=map_values(pred,labels,g.cuda(),s)[0]
                    buffer[name+"@"+str(k)].append(result.cpu())
            record["nominations"].append(nominated.cpu())
            if step%16==0:print(s,"queries",min((step+1)*12,len(ids)),len(ids),flush=True)
        rows={k:torch.cat(seq) for k,seq in buffer.items()}
        out=root/("pilot-"+s+".pt")
        save_tensor({"metrics":rows,"ids":ids,"labels":y[ids],"medoid_ids":reps.cpu(),
                     "responses":{k:torch.cat(seq) for k,seq in record.items()}},out)
        vals={k:vv.mean(0).tolist() for k,vv in rows.items()}
        atomic_json(dest,{"revision":revision(),"queries":len(ids),"gallery":len(g),"values":vals,"checks":notes,
                          "native_reconstruction_error":native_error,"per_query_sha256":checksum(out)})
        table[s]=vals
        win=max(vals,key=vals.get);print(s,win,vals[win],flush=True)
    m={k:[(table["sketchy_1"][k][i]+table["sketchy_2"][k][i])/2 for i in (0,1)] for k in table["sketchy_1"]}
    atomic_json(root/"pilot.json",{"splits":table,"means":m})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
