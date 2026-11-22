"""Test held-out gallery verification of token-space lexical interpolation."""

import argparse
import math
import json
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,checksum,read
from .conditional_semantic_index import SIZES,bare_words
from .crossdomain_benchmark import SPLIT_HASHES,digest,revision
from .crossdomain_inputs import CHECKPOINT
from .cross_view_region_cache import FAMILY as VIEWS
from .data import UNSEEN_CLASSES
from .encoding import load_frozen_clip,save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .regional_intervention_study import index as compact_index,mixscore
from .retrieval_diagnosis import global_inputs,stratified_sample
from .semantic import read_words
from .verified_text_spans import ObjectSpans,prepare

FAMILY="gallery-verified-text-interpolation"


def key(c):return json.dumps(c,separators=(",",":"))


@torch.inference_mode()
def run(w):
    root=w/"results"/FAMILY
    cfg,_,qs,pp,gs,qraw,praw=global_inputs(w,"sketchy")
    words=read_words(cfg.vocabulary)
    if len(words)!=20931:raise ValueError("Changed public dictionary")
    src,tgt=(a.cuda() for a in load_words(cfg,words));_,scale=bare_words(cfg,words)
    model=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda"))
    spans=ObjectSpans(model,words)
    qviews=torch.load(w/"cache"/VIEWS/"sketchy/views.pt",mmap=True,weights_only=True)["original_regions"]
    rev=revision()
    spec={"revision":rev,"checkpoint":CHECKPOINT,"seed":63241,"pilot_per_class":36,"groups":64,
          "top_names":[8,24],"primary":"verified","controls":["hard","output","always","local","reverse","shared","verified_output"],
          "endpoint":"full gallery, category rank metrics on two balanced Sketchy pilots"}
    atomic_json(root/"protocol.json",spec|{"sha256":digest(spec)})
    table={}
    for s,(qi,y) in qs.items():
        pi,g=gs[s]
        if (len(qi),len(pi),len(g.unique()))!=SIZES[s] or digest(UNSEEN_CLASSES[s])!=SPLIT_HASHES[s]:raise ValueError("Wrong split")
        fn=root/("pilot-"+s+".json")
        if fn.exists():
            old=read(fn)
            if old["revision"]!=rev:raise ValueError("Changed study")
            table[s]=old["values"];continue
        photo=praw[pi].cuda()
        heads,small=compact_index(w,s,photo,tgt,scale,digest({"study":rev,"gallery":digest([str(pp[i]) for i in pi.tolist()])})[:16])
        hist=torch.load(w/"cache/pragmatic-gallery-competition"/(s+"-named-rgb16.pt"),mmap=True,weights_only=True)
        ids=stratified_sample(y,36,63241)
        rows={};diagnostics={}
        for k in (8,24):
            group,panels=prepare(photo,tgt,src,spans,scale,k)
            diagnostics[str(k)]=[p.diagnostics for p in panels]
            for part in ids.split(60):
                label=y[part].cuda()
                q=qraw[qi[part]].cuda()
                zoom=qviews[qi[part]].cuda().float()
                z=zoom@src[heads].T*scale
                local=torch.logaddexp(z[:,:9].logsumexp(1)-math.log(9),z[:,9:].logsumexp(1)-math.log(4))-math.log(2)
                base=mixscore(local,small,hist[part].cuda().clamp_min(1e-30).log(),.4)
                if k==8:rows.setdefault("regional",[]).append(map_values(base,label,g.cuda(),s)[0].cpu())
                for pose,x in (("whole",q[:,None]),("zoom",zoom)):
                    obs=[p.responses(x,src,scale) for p in panels]
                    def combine(which,t):
                        return torch.logaddexp(*(p.groups(e,which,t) for p,e in zip(panels,obs)))-math.log(2)
                    hard=combine("hard",1.)
                    for method in ("hard","output","always","verified","local","reverse","shared","verified_output"):
                        for t in ((1.,4.) if method in ("verified","local","reverse","verified_output") else (1.,)):
                            answer=combine(method,t)
                            for rule,weight in (("group",.2),("group",.6),("increment",.5),("increment",2.)):
                                if method=="hard" and rule=="increment":continue
                                names=(k,pose,method,t,rule,weight)
                                score=(1-weight)*base+weight*answer[:,group] if rule=="group" else base+weight*(answer-hard)[:,group]
                                rows.setdefault(key(names),[]).append(map_values(score,label,g.cuda(),s)[0].cpu())
            print(s,k,diagnostics[str(k)],flush=True)
        rows={k:torch.cat(v) for k,v in rows.items()}
        vals={k:v.mean(0).tolist() for k,v in rows.items()}
        path=root/("pilot-"+s+".pt")
        save_tensor({"metrics":rows,"query_labels":y[ids],"query_ids":ids},path)
        atomic_json(fn,{"revision":rev,"queries":len(ids),"gallery":len(g),"values":vals,"verification":diagnostics,
                        "per_query_sha256":checksum(path)})
        table[s]=vals
        leader=max(vals,key=vals.get);print(s,leader,vals[leader],flush=True)
    m={k:[(table["sketchy_1"][k][j]+table["sketchy_2"][k][j])/2 for j in (0,1)] for k in table["sketchy_1"]}
    atomic_json(root/"pilot.json",{"means":m,"splits":table})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
