"""Measure whether paired regional interventions improve retrieval over the same zooms."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES, bare_words
from .crossdomain_benchmark import digest, revision, SPLIT_HASHES
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .regional_interventions import FAMILY, obtain, regional_logits, null_check
from .retrieval_diagnosis import global_inputs, stratified_sample
from .retrieval_evidence import load_kd, paired_delta
from .selective_concept_listener import compile_index
from .semantic import read_words
from .shared_concept_likelihood import fit_prior

SPLITS=("sketchy_1","sketchy_2")


def key(cell):
    return json.dumps(cell,separators=(",",":"))


def grid():
    joint=[("joint",rho,power,t,mix) for rho in (0.,.35,.75,1.) for power in (0.,.7,2.)
           if rho or power for t in (.5,1.) for mix in (.4,.8)]
    simpler=[(kind,rho,0.,1.,mix) for kind in ("zoom","keep","drop") for rho in (.35,.75,1.) for mix in (.4,.8)]
    return joint+simpler+[("native",0.,0.,1.,mix) for mix in (.4,.8)]


def best(means, kinds, count):
    names=sorted((k for k in means if json.loads(k)[0] in kinds), key=means.get, reverse=True)
    return [tuple(json.loads(k)) for k in names[:count]]


def comparisons(selected, simpler, native):
    _,rho,power,t,mix=selected
    return sorted({selected, simpler, native, ("shift",rho,power,t,mix), ("joint",rho,0.,t,mix),
                   ("keep",rho,0.,t,mix),("drop",rho,0.,t,mix),("zoom",rho,0.,t,mix),
                   ("native",0.,0.,1.,mix),("native",0.,0.,1.,0.)})


def gallery_word_prior(p,target,scale):
    em=p.new_empty((len(p),len(target)))
    for start in range(0,len(p),256):
        sl=slice(start,start+256)
        em[sl]=((p[sl]@target.T)*scale).softmax(1)
    return fit_prior(em)[0]["mle"]


def index(w,split,p,target,scale,info):
    folder=w/"cache"/FAMILY
    file=folder/(split+"-index-"+info+".pt")
    if file.is_file():
        x=torch.load(file,weights_only=True,mmap=True)
        return x["heads"].cuda(),x["weights"].cuda()
    prior=gallery_word_prior(p,target,scale)
    heads,weights,_=compile_index(p,target,scale,prior,64,"visual_rgb")
    save_tensor({"heads":heads.cpu(),"weights":weights.cpu()},file)
    return heads,weights


def mixscore(word_logits, weights, base, mix):
    if mix==0:
        return base
    code=(word_logits-word_logits.amax(1,keepdim=True)).exp()
    compact=(code@weights.T).clamp_min(1e-30).log()
    return mix*compact+(1-mix)*base


def measure(q,views,b,src,scale,historical,y,g,indices,cells,split,kd,workers,primary):
    sums={key(c):[] for c in cells}
    errors={}
    top=[]
    oracles=[]
    orig=[]
    replay=Replay(kd,g,split) if kd else None
    blocks=sorted(set(c[:4] for c in cells))
    context=ThreadPoolExecutor(max_workers=workers) if kd else nullcontext()
    eq=null_check(views[:4].cuda(),src,scale)
    with context as pool:
        for start in range(0,len(indices),64):
            group=indices[start:start+64]
            v=views[start:start+len(group)].cuda().float()
            qq=q[group].cuda()
            yy=y[group]
            hist=historical[group].cuda().clamp_min(1e-30).log()
            for node in blocks:
                logword=regional_logits(qq,v,src,scale,*node)
                for cell in (c for c in cells if c[:4]==node):
                    scores=mixscore(logword,b,hist,cell[-1])
                    ap,order,_,_=map_values(scores,yy.cuda(),g.cuda(),split)
                    name=key(cell)
                    if replay:
                        exact=replay.score(order,yy,pool)
                        delta=float((ap.cpu()-exact).abs().max())
                        if delta>2e-6:raise ValueError("Mismatch with original KD metrics")
                        errors[name]=max(errors.get(name,0.),delta)
                        if primary==name:top.append(order[:,:25].int().cpu())
                    else:exact=ap.cpu()
                    sums[name].append(exact)
            if kd is None and len(indices)<=36*30:
                logits=(qq@src.T)*scale
                standard=map_values(mixscore(logits,b,hist,.4),yy.cuda(),g.cuda(),split)[0][:,0]
                best=standard.clone()
                for i in range(v.size(1)):
                    val=(v[:,i,2]@src.T)*scale
                    ap=map_values(mixscore(val,b,hist,.4),yy.cuda(),g.cuda(),split)[0][:,0]
                    best=torch.maximum(best,ap)
                oracles.append(best.cpu());orig.append(standard.cpu())
    diagnostics={"max_shift_null_error":eq}
    if oracles:
        ceiling=torch.cat(oracles)-torch.cat(orig)
        diagnostics.update({"oracle_mean_gain_ap_pp":float(100*ceiling.mean()),
                            "queries_with_any_better_zoom":float((ceiling>0).float().mean())})
    return {k:torch.cat(v) for k,v in sums.items()},errors,torch.cat(top) if top else None,diagnostics


@torch.inference_mode()
def run(w,phase,workers):
    folder=w/"results"/FAMILY
    spec={"revision":revision(),"grid":grid(),"pilot_queries_per_class":36,"pilot_seed":63241,
          "primary":"joint","positive_control":"best accuracy among registered ordinary and one-sided alternatives"}
    signature=digest(spec)
    if (folder/"protocol.json").is_file():
        if read(folder/"protocol.json")["sha256"]!=signature:raise ValueError("Different study in output directory")
    else:atomic_json(folder/"protocol.json",spec|{"sha256":signature})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Changed KD source")
    stages={"pilot":("pilot",),"lock":("shortlist",),"audit":("final",),"all":("pilot","shortlist","final")}[phase]
    for stage in stages:
        if stage=="pilot":cells=grid()
        elif stage=="shortlist":
            means=read(folder/"pilot.json")["means"]
            cells=sorted(set(best(means,{"joint"},3)+best(means,{"native","zoom","keep","drop"},3)+best(means,{"native"},1)))
        else:
            lock=read(folder/"freeze.json")
            if lock["full_sketchy_sha256"]!=checksum(folder/"shortlist.json") or lock["revision"]!=revision():
                raise ValueError("Invalid Sketchy selection")
            selected,simpler,native=(tuple(lock[k]) for k in ("primary","simple","native"))
            cells=comparisons(selected,simpler,native)
        table={}
        for ds in (("sketchy","tuberlin") if stage=="final" else ("sketchy",)):
            cfg,paths,qs,pp,gs,qraw,praw=global_inputs(w,ds)
            plan={s:(stratified_sample(y,36,63241) if stage=="pilot" else torch.arange(len(y))) for s,(qi,y) in qs.items()}
            combined=torch.cat([qi[plan[s]] for s,(qi,_) in qs.items()]).unique(sorted=True)
            feats=obtain(w,ds,cfg,paths,qraw,combined)
            words=read_words(cfg.vocabulary)
            if len(words)!=20931:raise ValueError("Wrong public vocabulary")
            _,scale=bare_words(cfg,words)
            src,tgt=(v.cuda() for v in load_words(cfg,words))
            for s,(qi,y) in qs.items():
                pi,g=gs[s]
                if (len(qi),len(pi),len(g.unique()))!=SIZES[s] or digest(UNSEEN_CLASSES[s])!=SPLIT_HASHES[s]:
                    raise ValueError("Incorrect dataset geometry")
                out=folder/(s+".json" if stage=="final" else stage+"-"+s+".json")
                if out.exists():
                    rec=read(out)
                    if rec["signature"]!=signature:raise ValueError("Checkpoint mismatch")
                    if stage=="final" and rec["per_query_sha256"]!=checksum(folder/(s+".pt")):raise ValueError("Missing query evidence")
                    table[s]=rec["values"]
                    continue
                ids=plan[s]
                loc=torch.searchsorted(combined,qi[ids])
                if not torch.equal(combined[loc],qi[ids]):raise ValueError("Misaligned view order")
                info=digest({"scorer":signature,"paths":digest([str(pp[i]) for i in pi.tolist()]),"words":checksum(cfg.vocabulary)})[:16]
                heads,b=index(w,s,praw[pi].cuda(),tgt,scale,info)
                base=torch.load(w/"cache/pragmatic-gallery-competition"/(s+"-named-rgb16.pt"),weights_only=True,mmap=True)
                if base.shape!=(len(qi),len(pi)):raise ValueError("Wrong historical score geometry")
                rows,err,tops,diag=measure(qraw[qi],feats[loc],b,src[heads],scale,base,y,g,ids,cells,s,
                                         kd if stage=="final" else None,workers,key(selected) if stage=="final" else None)
                vals={k:v.mean(0).tolist() for k,v in rows.items()}
                rec={"signature":signature,"values":vals,"queries":len(ids),"gallery":len(pi),"diagnostics":diag}
                if stage=="final":
                    path=folder/(s+".pt")
                    save_tensor({"metrics":rows,"query_labels":y,"top25":tops},path)
                    rec.update({"protocol":{"split":s,"queries":len(qi),"gallery":len(pi),"classes":len(g.unique()),
                                            "checkpoint":CHECKPOINT,"vocabulary_sha256":checksum(cfg.vocabulary),"kd_sha256":KD_FUNCTION_SHA,
                                            "gpu":torch.cuda.get_device_name(),"revision":revision(),"p_k":200 if s=="sketchy_2" else 100,
                                            "map_k":200 if s=="sketchy_2" else 0},
                                "scores":{k:dict(zip(("mAP","precision"),v)) for k,v in vals.items()},
                                "primary":key(selected),"best_contrast":key(simpler),"matched_base":key(native),
                                "baseline":key(("native",0.,0.,1.,0.)),
                                "vs_controls":{k:paired_delta(rows[key(selected)],v,y) for k,v in rows.items()},
                                "per_query_sha256":checksum(path),"kd_max_abs_error":err,"original_kd_queries_per_condition":len(qi)})
                atomic_json(out,rec);table[s]=vals
                win=max(vals,key=vals.get);print(stage,s,win,vals[win],diag,flush=True)
        if stage!="final":
            means={k:[sum(table[s][k][j] for s in SPLITS)/2 for j in (0,1)] for k in table[SPLITS[0]]}
            atomic_json(folder/(stage+".json"),{"splits":table,"means":means})
            if stage=="shortlist":
                choose={"primary":best(means,{"joint"},1)[0],"simple":best(means,{"native","zoom","keep","drop"},1)[0],
                        "native":best(means,{"native"},1)[0],"revision":revision(),"full_sketchy_sha256":checksum(folder/"shortlist.json")}
                atomic_json(folder/"freeze.json",choose|{"sha256":digest(choose)})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--phase",choices=("pilot","lock","audit","all"),default="all")
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    run(args.workspace,args.phase,args.workers)
