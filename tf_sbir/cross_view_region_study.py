"""Score candidate-conditioned held-out region selection against equal-cost controls."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,checksum,read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES,bare_words
from .crossdomain_benchmark import SPLIT_HASHES,digest,revision
from .crossdomain_inputs import CHECKPOINT
from .cross_view_region_cache import FAMILY
from .cross_view_regions import CandidateRegions
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .regional_intervention_study import index as compact_index
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import load_kd,paired_delta
from .semantic import read_words


def key(c):return json.dumps(c,separators=(",",":"))


def grid():
    readouts=[(m,t,b) for m in ("cross","self") for t in (.25,1.,3.) for b in ("gallery","raw")]
    readouts += [("ratio",t,"raw") for t in (.25,1.,3.)]+[(m,0.,"raw") for m in ("uniform","native")]
    return [(*r,g,a) for r in readouts for g in ("first","both") for a in (.35,.65,.9)]


def select(means,kinds,n):
    return [tuple(json.loads(k)) for k in sorted((k for k in means if json.loads(k)[0] in kinds),key=means.get,reverse=True)[:n]]


def eval_split(q,views,reverse,mirrored,src,heads,small,full_index,base,y,g,ids,cells,scale,split,ref,workers,primary):
    out={key(c):[] for c in cells};err={};tops=[];max_null=0.
    context=ThreadPoolExecutor(max_workers=workers) if ref else nullcontext()
    parts=sorted(set(c[:3] for c in cells))
    with context as ex:
        for off in range(0,len(ids),32):
            part=ids[off:off+32]
            hist=base[part].cuda()
            batch=CandidateRegions(q[part].cuda(),mirrored[off:off+len(part)].cuda(),views[off:off+len(part)].cuda(),
                                   reverse[off:off+len(part)].cuda(),src,heads,small,full_index,hist)
            batch.prepare(scale)
            if off==0:max_null=batch.check()
            names=y[part]
            for r in parts:
                reg=batch.read(*r)
                for c in (c for c in cells if c[:3]==r):
                    global_=batch.both if c[3]=="both" else batch.first
                    scores=c[4]*reg+(1-c[4])*global_
                    approx,rank,_,_=map_values(scores,names.cuda(),g.cuda(),split)
                    nm=key(c)
                    if ref:
                        actual=ref.score(rank,names,ex)
                        error=float((approx.cpu()-actual).abs().max())
                        if error>2e-6:raise ValueError("Original KD scores differ")
                        err[nm]=max(err.get(nm,0.),error)
                        if nm==primary:tops.append(rank[:,:25].cpu().int())
                    else:actual=approx.cpu()
                    out[nm].append(actual)
    return {k:torch.cat(v) for k,v in out.items()},err,torch.cat(tops) if tops else None,max_null


@torch.inference_mode()
def run(w,phase,workers):
    root=w/"results"/FAMILY
    rev=revision()
    spec={"revision":rev,"grid":grid(),"primary_family":["cross","ratio"],"controls":["self","uniform","native"],
          "seed":63241,"pilot_sketches_per_class":36,"selection":"mean full Sketchy mAP, then mean precision"}
    sig=digest(spec)
    if (root/"protocol.json").is_file():
        if read(root/"protocol.json")["sha256"]!=sig:raise ValueError("Incompatible experiment directory")
    else:atomic_json(root/"protocol.json",spec|{"sha256":sig})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Changed evaluator")
    stages={"pilot":("pilot",),"lock":("shortlist",),"audit":("final",),"all":("pilot","shortlist","final")}[phase]
    for stage in stages:
        if stage=="pilot":cells=grid()
        elif stage=="shortlist":
            m=read(root/"pilot.json")["means"]
            cells=sorted(set(select(m,{"cross"},3)+select(m,{"ratio"},2)+select(m,{"self"},2)+select(m,{"uniform","native"},3)))
        else:
            lock=read(root/"freeze.json")
            if lock["full_sketchy_sha256"]!=checksum(root/"shortlist.json") or lock["revision"]!=rev:raise ValueError("Invalid freeze")
            selected,self_,uniform=(tuple(lock[z]) for z in ("primary","self","uniform"))
            mode,t,n,glob,mix=selected
            cells=sorted({selected,self_,uniform,("native",0.,"raw","first",.4),("native",0.,"raw","first",0.),
                          ("uniform",0.,"raw",glob,mix),("native",0.,"raw",glob,mix),
                          *[(k,t,n,glob,mix) for k in ("cross","self","wrong","ratio","ratio_wrong","opposed","global_selector")]})
        table={}
        for ds in (("sketchy","tuberlin") if stage=="final" else ("sketchy",)):
            cfg,qp,qs,pp,gs,qraw,praw=global_inputs(w,ds)
            words=read_words(cfg.vocabulary)
            if len(words)!=20931:raise ValueError("Different vocabulary")
            src,tgt=(a.cuda() for a in load_words(cfg,words));_,scale=bare_words(cfg,words)
            second=torch.load(w/"cache"/FAMILY/ds/"views.pt",mmap=True,weights_only=True)
            if len(second["regions"])!=len(qp):raise ValueError("Missing reflected views")
            prior_views=None
            if "original_regions" not in second:
                prior_views=torch.load(w/"cache/regional-evidence-interventions"/ds/"regions.pt",mmap=True,weights_only=True)
            for s,(qi,y) in qs.items():
                pi,g=gs[s]
                if (len(qi),len(pi),len(g.unique()))!=SIZES[s] or digest(UNSEEN_CLASSES[s])!=SPLIT_HASHES[s]:raise ValueError("Wrong evaluation split")
                file=root/(s+".json" if stage=="final" else stage+"-"+s+".json")
                if file.exists():
                    old=read(file)
                    if old["signature"]!=sig or (stage=="final" and old["per_query_sha256"]!=checksum(root/(s+".pt"))):raise ValueError("Bad checkpoint")
                    table[s]=old["values"];continue
                ids=stratified_sample(y,36,63241) if stage=="pilot" else torch.arange(len(qi))
                ix=qi[ids]
                if prior_views is None:
                    v=second["original_regions"][ix]
                else:
                    loc=torch.searchsorted(prior_views["ids"],ix)
                    if not torch.equal(prior_views["ids"][loc],ix):raise ValueError("Wrong region rows")
                    v=prior_views["features"][loc,:,2]
                revviews=second["regions"][ix]
                flipped=second["global"][ix]
                ident=digest({"signature":sig,"gallery":digest([str(pp[i]) for i in pi.tolist()]),"vocabulary":checksum(cfg.vocabulary)})[:16]
                heads,small=compact_index(w,s,praw[pi].cuda(),tgt,scale,ident)
                gidx=torch.load(w/"cache/pose-conditional-layout"/(s+"-gallery-uniform.pt"),mmap=True,weights_only=True).cuda()
                base=torch.load(w/"cache/pragmatic-gallery-competition"/(s+"-named-rgb16.pt"),mmap=True,weights_only=True)
                if gidx.shape!=(len(pi),len(words)) or base.shape!=(len(qi),len(pi)):raise ValueError("Wrong base or full index")
                out,err,top,check=eval_split(qraw[qi],v,revviews,flipped,src,heads,small,gidx,base,y,g,ids,cells,scale,s,
                                            Replay(kd,g,s) if stage=="final" else None,workers,key(selected) if stage=="final" else None)
                vals={k:v.mean(0).tolist() for k,v in out.items()};table[s]=vals
                rec={"signature":sig,"queries":len(ids),"gallery":len(pi),"values":vals,"max_null_identity_error":check}
                if stage=="final":
                    path=root/(s+".pt");save_tensor({"metrics":out,"query_labels":y,"top25":top},path)
                    rec.update({"protocol":{"split":s,"queries":len(qi),"gallery":len(pi),"classes":len(g.unique()),
                                   "vocabulary_sha256":checksum(cfg.vocabulary),"checkpoint":CHECKPOINT,"kd_sha256":KD_FUNCTION_SHA,"revision":rev,
                                   "gpu":torch.cuda.get_device_name(),"p_k":200 if s=="sketchy_2" else 100,"map_k":200 if s=="sketchy_2" else 0},
                                "scores":{k:dict(zip(("mAP","precision"),val)) for k,val in vals.items()},
                                "primary":key(selected),"best_contrast":key(self_),"matched_base":key(uniform),"baseline":key(("native",0.,"raw","first",0.)),
                                "vs_controls":{k:paired_delta(out[key(selected)],val,y) for k,val in out.items()},
                                "kd_max_abs_error":err,"original_kd_queries_per_condition":len(qi),"per_query_sha256":checksum(path)})
                atomic_json(file,rec)
                leader=max(vals,key=vals.get);print(stage,s,leader,vals[leader],flush=True)
                del gidx
        if stage!="final":
            means={k:[(table["sketchy_1"][k][i]+table["sketchy_2"][k][i])/2 for i in (0,1)] for k in table["sketchy_1"]}
            atomic_json(root/(stage+".json"),{"splits":table,"means":means})
            if stage=="shortlist":
                lock={"primary":select(means,{"cross","ratio"},1)[0],"self":select(means,{"self"},1)[0],
                      "uniform":select(means,{"uniform","native"},1)[0],"revision":rev,"full_sketchy_sha256":checksum(root/"shortlist.json")}
                atomic_json(root/"freeze.json",lock|{"sha256":digest(lock)})


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    a.add_argument("--phase",choices=("pilot","lock","audit","all"),default="all")
    a.add_argument("--workers",type=int,default=8)
    args=a.parse_args();run(args.workspace,args.phase,args.workers)
