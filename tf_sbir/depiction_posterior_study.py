"""Test a gallery-derived noise model for how individual sketches express visual words."""

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
from .crossdomain_benchmark import digest,revision,SPLIT_HASHES
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .depiction_posterior import DepictionModel
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .regional_intervention_study import index as compact_index,mixscore
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import load_kd,paired_delta
from .semantic import read_words
from .shared_concept_likelihood import fit_prior

FAMILY="depiction-noise-posterior"
SPLITS=("sketchy_1","sketchy_2")


def key(c):return json.dumps(c,separators=(",",":"))


def grid():
    core=[(k,view,t,e,kind,"gallery",mix) for k in (128,256) for view in ("native","zoom")
          for t in (.12,.5,2.) for e in (0.,.5,1.) for kind in ("mean","predictive","ratio") for mix in (.4,.8)]
    return core+[(128,view,0.,0.,"native","gallery",mix) for view in ("native","zoom") for mix in (.4,.8)]


def subset(means,which,n):
    def test(c):
        if which=="native":return c[4]=="native"
        return c[4]!="native" and ((c[3]>0) if which=="individual" else c[3]==0)
    return [tuple(json.loads(k)) for k in sorted((k for k in means if test(json.loads(k))),key=means.get,reverse=True)[:n]]


def fitted(w,s,k,gv,text,scale,ident):
    file=w/"cache"/FAMILY/(s+"-"+str(k)+"-"+ident+".pt")
    if file.is_file():
        state=torch.load(file,weights_only=True,map_location="cpu",mmap=True)
        obj=object.__new__(DepictionModel)
        for a,v in state.items():setattr(obj,a,v.cuda() if torch.is_tensor(v) else v)
        return obj
    obj=DepictionModel(gv,text,scale)
    save_tensor({a:(v.cpu() if torch.is_tensor(v) else v) for a,v in vars(obj).items()},file)
    return obj


def score(w,s,q,views,gviews,src,tgt,heads,compact,terms,scale,historical,y,g,ids,cells,ref,workers,primary,ident):
    rows={key(c):[] for c in cells};errors={};checks={};tops=[]
    ctx=ThreadPoolExecutor(max_workers=workers) if ref else nullcontext()
    with ctx as pool:
        for k in sorted(set(c[0] for c in cells)):
            word=terms[:k]
            # Use the exact indices selected by the historical compact gallery rather than assuming a tie order.
            positions=(heads[:,None]==word).nonzero()
            if positions.size(0)!=len(heads) or not torch.equal(positions[:,0],torch.arange(len(heads),device=heads.device)):
                raise ValueError("Compact heads absent from the observation vocabulary")
            slots=positions[:,1]
            model=fitted(w,s,k,gviews,tgt[word],scale,ident)
            checks[str(k)]=model.diagnostic
            relevant=[c for c in cells if c[0]==k]
            for off in range(0,len(ids),96):
                idx=ids[off:off+96]
                hist=historical[idx].cuda().clamp_min(1e-30).log()
                labels=y[idx]
                x=model.observe(q[idx].cuda(),views[off:off+len(idx)].cuda(),src[word],scale)
                for config in sorted(set(c[1:6] for c in relevant)):
                    view,tau,eta,kind,geometry=config
                    pred=model.likelihood(x,view,tau,eta,kind,geometry)[:,slots]
                    for c in (c for c in relevant if c[1:6]==config):
                        row,order,_,_=map_values(mixscore(pred,compact,hist,c[-1]),labels.cuda(),g.cuda(),s)
                        nm=key(c)
                        if ref:
                            actual=ref.score(order,labels,pool)
                            err=float((actual-row.cpu()).abs().max())
                            if err>2e-6:raise ValueError("Mismatch with original KD evaluator")
                            errors[nm]=max(errors.get(nm,0.),err)
                            if nm==primary:tops.append(order[:,:25].int().cpu())
                        else:actual=row.cpu()
                        rows[nm].append(actual)
    return {k:torch.cat(v) for k,v in rows.items()},errors,checks,torch.cat(tops) if tops else None


@torch.inference_mode()
def run(w,phase,workers):
    root=w/"results"/FAMILY
    rev=revision()
    protocol={"revision":rev,"grid":grid(),"primary":"individual observation posterior","seed":63241,"pilot_each_class":36,
              "selection":"equal full Sketchy mean mAP then precision","target_views":"unlabelled gallery only"}
    sig=digest(protocol)
    if (root/"protocol.json").exists():
        if read(root/"protocol.json")["sha256"]!=sig:raise ValueError("Changed protocol")
    else:atomic_json(root/"protocol.json",protocol|{"sha256":sig})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Changed KD source")
    stages={"pilot":("pilot",),"lock":("shortlist",),"audit":("final",),"all":("pilot","shortlist","final")}[phase]
    for stage in stages:
        if stage=="pilot":cells=grid()
        elif stage=="shortlist":
            m=read(root/"pilot.json")["means"]
            cells=subset(m,"individual",4)+subset(m,"global",3)+subset(m,"native",2)
        else:
            lock=read(root/"freeze.json")
            if lock["full_sketchy_sha256"]!=checksum(root/"shortlist.json") or lock["revision"]!=rev:raise ValueError("Wrong freeze")
            best,global_,native=(tuple(lock[k]) for k in ("primary","global","native"))
            k,view,t,e,kind,geometry,mix=best
            cells=sorted({best,global_,native,(k,view,t,0.,kind,geometry,mix),(k,view,0.,0.,"native","gallery",mix),
                          (k,view,0.,0.,"native","gallery",0.),*[(k,view,t,e,kind,h,mix) for h in ("word","scalar_query","spherical","shuffled_query","reversed")],
                          *[(k,view,t,e,h,geometry,mix) for h in ("mean","ratio","predictive")]})
        tables={}
        for ds in (("sketchy","tuberlin") if stage=="final" else ("sketchy",)):
            cfg,_,qs,pp,gs,qraw,praw=global_inputs(w,ds)
            ws=read_words(cfg.vocabulary)
            if len(ws)!=20931:raise ValueError("Wrong public vocabulary")
            src,tgt=(v.cuda() for v in load_words(cfg,ws));_,scale=bare_words(cfg,ws)
            qbank=torch.load(w/"cache/regional-evidence-interventions"/ds/"regions.pt",weights_only=True,mmap=True)
            pbank=torch.load(w/"cache/nested-scale-word-evidence"/ds/"photo.pt",weights_only=True,mmap=True)
            for s,(qi,y) in qs.items():
                pi,g=gs[s]
                if (len(qi),len(pi),len(g.unique()))!=SIZES[s] or digest(UNSEEN_CLASSES[s])!=SPLIT_HASHES[s]:raise ValueError("Split mismatch")
                dest=root/(s+".json" if stage=="final" else stage+"-"+s+".json")
                if dest.exists():
                    j=read(dest)
                    if j["signature"]!=sig or (stage=="final" and j["per_query_sha256"]!=checksum(root/(s+".pt"))):raise ValueError("Invalid checkpoint")
                    tables[s]=j["values"];continue
                idx=stratified_sample(y,36,63241) if stage=="pilot" else torch.arange(len(qi))
                loc=torch.searchsorted(qbank["ids"],qi[idx])
                if not torch.equal(qbank["ids"][loc],qi[idx]):raise ValueError("Missing current query windows")
                qv=qbank["features"][loc,:,2];pv=pbank[pi]
                p=praw[pi].cuda()
                ident=digest({"signature":sig,"gallery":digest([str(pp[i]) for i in pi.tolist()]),"vocabulary":checksum(cfg.vocabulary)})[:16]
                heads,compact=compact_index(w,s,p,tgt,scale,ident)
                em=p.new_empty((len(p),len(ws)))
                for i in range(0,len(p),256):em[i:i+256]=((p[i:i+256]@tgt.T)*scale).softmax(1)
                words=fit_prior(em)[0]["mle"].topk(256).indices
                del em
                historical=torch.load(w/"cache/pragmatic-gallery-competition"/(s+"-named-rgb16.pt"),weights_only=True,mmap=True)
                if historical.shape!=(len(qi),len(pi)):raise ValueError("Wrong native baseline")
                rows,err,checks,tops=score(w,s,qraw[qi],qv,pv,src,tgt,heads,compact,words,scale,historical,y,g,idx,cells,
                                          Replay(kd,g,s) if stage=="final" else None,workers,key(best) if stage=="final" else None,ident)
                vals={k:v.mean(0).tolist() for k,v in rows.items()};tables[s]=vals
                rec={"signature":sig,"queries":len(idx),"gallery":len(pi),"values":vals,"checks":checks}
                if stage=="final":
                    path=root/(s+".pt");save_tensor({"metrics":rows,"query_labels":y,"top25":tops},path)
                    rec.update({"protocol":{"split":s,"queries":len(qi),"gallery":len(pi),"classes":len(g.unique()),"checkpoint":CHECKPOINT,
                                 "vocabulary_sha256":checksum(cfg.vocabulary),"kd_sha256":KD_FUNCTION_SHA,"revision":rev,"gpu":torch.cuda.get_device_name(),
                                 "map_k":200 if s=="sketchy_2" else 0,"p_k":200 if s=="sketchy_2" else 100},
                                "primary":key(best),"fixed_global":key(global_),"matched_base":key(native),
                                "baseline":key((best[0],best[1],0.,0.,"native","gallery",0.)),
                                "scores":{k:dict(zip(("mAP","precision"),v)) for k,v in vals.items()},
                                "vs_controls":{k:paired_delta(rows[key(best)],v,y) for k,v in rows.items()},
                                "original_kd_queries_per_condition":len(qi),"kd_max_abs_error":err,"per_query_sha256":checksum(path)})
                atomic_json(dest,rec)
                win=max(vals,key=vals.get);print(stage,s,win,vals[win],flush=True)
        if stage!="final":
            m={k:[sum(tables[s][k][i] for s in SPLITS)/2 for i in (0,1)] for k in tables[SPLITS[0]]}
            atomic_json(root/(stage+".json"),{"means":m,"splits":tables})
            if stage=="shortlist":
                lock={"primary":subset(m,"individual",1)[0],"global":subset(m,"global",1)[0],"native":subset(m,"native",1)[0],
                      "revision":rev,"full_sketchy_sha256":checksum(root/"shortlist.json")}
                atomic_json(root/"freeze.json",lock|{"sha256":digest(lock)})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--phase",choices=("pilot","lock","audit","all"),default="all")
    p.add_argument("--workers",type=int,default=8)
    a=p.parse_args();run(a.workspace,a.phase,a.workers)
