"""Does the dependence of regional words across scales transfer between sketch and photo?"""

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
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .nested_scale_cache import FAMILY
from .nested_scale_words import PairIndex,word_bank,correction
from .quickdraw_locked import KD_FUNCTION_SHA
from .regional_intervention_study import mixscore,index as compact_index
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import load_kd,paired_delta
from .semantic import read_words


def key(c):return json.dumps(c,separators=(",",":"))


def grid():
    return [(n,t,h,l,m,v) for n in (40,80) for t in (.45,1.) for h in (0.,.65)
            for l in (.15,.55) for m in ("cross","null","direct")
            for v in ((1.,4.,12.) if m=="cross" else ((1.,) if m=="direct" else (0.,)))]


def best(means,kind,count):
    return [tuple(json.loads(k)) for k in sorted((k for k in means if json.loads(k)[4]==kind),key=means.get,reverse=True)[:count]]


def measure(w,s,photos,target,source,scale,compact,compact_src,crops,qv,historic,y,g,ids,variants,ref,workers,primary):
    pp={}
    rows={key(c):[] for c in variants}
    errs={}
    tests={}
    tops=[]
    pool=ThreadPoolExecutor(max_workers=workers) if ref else nullcontext()
    with pool as ex:
        for n,t,h in sorted(set(c[:3] for c in variants)):
            if n not in pp:pp[n]=word_bank(photos,crops,target,scale,n)
            these=[c for c in variants if c[:3]==(n,t,h)]
            wrong=any(c[4].startswith("wrong") for c in these)
            bank=PairIndex(photos,crops,target,scale,*pp[n],t,h,with_wrong=wrong)
            tests[key((n,t,h))]={"max_word_marginal_error":bank.marginal_error,
                                "outer_word_ids":pp[n][0].tolist(),"inner_word_ids":pp[n][1].tolist()}
            for offset in range(0,len(ids),72):
                part=ids[offset:offset+72]
                feats=qv[offset:offset+len(part)].cuda().float()
                labels=y[part]
                historical=historic[part].cuda().clamp_min(1e-30).log()
                # Exactly the two-scale independent-zoom query baseline, with equal mass per scale.
                z=(feats@compact_src.T)*scale
                zoom=torch.logaddexp(z[:,:9].logsumexp(1)-z.new_tensor(9.).log(),
                                     z[:,9:].logsumexp(1)-z.new_tensor(4.).log())-z.new_tensor(2.).log()
                baseline=mixscore(zoom,compact,historical,.4)
                terms=bank.overlaps(feats,source,wrong)
                for c in these:
                    if c[3]==0. and c[5]==0.:scores=baseline
                    else:scores=(1-c[3])*baseline+c[3]*terms["00"]+c[5]*correction(terms,c[4])
                    ap,order,_,_=map_values(scores,labels.cuda(),g.cuda(),s)
                    nm=key(c)
                    if ref:
                        official=ref.score(order,labels,ex)
                        err=float((official-ap.cpu()).abs().max())
                        if err>2e-6:raise ValueError("Official KD metric replay differs")
                        errs[nm]=max(errs.get(nm,0.),err)
                        if nm==primary:tops.append(order[:,:25].cpu().int())
                    else:official=ap.cpu()
                    rows[nm].append(official)
            print(s,"indexed",n,t,h,flush=True)
            del bank
    return {k:torch.cat(v) for k,v in rows.items()},errs,tests,torch.cat(tops) if tops else None


@torch.inference_mode()
def run(w,phase,workers):
    root=w/"results"/FAMILY
    src_rev=revision()
    spec={"revision":src_rev,"grid":grid(),"primary":"cross","seed":63241,"pilot_each_class":36,
          "selection":"highest equal mean mAP over full Sketchy splits, then mean precision"}
    sig=digest(spec)
    if (root/"protocol.json").is_file():
        if read(root/"protocol.json")["sha256"]!=sig:raise ValueError("Incompatible previous output")
    else:atomic_json(root/"protocol.json",spec|{"sha256":sig})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Changed KD function")
    stages={"pilot":("pilot",),"lock":("shortlist",),"audit":("final",),"all":("pilot","shortlist","final")}[phase]
    for stage in stages:
        if stage=="pilot":cells=grid()
        elif stage=="shortlist":
            means=read(root/"pilot.json")["means"]
            cells=best(means,"cross",3)+best(means,"null",2)+best(means,"direct",2)
        else:
            lock=read(root/"freeze.json")
            if lock["full_sketchy_sha256"]!=checksum(root/"shortlist.json") or lock["revision"]!=src_rev:raise ValueError("Broken lock")
            chosen,unpaired,direct=(tuple(lock[k]) for k in ("primary","unpaired","direct"))
            n,t,h,blend,_,strength=chosen
            cells=sorted({chosen,unpaired,direct,(n,t,h,blend,"null",0.),(n,t,h,0.,"null",0.),
                          *[(n,t,h,blend,m,strength) for m in ("query","gallery","wrong_query","wrong_both","reverse","direct")]})
        table={}
        for ds in (("sketchy","tuberlin") if stage=="final" else ("sketchy",)):
            cfg,_,qs,_,gs,qall,pall=global_inputs(w,ds)
            words=read_words(cfg.vocabulary)
            if len(words)!=20931:raise ValueError("Changed vocabulary")
            _,scale=bare_words(cfg,words);source,target=(v.cuda() for v in load_words(cfg,words))
            pviews=torch.load(w/"cache"/FAMILY/ds/"photo.pt",mmap=True,weights_only=True)
            qbank=torch.load(w/"cache/regional-evidence-interventions"/ds/"regions.pt",mmap=True,weights_only=True)
            for s,(qi,y) in qs.items():
                pi,g=gs[s]
                if (len(qi),len(pi),len(g.unique()))!=SIZES[s] or digest(UNSEEN_CLASSES[s])!=SPLIT_HASHES[s]:
                    raise ValueError("Changed split")
                dest=root/(s+".json" if stage=="final" else stage+"-"+s+".json")
                if dest.exists():
                    record=read(dest)
                    if record["signature"]!=sig or (stage=="final" and record["per_query_sha256"]!=checksum(root/(s+".pt"))):
                        raise ValueError("Previous result differs")
                    table[s]=record["values"];continue
                ids=stratified_sample(y,36,63241) if stage=="pilot" else torch.arange(len(y))
                poss=torch.searchsorted(qbank["ids"],qi[ids])
                if not torch.equal(qbank["ids"][poss],qi[ids]):raise ValueError("Missing source query features")
                qv=qbank["features"][poss,:,2]
                crops=pviews[pi]
                photos=pall[pi].cuda()
                heads,compact=compact_index(w,s,photos,target,scale,sig[:16])
                base=torch.load(w/"cache/pragmatic-gallery-competition"/(s+"-named-rgb16.pt"),mmap=True,weights_only=True)
                if base.shape!=(len(qi),len(pi)):raise ValueError("Wrong native baseline")
                rows,err,checks,tops=measure(w,s,photos,target,source,scale,compact,source[heads],crops,qv,base,y,g,ids,cells,
                                             Replay(kd,g,s) if stage=="final" else None,workers,key(chosen) if stage=="final" else None)
                vals={k:v.mean(0).tolist() for k,v in rows.items()}
                table[s]=vals
                record={"values":vals,"signature":sig,"native_queries":len(qi),"queries":len(ids),"gallery":len(pi),"checks":checks}
                if stage=="final":
                    path=root/(s+".pt");save_tensor({"metrics":rows,"query_labels":y,"top25":tops},path)
                    record.update({"protocol":{"split":s,"queries":len(qi),"gallery":len(pi),"classes":len(g.unique()),
                                               "checkpoint":CHECKPOINT,"vocabulary_sha256":checksum(cfg.vocabulary),"kd_sha256":KD_FUNCTION_SHA,
                                               "gpu":torch.cuda.get_device_name(),"revision":src_rev,"map_k":200 if s=="sketchy_2" else 0,
                                               "p_k":200 if s=="sketchy_2" else 100},
                                   "primary":key(chosen),"best_contrast":key(direct),"matched_base":key(unpaired),
                                   "baseline":key((chosen[0],chosen[1],chosen[2],0.,"null",0.)),
                                   "scores":{k:dict(zip(("mAP","precision"),v)) for k,v in vals.items()},
                                   "vs_controls":{k:paired_delta(rows[key(chosen)],v,y) for k,v in rows.items()},
                                   "kd_max_abs_error":err,"original_kd_queries_per_condition":len(qi),"per_query_sha256":checksum(path)})
                atomic_json(dest,record)
                leader=max(vals,key=vals.get);print(stage,s,leader,vals[leader],flush=True)
        if stage!="final":
            means={k:[(table["sketchy_1"][k][i]+table["sketchy_2"][k][i])/2 for i in (0,1)] for k in table["sketchy_1"]}
            atomic_json(root/(stage+".json"),{"means":means,"splits":table})
            if stage=="shortlist":
                lock={"primary":best(means,"cross",1)[0],"unpaired":best(means,"null",1)[0],"direct":best(means,"direct",1)[0],
                      "revision":src_rev,"full_sketchy_sha256":checksum(root/"shortlist.json")}
                atomic_json(root/"freeze.json",lock|{"sha256":digest(lock)})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",default=Path("/content/sbir"),type=Path)
    parser.add_argument("--phase",choices=("pilot","lock","audit","all"),default="all")
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args();run(args.workspace,args.phase,args.workers)
