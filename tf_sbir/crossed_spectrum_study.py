"""Exploratory selection on Sketchy and transfer of frozen spectral evidence to TU."""

from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import os
import time

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES,bare_words
from .crossdomain_benchmark import revision,digest,SPLIT_HASHES
from .crossed_spectrum import CrossedSpectrum,read_basis,verify_crossed_formula
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import load_kd,paired_delta
from .semantic import read_words
from .shared_concept_likelihood import emission


FAMILY="crossed-spectral-evidence"
RANK=128


def name(c):
    return json.dumps(list(c),separators=(",",":"))


def data(w,dataset):
    cfg,_,qs,_,gs,q,p=global_inputs(w,dataset)
    terms=read_words(cfg.vocabulary)
    _,scale=bare_words(cfg,terms)
    banks=tuple(x.cuda() for x in load_words(cfg,terms))
    if len(terms)!=20931:
        raise ValueError("Unexpected public dictionary")
    for split,(qi,ql) in qs.items():
        pi,pl=gs[split]
        if (len(qi),len(pi),len(pl.unique()))!=SIZES[split] or digest(UNSEEN_CLASSES[split])!=SPLIT_HASHES[split]:
            raise ValueError("Unexpected split")
        photos=p[pi].cuda()
        probs=((photos@banks[1].T)*scale).softmax(1)
        yield split,q[qi].cuda(),ql,photos,pl,probs,banks,scale,checksum(cfg.vocabulary)


def evaluate(model,query,labels,glabels,source,scale,split,configs,*,replay=None,workers=8):
    collected={c:[] for c in configs}
    qtops={c:[] for c in configs} if replay else None
    errors={c:0. for c in configs}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(0,len(query),192):
            l=emission(query[i:i+192],source,scale)
            stats=model.estimates(l)
            y=labels[i:i+192]
            for c in configs:
                s=model.scores(stats,c[2:]) if c[4]!="direct" else stats[0]
                if not torch.isfinite(s).all():
                    raise ValueError("Nonfinite scores")
                fast,order,_,_=map_values(s,y.cuda(),glabels.cuda(),split)
                if replay:
                    got=replay.score(order,y,pool)
                    errors[c]=max(errors[c],float((fast.cpu()-got).abs().max()))
                    if errors[c]>2e-6:
                        raise ValueError("Evaluator disagreement")
                    qtops[c].append(order[:,:100].int().cpu())
                else:
                    got=fast.cpu()
                collected[c].append(got)
    return ({name(c):torch.cat(v) for c,v in collected.items()},
            None if not replay else {name(c):torch.cat(v) for c,v in qtops.items()},
            {name(c):v for c,v in errors.items()})


def candidates():
    return [(v,k,r,8,m,p,y) for v,k,r,m,p,y in
            product(("rgb","joint"),(24,96),(24,64,128),("fixed","cross"),(0,4),(0.,.16))]


def entire(root,all_trials):
    means={c:[sum(all_trials[s][c][i] for s in ("sketchy_1","sketchy_2"))/2 for i in range(2)]
           for c in all_trials["sketchy_1"]}
    return means


def select_k(means,mode,count):
    ids=[k for k in means if json.loads(k)[4]==mode]
    return [tuple(json.loads(k)) for k in sorted(ids,key=means.get,reverse=True)[:count]]


def groups(configs):
    return sorted(set((c[0],c[1]) for c in configs))


def witness_diagnosis(profiles,l,glabels,y,top,graph_ids):
    # Labels enter only after rankings and the complete gallery index are fixed.
    classmass=profiles.new_zeros((int(glabels.max())+1,profiles.size(1)))
    classmass.index_add_(0,glabels,profiles)
    wtotal=classmass.sum(0)
    first=top[:,0].long().cuda()
    own=profiles[first]
    responsibility=own*l
    responsibility=responsibility/responsibility.sum(1,keepdim=True).clamp_min(1e-30)
    remain=(wtotal-own).clamp_min(1e-30)
    same=(classmass[glabels[first]]-own).clamp_min(0)/remain
    wpurity=(responsibility*same).sum(1)
    npurity=(glabels[graph_ids[first]]==glabels[first,None]).float().mean(1)
    return {"queries":len(y),"mean_off_class_witness_mass":float(1-wpurity.mean()),
            "mean_same_class_nearest_gallery":float(npurity.mean()),"correct_top1":int((glabels[first]==y).sum()),
            "top1_wrong_mean_off_class_witness_mass":float(1-wpurity[glabels[first]!=y].mean())}


def view_model(folder,split,p,probs,target,view,k):
    basis=read_basis(folder/"index",split,p,probs,target,view,k,RANK)
    return CrossedSpectrum(basis,probs),basis


@torch.inference_mode()
def run(w,workers):
    folder=w/"results"/FAMILY
    protocol={"revision":revision(),"primary":"cross","candidates":candidates(),
              "selection":"balanced pilot then full Sketchy mean mAP then precision, separately for fixed and crossed",
              "sketchy_pilot_per_class":24,"pilot_seed":74217,"cross_axes":["photo","word"]}
    atomic_json(folder/"protocol.json",protocol|{"sha256":digest(protocol)})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Wrong KD evaluator")
    for stage in ("pilot","shortlist"):
        trial={}
        if stage=="pilot":
            specs=candidates()
        else:
            table=read(folder/"pilot.json")["means"]
            specs=select_k(table,"fixed",3)+select_k(table,"cross",3)
        for split,q,y,p,g,probs,banks,scale,vhash in data(w,"sketchy"):
            idx=stratified_sample(y,24,74217) if stage=="pilot" else torch.arange(len(y))
            trial[split]={}
            for view,k in groups(specs):
                model,basis=view_model(folder,split,p,probs,banks[1],view,k)
                configs=[c for c in specs if c[:2]==(view,k)]
                val,_,_=evaluate(model,q[idx],y[idx],g,banks[0],scale,split,configs)
                trial[split].update({c:v.mean(0).tolist() for c,v in val.items()})
                best=max(val,key=lambda c:val[c][:,0].mean())
                print(stage,split,view,k,"best",best,trial[split][best],flush=True)
                del model,basis
        means=entire(folder,trial)
        atomic_json(folder/(stage+".json"),{"scores":trial,"means":means,"protocol_sha256":checksum(folder/"protocol.json")})
    best=select_k(means,"cross",1)[0]
    fixed=select_k(means,"fixed",1)[0]
    decision={"primary":best,"best_fixed":fixed,"revision":revision(),"full_sketchy_sha256":checksum(folder/"shortlist.json")}
    atomic_json(folder/"freeze.json",decision|{"sha256":digest(decision)})
    finals={best,fixed}
    for mode in ("direct","fixed","photos","words"):
        finals.add((*best[:4],mode,*best[5:]))
    finals=sorted(finals)
    for dataset in ("sketchy","tuberlin"):
        for split,q,y,p,g,probs,banks,scale,vhash in data(w,dataset):
            start=time.perf_counter()
            per,tops,errors={},{},{}
            diag={}
            for view,k in groups(finals):
                model,basis=view_model(folder,split,p,probs,banks[1],view,k)
                configs=[c for c in finals if c[:2]==(view,k)]
                v,t,err=evaluate(model,q,y,g,banks[0],scale,split,configs,replay=Replay(kd,g,split),workers=workers)
                per.update(v);tops.update(t);errors.update(err)
                if best[:2]==(view,k):
                    ids=stratified_sample(y,24,74217)
                    ll=emission(q[ids],banks[0],scale)
                    diag={"cross_identity_error":verify_crossed_formula(model,ll),"basis_residual":basis["max_residual"],
                          "eigenvalues":basis["eigenvalues"].tolist(),
                          "witness":witness_diagnosis(probs,ll,g.cuda(),y[ids].cuda(),tops[name(best)][ids],basis["neighbors"].cuda())}
                del model,basis
            details=folder/(split+".pt")
            save_tensor({"metrics":per,"query_labels":y,"top100":{k:v for k,v in tops.items() if k==name(best)}},details)
            base=torch.load(w/"results/gallery-lexical-feedback"/(split+".pt"),map_location="cpu",weights_only=True)
            if not torch.equal(base["query_labels"],y): raise ValueError("Historical baseline query alignment")
            historical=base["metrics"]
            meta={"split":split,"revision":revision(),"gpu":torch.cuda.get_device_name(),"queries":len(y),
                  "gallery":len(p),"classes":len(g.unique()),"checkpoint":CHECKPOINT,"vocabulary_sha256":vhash,
                  "kd_sha256":KD_FUNCTION_SHA,"map_k":200 if split=="sketchy_2" else 0,"p_k":200 if split=="sketchy_2" else 100}
            obj={"protocol":meta,"primary":name(best),"fixed":name(fixed),"diagnosis":diag,
                 "scores":{k:dict(zip(("mAP","precision"),v.mean(0).tolist())) for k,v in per.items()},
                 "vs_internal":{k:paired_delta(per[name(best)],v,y) for k,v in per.items()},
                 "vs_historical":{k:paired_delta(per[name(best)],historical[k],y) for k in ("diffusion","distinct_witness")},
                 "kd_max_abs_error":errors,"per_query_sha256":checksum(details),"elapsed_seconds":time.perf_counter()-start}
            atomic_json(folder/(split+".json"),obj)
            print("official",split,obj["scores"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--workers",type=int,default=min(10,len(os.sched_getaffinity(0))))
    args=parser.parse_args()
    run(args.workspace,args.workers)
