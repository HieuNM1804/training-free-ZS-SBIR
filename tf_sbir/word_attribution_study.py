"""Measure head-attribution agreement on independent Sketchy queries and full photo galleries."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,checksum,read
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import digest,revision
from .crossdomain_inputs import CHECKPOINT
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import paired_delta
from .semantic import read_words
from .verification_inputs import get_inputs
from .visual_attribution_cache import FAMILY
from .word_attribution import ConditionalHeads
from .word_conditional_verifier import zscore

WEIGHTS=(.035,.1,.3,.8)
WIDTHS=(4,16,64)
TEMPS=(.3,1.)
MODES=("transport","aligned","raw","permuted","scalar")


def comparable_pairs(base,labels,truth):
    positive,negative,query=[],[],[]
    for i in range(len(base)):
        order=base[i].argsort(descending=True,stable=True)
        correct=truth[order]==labels[i]
        first=order[:2048][correct[:2048]]
        if not len(first):continue
        ix=first[torch.linspace(0,len(first)-1,min(32,len(first)),device=base.device).long()]
        choices=order[~correct].flip((0,))
        strength=base[i,choices]
        at=torch.searchsorted(strength,base[i,ix]).clamp(max=len(choices)-1)
        prev=(at-1).clamp(min=0)
        right=(strength[at]-base[i,ix]).abs()<(strength[prev]-base[i,ix]).abs()
        neg=choices[torch.where(right,at,prev)]
        positive.append(ix);negative.append(neg);query.append(torch.full_like(ix,i))
    if not query:
        empty=torch.empty(0,device=base.device,dtype=torch.long)
        return empty,empty,empty
    return torch.cat(query),torch.cat(positive),torch.cat(negative)


@torch.inference_mode()
def run(w):
    root=w/"results"/FAMILY
    cfg,_,queries,_,galleries,_,_=global_inputs(w,"sketchy")
    words=read_words(cfg.vocabulary)
    if len(words)!=20931:raise ValueError("Different public vocabulary")
    source,target=(v.cuda() for v in load_words(cfg,words))
    _,scale=bare_words(cfg,words)
    store=w/"cache"/FAMILY/"sketchy"
    inputs={a:torch.load(store/(a+".pt"),weights_only=True,mmap=True) for a in ("sketch","photo")}
    rev=revision()
    spec={"revision":rev,"checkpoint":CHECKPOINT,"weights":WEIGHTS,"query_words":WIDTHS,"temperatures":TEMPS,"pilot_seed":63241,
          "per_class":36,"hypothesis":"transport, 48 actual heads","index_checks":{a:read(store/(a+".json")) for a in inputs}}
    sig=digest(spec)
    if (root/"protocol.json").is_file() and read(root/"protocol.json")["sha256"]!=sig:
        raise ValueError("Conflicting study revision")
    atomic_json(root/"protocol.json",spec|{"sha256":sig})
    table={}
    for s,(data,y,g) in get_inputs(w,"sketchy").items():
        file=root/("pilot-"+s+".json")
        if file.is_file():
            old=read(file)
            if old["signature"]!=sig or old["per_query_sha256"]!=checksum(file.with_suffix(".pt")):
                raise ValueError("Stale pilot")
            table[s]=old["values"];continue
        pi=galleries[s][0]
        qi=queries[s][0]
        gallery=inputs["photo"][pi].cuda()
        labels=g.cuda()
        ids=stratified_sample(y,36,63241)
        rows=defaultdict(list);evidence=defaultdict(list);diagnostics={}
        for level in ("heads","blocks"):
            head=ConditionalHeads(gallery,source,target,level)
            diagnostics[level]={"orthogonality":head.orthogonality_error,"same_photo_prompt_correlation_median":float(head.rho.median())}
            for pos,ix in enumerate(ids.split(48)):
                base=zscore(data["regional_score"][ix].cuda())
                parts=inputs["sketch"][qi[ix]].cuda()
                truth=y[ix].cuda()
                q=data["query_native"][ix].cuda().float()
                logits,term_ids=(q@source.T*scale).topk(max(WIDTHS))
                if level=="heads":rows["regional"].append(map_values(base,truth,labels,s)[0].cpu())
                qid,good,bad=comparable_pairs(base,truth,labels)
                for n in WIDTHS:
                    terms=term_ids[:,:n]
                    measured=head.query(parts,terms)
                    for t in TEMPS:
                        prob=(logits[:,:n]*t).softmax(1)
                        if pos==0 and n==16 and t==.3:
                            err=head.explicit_error(gallery[:64],parts[:2],terms[:2],prob[:2],tuple(v[:2] for v in measured))
                            diagnostics[level]["factorized_relative_rmse"]=err
                            if err>.03:raise ValueError(f"Factorized attribution score diverged from direct calculation {err}")
                        for m in MODES:
                            if level!="heads" and m=="scalar":continue
                            aux=zscore(head.read(terms,prob,measured,m))
                            stub=f"{level}/{m}/{n}/{t:g}"
                            if len(qid):
                                diff=aux[qid,good]-aux[qid,bad]
                                evidence[stub].append(torch.stack(((diff>0).float(),diff.abs()),1).cpu())
                            for a in WEIGHTS:
                                scores=map_values(base+a*aux,truth,labels,s)[0]
                                rows[stub+f"/{a:g}"].append(scores.cpu())
            del head
            print(s,level,diagnostics[level],flush=True)
        metrics={k:torch.cat(v) for k,v in rows.items()}
        vals={k:v.mean(0).tolist() for k,v in metrics.items()}
        diagnosis={k:{"pairs":sum(len(z) for z in v),"true_above_matched_wrong":float(torch.cat(v)[:,0].mean()),
                      "mean_absolute_margin":float(torch.cat(v)[:,1].mean())} for k,v in evidence.items()}
        path=file.with_suffix(".pt")
        save_tensor({"metrics":metrics,"query_labels":y[ids],"query_ids":ids},path)
        atomic_json(file,{"signature":sig,"queries":len(ids),"gallery":len(g),"values":vals,"checks":diagnostics,
                          "matched_pairs":diagnosis,"per_query_sha256":checksum(path)})
        table[s]=vals
    avg={k:[sum(table[s][k][i] for s in table)/2 for i in (0,1)] for k in table["sketchy_1"]}
    chosen=max((k for k in avg if k.startswith("heads/transport/")),key=avg.get)
    matched=["regional",chosen.replace("/transport/","/permuted/"),chosen.replace("/transport/","/scalar/"),
             chosen.replace("heads/","blocks/"),chosen.replace("/transport/","/raw/"),chosen.replace("/transport/","/aligned/")]
    paired={}
    for s in table:
        data=torch.load(root/("pilot-"+s+".pt"),weights_only=True)
        paired[s]={k:paired_delta(data["metrics"][chosen],data["metrics"][k],data["query_labels"]) for k in matched}
    atomic_json(root/"pilot.json",{"splits":table,"means":avg,"nominee":chosen,"matched":matched,"paired":paired})
    print("selected",chosen,avg[chosen],flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(a.parse_args().workspace)
