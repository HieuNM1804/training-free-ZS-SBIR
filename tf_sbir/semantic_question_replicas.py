"""Evaluate minimax semantic decisions with unlabeled, cross-panel examples of each rival."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .depiction_path_diagnostic import class_interval
from .gallery_panel_resume import semantic_inputs
from .semantic_invariant_question_study import FAMILY as PILOT, accuracy, word_rows
from .semantic_question_source_transfer import match_sources
from .semantic_replica_decision import decision_rows, optimize, read, validate


FAMILY="semantic-question-replicas"
ARMS=("original","pooled","robust","exchanged")


def neighbors(gallery, features, pairs):
    order=torch.randperm(len(gallery),generator=torch.Generator().manual_seed(42))
    panel=torch.zeros(len(gallery),dtype=torch.bool)
    panel[order[::2]]=True
    indexes={k:i for i,k in enumerate(gallery.tolist())}
    gallery_features=F.normalize(features[gallery].cuda().float(),dim=-1)
    origins=sorted(set(i for pair in pairs for i in pair))
    positions=torch.tensor([indexes[i] for i in origins])
    nearest={}
    for batch in positions.split(128):
        sim=gallery_features[batch] @ gallery_features.T
        sim.masked_fill_((panel[batch,None]==panel[None]).cuda(),-torch.inf)
        top=sim.argsort(dim=-1,descending=True,stable=True)[:,:2].cpu()
        nearest.update(zip(gallery[batch].tolist(),gallery[top].tolist()))
    return {tuple(pair):tuple(next(k for k in nearest[i] if k not in pair) for i in pair) for pair in pairs}


def evaluate_row(row, replacements, original, query_words, photo_words, labels, verify):
    q,pos,neg=row["q"],row["pos"],row["neg"]
    a,b=sorted((pos,neg))
    orient=1 if pos==a else -1
    u,v=replacements[a,b]
    def get(i):
        return photo_words[i][1]
    left,right=np.stack((get(a),get(u))),np.stack((get(b),get(v)))
    info=dict(q=q,cls=row["mode"],pos=pos,neg=neg,original=original,ids=[a,b],donors=[u,v],
              collision=original in (u,v),shared=u==v,purity=[labels[a]==labels[u],labels[b]==labels[v]])
    for arm in ARMS:
        signed=decision_rows(left,right,arm)
        best=optimize(signed)
        if verify:
            native=optimize(signed,full=True)
            assert max(best.lower,native.lower)<=min(best.upper,native.upper)+2e-8
        method=dict(fit=best.lower,weights=best.weights.tolist(),bounds=[best.lower,best.upper])
        for key,reading in (("photo",photo_words[original][0]),("sketch",query_words[q][0])):
            verdict=read(signed,best,reading)
            if verify:
                native_tie=optimize(signed,reading,full=True)
                fixed=np.sign(reading@native.answer)*max(0.,(native.lower-native_tie.upper)/2)
                assert abs(fixed-verdict["certificate"])<5e-7
            method[key]={**verdict,"correct":accuracy(orient*verdict["certificate"])}
        info[arm]=method
    return info


def outcomes(rows):
    if not rows:
        return {}
    labs=torch.tensor([r["cls"] for r in rows])
    def field(arm,endpoint):
        return torch.tensor([r[arm][endpoint]["correct"] for r in rows],dtype=torch.double)
    result={}
    for arm in ARMS:
        a={}
        a["fit_one_word_accuracy"]=float(np.mean([(1+r[arm]["fit"])/2 for r in rows]))
        for endpoint in ("sketch","photo"):
            val=field(arm,endpoint)
            diff=field("robust",endpoint)-val
            a[endpoint]=dict(accuracy=float(val.mean()),ties=int((val==.5).sum()),
                             robust_minus=float(diff.mean()),interval=class_interval(diff,labs,seed=42))
        result[arm]=a
    return dict(decisions=len(rows),queries=len(set(r["q"] for r in rows)),
                fraction_donors_same_class=float(np.mean([r["purity"] for r in rows])),
                shared_donors=sum(r["shared"] for r in rows),arms=result)


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    validate()
    work=Path("/content/sbir")
    root=work/"results"/FAMILY
    root.mkdir(parents=True,exist_ok=True)
    _,paths,_,photos,galleries,qf,pf,src,tgt,scale=semantic_inputs(work,"sketchy")
    for split,(ids,labs) in galleries.items():
        previous=work/"results"/PILOT/(split+"-pairs.json")
        rows=json.loads(previous.read_bytes())["boundary"]
        pairs=sorted(set(tuple(sorted((r["pos"],r["neg"]))) for r in rows))
        replacements=neighbors(ids,pf,pairs)
        originals=match_sources(rows,paths,photos,ids,labs)
        label_lookup=dict(zip(ids.tolist(),labs.tolist()))
        candidates={i for pair in pairs for i in (*pair,*replacements[pair])}
        candidates.update(originals.values())
        qw=word_rows(qf,set(originals),[src],scale)
        pw=word_rows(pf,candidates,[src,tgt],scale)
        def assess(item):
            idx,row=item
            return evaluate_row(row,replacements,originals[row["q"]],qw,pw,label_lookup,verify=(idx==0))
        with ThreadPoolExecutor(max_workers=2) as pool:
            data=list(pool.map(assess,enumerate(rows)))
        exclude={r["q"] for r in data if r["collision"]}
        primary=[r for r in data if r["q"] not in exclude]
        assert len(primary)==2*len(set(r["q"] for r in primary))
        report=dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),excluded_queries=len(exclude),
                    primary=outcomes(primary),all=outcomes(data))
        atomic_json(root/(split+"-pairs.json"),data)
        atomic_json(root/(split+".json"),report)
        print(split,[(a, report["primary"]["arms"][a]["sketch"]["accuracy"]) for a in ARMS],flush=True)


if __name__=="__main__":
    run()
