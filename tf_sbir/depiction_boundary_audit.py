"""Distinguish repairing an adjacent ranking error from reinforcing an already-correct class transition."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .crossdomain_benchmark import digest, revision
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .predictive_depiction_distance import FAMILY, fixed_distance, predict, similarity
from .predictive_depiction_study import baseline, candidate_axes, parse


def outcomes(logits, score, y, gl, strength):
    val,ix=logits.sort(dim=1,descending=True,stable=True)
    val=val[:,:200]
    ix=ix[:,:200]
    rel=gl[ix]==y[:,None]
    e=score.gather(1,ix)
    sign=2*rel[:,:-1].float()-1
    unit=sign*(e[:,:-1]-e[:,1:])
    gap=val[:,:-1]-val[:,1:]
    before=rel[:,:-1] & ~rel[:,1:]
    mistaken=(~rel[:,:-1]) & rel[:,1:]
    changed=(strength*(e[:,1:]-e[:,:-1]))>gap
    med=gap.median(dim=1,keepdim=True).values
    pools={"all":torch.ones_like(before),"small_margin":gap<=med,
           "first50":torch.arange(199,device=ix.device)[None]<49}
    result={}
    for pool,mask in pools.items():
        for label,m in (("right_above",before),("right_below",mistaken)):
            sel=mask&m
            result[pool+"/"+label]=torch.stack((sel.sum(1),(sel&(unit>0)).sum(1),(sel&(unit==0)).sum(1),
                 (sel&changed).sum(1),(gap*sel).sum(1),(unit*sel).sum(1)),1).cpu()
    return result


def describe(rows):
    z=rows.double()
    use=z[:,0]>0
    if not use.any():
        return {"queries":0,"pairs":0}
    val=z[use]
    a=val[:,0]
    return {"queries":int(use.sum()),"pairs":int(a.sum()),"strictly_reordered_pairs":int(val[:,3].sum()),
            "correct_evidence_per_query":float(((val[:,1]+.5*val[:,2])/a).mean()),
            "correct_evidence_per_pair":float((val[:,1]+.5*val[:,2]).sum()/a.sum()),
            "fraction_crossing_original_margin_per_query":float((val[:,3]/a).mean()),
            "mean_incumbent_margin_per_query":float((val[:,4]/a).mean()),
            "mean_relevant_minus_wrong_per_query":float((val[:,5]/a).mean())}


@torch.inference_mode()
def run(w):
    folder=w/"results"/FAMILY
    chosen=read(folder/"pilot.json")
    n,mode,ridge,gain=parse(chosen["nominee"])
    inputs=semantic_inputs(w,"sketchy")
    _,_,qs,_,gs,raw,pr,src,tgt,scale=inputs
    data,summary={},{}
    for s,(qi,y) in qs.items():
        archived=torch.load(folder/("pilot-"+s+".pt"),mmap=True,weights_only=True)
        ids=archived["query_ids"]
        own=w/"cache"/FAMILY/(s+"-regional-"+digest(ids.tolist())[:16]+".pt")
        base=(torch.load(own,weights_only=True,mmap=True) if own.is_file() else baseline(w,s,inputs,ids))
        pi,g=gs[s]
        truth=g.cuda()
        photo=pr[pi].cuda()
        capture=defaultdict(list)
        for off in range(0,len(ids),12):
            sl=slice(off,off+12)
            q=raw[qi[ids[sl]]].cuda()
            x,yphot,u,v=[t[...,:n] for t in candidate_axes(q,photo,src,tgt,scale)]
            r=x-yphot
            ordinary=fixed_distance(r)
            val,inside,_=predict(r,u,v,ridge)
            perm=torch.randperm(n,generator=torch.Generator().manual_seed(92741)).to(v.device)
            skew=similarity(predict(r,u,v.index_select(-1,perm),ridge)[0])
            own_u=similarity(predict(r,u,torch.zeros_like(v),ridge)[0])
            own_v=similarity(predict(r,torch.zeros_like(u),v,ridge)[0])
            effects={"native":ordinary,"predict":similarity(val),"gain":similarity(val)-ordinary,
                     "inside_gain":similarity(inside)-ordinary,"source_gain":own_u-ordinary,
                     "target_gain":own_v-ordinary,"permuted_gain":skew-ordinary,
                     "predict_specificity":similarity(val)-skew}
            z=base[sl].cuda()
            for key,eff in effects.items():
                for field,tensor in outcomes(z,eff,y[ids[sl]].cuda(),truth,gain).items():
                    capture[key+"/"+field].append(tensor)
        packed={k:torch.cat(v) for k,v in capture.items()}
        data[s]={"query_ids":ids,"query_labels":y[ids],"fields":packed}
        summary[s]={k:describe(v) for k,v in packed.items()}
        print(s,"predictive gain on correct and inverted boundaries",
              summary[s]["gain/all/right_above"],summary[s]["gain/all/right_below"],flush=True)
    path=folder/"hard-boundary-audit.pt"
    save_tensor(data,path)
    atomic_json(path.with_suffix(".json"),{"revision":revision(),"nominee":chosen["nominee"],"public_shape":n,
                "original_pilot":checksum(folder/"pilot.json"),"summary":summary,"per_query_sha256":checksum(path),
                "row_fields":["pairs","favorable","ties","strictly reordered","original gap sum","signed evidence sum"]})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
