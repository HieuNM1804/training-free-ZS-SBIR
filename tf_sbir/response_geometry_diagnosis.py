"""Measure shared concept relationships on fixed true and false candidates of the existing retriever."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .concept_response_geometry import ARMS, SIZES, checks, descriptors, match
from .crossdomain_benchmark import revision
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .response_geometry_views import FAMILY, PAIR_NAMES, inputs_plan, query_pack, read_photos, study_base


def isotropic_check(device):
    gen=torch.Generator(device=device).manual_seed(7229)
    cov=torch.tensor([-.8,-.3,0.,.3,.8],device=device)
    normals=torch.randn((180000,2),device=device,generator=gen)
    x=normals[:,0,None]
    y=x*cov+normals[:,1,None]*(1-cov.square()).sqrt()
    measured=(x.sign()*y.sign()).mean(0)
    error=float((measured-2/torch.pi*cov.asin()).abs().max())
    if error>.015:raise ValueError("Rotational reference does not reproduce empirical sign concordance")
    return error


def stats(delta,valid,labels):
    if not bool(valid.any()):
        return {"queries":0}
    d=delta[valid]
    y=labels[valid]
    accuracy=(d>0).double()+.5*(d==0).double()
    n=labels.unique().numel()
    cnt=y.bincount(minlength=n).double()
    wins=torch.zeros(n,dtype=torch.float64).index_add_(0,y,accuracy)
    sample=torch.randint(n,(5000,n),generator=torch.Generator().manual_seed(9657))
    denom=cnt[sample].sum(1)
    rep=wins[sample].sum(1)[denom>0]/denom[denom>0]
    lo,hi=rep.quantile(torch.tensor([.025,.975],dtype=torch.float64))
    return {"queries":len(d),"select_relevant":float(accuracy.mean()),"ci95_class":[float(lo),float(hi)],
            "mean_relevant_minus_wrong":float(d.mean()),"ties":int((d==0).sum()),"present_classes":int((cnt>0).sum())}


@torch.inference_mode()
def run(w):
    inputs=semantic_inputs(w,"sketchy")
    _,_,qs,_,gs,q,p,src,tgt,scale=inputs
    folder,plan=inputs_plan(w,"sketchy",inputs)
    root=w/"results"/FAMILY
    invariants=checks(src.device)
    invariants["isotropic_sign_MC"]=isotropic_check(src.device)
    out,summary={},{}
    for s,setting in plan["splits"].items():
        ids,pair=setting["ids"],setting["pairs"]
        qi,y=qs[s]
        pi,g=gs[s]
        unique=pair[pair>=0].unique().sort().values
        known=read_photos(folder,plan,unique)
        source=query_pack(w,s,inputs,ids)
        local=torch.searchsorted(unique,pair.clamp_min(int(unique[0]))).clamp_max(len(unique)-1)
        if not torch.equal(unique[local][pair>=0],pair[pair>=0]):raise ValueError("Matched gallery image changed")
        base=study_base(w,s,inputs,ids)
        correct_top=g[base.argmax(1)]==y[ids]
        diff=defaultdict(list)
        for off in range(0,len(ids),24):
            sl=slice(off,off+24)
            b=len(ids[sl])
            qnow=q[qi[ids[sl]]].cuda()
            nominations=(scale*(qnow@src.T)).topk(max(SIZES),1).indices
            qview=source[sl].cuda()
            loc=local[sl].flatten()
            pview=known[loc].cuda()
            pnow=p[unique[loc]].cuda()
            for size in SIZES:
                wc=nominations[:,:size]
                qsigs=descriptors(qview,qnow,src[wc],wc,"source")
                wc2=wc.repeat_interleave(len(PAIR_NAMES)*2,0)
                psigs=descriptors(pview,pnow,tgt[wc2],wc2,"target")
                qmany={k:v.repeat_interleave(len(PAIR_NAMES)*2,0) for k,v in qsigs.items()}
                measured=match(qmany,psigs)
                for name,v in measured.items():
                    ranked=v.reshape(b,len(PAIR_NAMES),2)
                    diff[f"{size}/{name}"].append((ranked[:,:,0]-ranked[:,:,1]).cpu())
        labels=y[ids]
        deltas={k:torch.cat(v) for k,v in diff.items()}
        groups={name:(i,pair[:,i,0]>=0) for i,name in enumerate(PAIR_NAMES)}
        groups["top1_wrong"]=(0,~correct_top)
        groups["top1_correct"]=(0,correct_top)
        summary[s]={name:{k:stats(t[:,j],ok,labels) for k,t in deltas.items()} for name,(j,ok) in groups.items()}
        out[s]={"query_ids":ids,"query_labels":labels,"positive_negative_ids":pair,"delta":deltas,"original_top1_correct":correct_top}
        print("fixed-pair test",s,{m:round(summary[s]["first_wrong_above_right"]["16/"+m]["select_relevant"],4)
              for m in ("interaction","permuted","marginals","pearson","without_reference")},flush=True)
    f=root/"pair-diagnosis.pt"
    save_tensor(out,f)
    atomic_json(f.with_suffix(".json"),{"revision":revision(),"checks":invariants,"method_names":ARMS,
                  "pair_order":PAIR_NAMES,"summary":summary,"per_query_sha256":checksum(f),
                  "note":"fixed relevant and irrelevant photos from the unchanged ranker; closures are descriptive measures, not official retrieval metrics"})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
