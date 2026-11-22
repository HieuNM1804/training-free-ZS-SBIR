"""Trace where the shared-concept signal is lost, using source photo IDs only after retrieval."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json, checksum
from .concept_response_geometry import ARMS, descriptors
from .crossdomain_benchmark import revision
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .response_geometry_diagnosis import stats
from .response_geometry_views import FAMILY, inputs_plan, read_photos, query_pack, study_base


def examples(qp,pp,qi,pi,y,g,ids,baseline):
    pool={}
    for j in range(len(pi)):
        path=Path(pp[int(pi[j])])
        pool.setdefault((path.parent.name,path.stem),[]).append(j)
    use,triples,rank_distances=[],[],[]
    for ix,idx in enumerate(ids.tolist()):
        sk=Path(qp[int(qi[idx])])
        stem,sep,last=sk.stem.rpartition("-")
        known=pool.get((sk.parent.name,stem),[])
        if not sep or not last.isdigit() or len(known)!=1:continue
        exact=known[0]
        if g[exact]!=y[idx]:raise ValueError("Verified object has the wrong split category")
        order=baseline[ix].argsort(descending=True,stable=True)
        rank=torch.empty_like(order).scatter_(0,order,torch.arange(len(order)))
        distance=(rank-rank[exact]).abs()
        alt=torch.where((g==y[idx]) & (torch.arange(len(g))!=exact),distance,len(order)+1).argmin()
        wrong=torch.where(g!=y[idx],distance,len(order)+1).argmin()
        use.append(ix)
        triples.append([int(pi[exact]),int(pi[alt]),int(pi[wrong])])
        rank_distances.append([int(rank[exact]),int(rank[alt]),int(rank[wrong])])
    return torch.tensor(use),torch.tensor(triples),torch.tensor(rank_distances)


def measure(left,right):
    return {k:((left[k]*right[k]).mean(-1) if k in ("interaction","permuted","without_reference","without_projection")
               else F.cosine_similarity(left[k],right[k],dim=-1,eps=1e-8))
            for k in ARMS if left[k].shape[-1]==right[k].shape[-1]}


@torch.inference_mode()
def run(w):
    inputs=semantic_inputs(w,"sketchy")
    _,qp,qs,pp,gs,q,p,src,tgt,scale=inputs
    folder,plan=inputs_plan(w,"sketchy",inputs)
    root=w/"results"/FAMILY
    data,summary={},{}
    for s,sp in plan["splits"].items():
        ids=sp["ids"]
        qi,y=qs[s]
        pi,g=gs[s]
        base=study_base(w,s,inputs,ids)
        select,triples,rank=examples(qp,pp,qi,pi,y,g,ids,base)
        candidates,inv=torch.unique(triples,return_inverse=True,sorted=True)
        pv=read_photos(folder,plan,candidates)
        qv=query_pack(w,s,inputs,ids)[select]
        labels=y[ids[select]]
        records=defaultdict(list)
        for start in range(0,len(select),12):
            sl=slice(start,start+12)
            native=q[qi[ids[select[sl]]]].cuda()
            words=(scale*(native@src.T)).topk(16,1).indices
            b=len(words)
            qsig=descriptors(qv[sl].cuda(),native,src[words],words,"source")
            taken=inv[sl].flatten()
            px=pv[taken].cuda()
            z=p[candidates[taken]].cuda()
            names=words.repeat_interleave(3,0)
            dest=descriptors(px,z,tgt[names],names,"target")
            qmany={k:v.repeat_interleave(3,0) for k,v in qsig.items()}
            for k,quant in measure(qmany,dest).items():
                xy=quant.view(b,3)
                records["sketch_source_vs_other_same_class/"+k].append((xy[:,0]-xy[:,1]).cpu())
                records["sketch_source_vs_rank_matched_wrong/"+k].append((xy[:,0]-xy[:,2]).cpu())
                records["sketch_other_same_class_vs_wrong/"+k].append((xy[:,1]-xy[:,2]).cpu())
            for label,qview,pview,qa,pa in (("photo_mirrored_instance",px[::3,:,:1],px[:,:,1:],((0,9),(9,13)),((0,9),(9,13))),
                                             ("photo_cross_scale_instance",px[::3],px,((0,9),),((9,13),))):
                one=descriptors(qview,z[::3],tgt[words],words,"source",qa)
                other=descriptors(pview,z,tgt[names],names,"target",pa)
                repeated={k:val.repeat_interleave(3,0) for k,val in one.items()}
                for k,quant in measure(repeated,other).items():
                    xy=quant.view(b,3)
                    records[label+"/"+k].append((xy[:,0]-xy[:,1]).cpu())
        deltas={k:torch.cat(v) for k,v in records.items()}
        mask=torch.ones(len(select),dtype=torch.bool)
        summary[s]={"pilot_queries":len(ids),"verified_unique_source":len(select),"category_count":int(labels.unique().numel()),
                    "median_abs_rank_gap_other_same_class":float((rank[:,1]-rank[:,0]).abs().float().median()),
                    "median_abs_rank_gap_wrong":float((rank[:,2]-rank[:,0]).abs().float().median()),
                    "choices":{k:stats(v,mask,labels) for k,v in deltas.items()}}
        data[s]={"query_ids":ids[select],"labels":labels,"photo_ids_source_other_positive_other_negative":triples,
                 "baseline_ranks_zero_based":rank,"differences":deltas}
        print("transfer localization",s,len(select),{k:round(v["select_relevant"],4) for k,v in summary[s]["choices"].items()
              if k.endswith("/interaction")},flush=True)
    dest=root/"instance-transfer-audit.pt"
    save_tensor(data,dest)
    atomic_json(dest.with_suffix(".json"),{"revision":revision(),"vocabulary_size_per_query":16,"splits":summary,
                "per_query_sha256":checksum(dest),"design":"verified sketch-to-source filenames used only for diagnostic outcomes. Competitor photos are the same-class and different-class items nearest in original rank to that exact photo. Identical 16 sketch-nominated public names in all conditions. Mirror views overlap exactly in content. Different scales may still see overlapping pixels."})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
