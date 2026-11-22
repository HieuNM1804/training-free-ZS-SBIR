"""Use verified sketches of the same objects to separate actual depiction change from merely changing prompt words."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum
from .concept_response_geometry import descriptors
from .crossdomain_benchmark import revision
from .encoding import save_tensor
from .gallery_panel_resume import region_views, semantic_inputs
from .response_geometry_diagnosis import stats
from .response_geometry_instance_audit import measure
from .response_geometry_views import FAMILY, inputs_plan, query_pack, read_photos, study_base


def paired_objects(qp,pp,qi,pi,labels,gl,ids,base):
    by_photo=defaultdict(list)
    by_sketch=defaultdict(list)
    for i,idx in enumerate(pi.tolist()):
        f=Path(pp[idx])
        by_photo[(f.parent.name,f.stem)].append(i)
    for idx in qi.tolist():
        f=Path(qp[idx])
        stem,sep,last=f.stem.rpartition("-")
        if sep and last.isdigit():by_sketch[(f.parent.name,stem)].append(idx)
    allowed={entries[0]:name for name,entries in by_photo.items() if len(entries)==1 and name in by_sketch}
    allowed_ids=torch.tensor(sorted(allowed))
    photo,draw,keep=[],[],[]
    rng=torch.Generator().manual_seed(29137)
    def pick(values):
        return values[int(torch.randint(len(values),(1,),generator=rng))]
    for j,i in enumerate(ids.tolist()):
        idx=int(qi[i])
        f=Path(qp[idx])
        name=(f.parent.name,f.stem.rpartition("-")[0])
        positions=by_photo.get(name,[])
        siblings=[a for a in by_sketch.get(name,[]) if a!=idx]
        if len(positions)!=1 or not siblings:continue
        me=positions[0]
        yes=gl[allowed_ids]==labels[i]
        alt=allowed_ids[yes & (allowed_ids!=me)]
        negative=allowed_ids[~yes]
        if not len(alt) or not len(negative):continue
        order=base[j].argsort(descending=True,stable=True)
        rank=torch.empty_like(order).scatter_(0,order,torch.arange(len(order)))
        far=(rank-rank[me]).abs()
        a=int(alt[far[alt].argmin()])
        b=int(negative[far[negative].argmin()])
        group=(me,a,b)
        photo.append([int(pi[p]) for p in group])
        draw.append([pick(siblings),pick(by_sketch[allowed[a]]),pick(by_sketch[allowed[b]])])
        keep.append(j)
    return torch.tensor(keep),torch.tensor(photo),torch.tensor(draw)


@torch.inference_mode()
def run(w):
    inputs=semantic_inputs(w,"sketchy")
    cfg,qp,qs,pp,gs,q,p,src,tgt,scale=inputs
    folder,plan=inputs_plan(w,"sketchy",inputs)
    root=w/"results"/FAMILY
    data,reports={},{}
    for s,row in plan["splits"].items():
        ids=row["ids"]
        qi,y=qs[s]
        pi,gl=gs[s]
        base=study_base(w,s,inputs,ids)
        keep,pi3,si3=paired_objects(qp,pp,qi,pi,y,gl,ids,base)
        photos,pix=pi3.unique(return_inverse=True,sorted=True)
        sketches,six=si3.unique(return_inverse=True,sorted=True)
        pv=read_photos(folder,plan,photos)
        sv=region_views(w,s+"-diagnostic-redraw",cfg,qp,sketches,q)[0]["regions"]
        qv=query_pack(w,s,inputs,ids)[keep]
        accumulator=defaultdict(list)
        for off in range(0,len(keep),24):
            sl=slice(off,off+24)
            qid=qi[ids[keep[sl]]]
            source=q[qid].cuda()
            words=(scale*(source@src.T)).topk(16,1).indices
            names=words.repeat_interleave(3,0)
            photo_views=pv[pix[sl].flatten()].cuda()
            drawing_views=sv[six[sl].flatten()].cuda()
            nativep=p[pi3[sl].flatten()].cuda()
            natives=q[si3[sl].flatten()].cuda()
            # Probe all physical conditions with the same sketch-nominated names.
            query_visual={"sketch":(qv[sl,:,:1].cuda(),source),"photo":(photo_views[::3,:,:1],nativep[::3])}
            other_visual={"sketch":(drawing_views[:,:,1:],natives),"photo":(photo_views[:,:,1:],nativep)}
            left,right={},{}
            for v,(features,vector) in query_visual.items():
                for text,t in (("sketch",src),("photo",tgt)):
                    found=descriptors(features,vector,t[words],words,"source")
                    left[v,text]={k:a.repeat_interleave(3,0) for k,a in found.items()}
            for v,(features,vector) in other_visual.items():
                for text,t in (("sketch",src),("photo",tgt)):
                    right[v,text]=descriptors(features,vector,t[names],names,"target")
            for (a,pa),qsigs in left.items():
                for (b,pb),psigs in right.items():
                    compared=measure(qsigs,psigs)
                    for mode in ("interaction","without_reference","permuted","marginals","mean","pearson"):
                        z=compared[mode].view(-1,3)
                        name=f"{a}-{b}/{pa}-{pb}/{mode}"
                        accumulator[name].append(torch.stack((z[:,0]-z[:,1],z[:,0]-z[:,2],z[:,1]-z[:,2]),1).cpu())
            for a,(_,x) in query_visual.items():
                for b,(_,z) in other_visual.items():
                    similarity=(x[:,None]*z.view(len(x),3,-1)).sum(-1)
                    accumulator[f"{a}-{b}/native/cosine"].append(torch.stack((similarity[:,0]-similarity[:,1],
                                             similarity[:,0]-similarity[:,2],similarity[:,1]-similarity[:,2]),1).cpu())
        labels=y[ids[keep]]
        mask=torch.ones(len(labels),dtype=torch.bool)
        fields=("exact_vs_same_class","exact_vs_wrong_class","other_same_class_vs_wrong")
        deltas={k:torch.cat(rows) for k,rows in accumulator.items()}
        score={k:{field:stats(z[:,j],mask,labels) for j,field in enumerate(fields)} for k,z in deltas.items()}
        reports[s]={"queries":len(labels),"source_and_competitors":len(photos),"unique_other_drawings":len(sketches),"scores":score}
        data[s]={"labels":labels,"query_ids":ids[keep],"gallery_photos":pi3,"diagnosis_only_other_sketches":si3,"deltas":deltas}
        print(s,len(labels),"exact versus different instance of the same class",
              {k:round(v[fields[0]]["select_relevant"],4) for k,v in score.items() if k.endswith("sketch-photo/interaction")},flush=True)
    path=root/"actual-depiction-factorial.pt"
    save_tensor(data,path)
    atomic_json(path.with_suffix(".json"),{"revision":revision(),"splits":reports,"per_query_sha256":checksum(path),
               "scope":"post-hoc only; verified source photographs, different actual drawings of the same object, same-category alternative and off-category alternative; same objects for every visual domain/prompt crossing; photo self-control uses mirrored content."})


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(cli.parse_args().workspace)
