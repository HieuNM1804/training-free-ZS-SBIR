"""Attribute actual top-rank changes to image-specific or layout-specific fragment responses under matched scoring."""
import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json,checksum,read
from .crossdomain_benchmark import digest,revision
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .query_fragment_likelihood import GalleryWalk
from .query_fragment_reference_inputs import FAMILY,dataset,load
from .query_fragment_reference_study import split_context


GAMMA=.8
TAU=64.
REFERENCES=("native","mean_colour","empty_token","area_shuffle","permuted","appearance")
ORDERS=("direct","walk")
COMPARISONS=("native","mean_colour","empty_token","area_shuffle","permuted")


def views(bank,indices):
    names=("appearance","mean_colour","empty_token","area_shuffle","permuted_fragments_A","permuted_fragments_B")
    refs={k:bank["modes"][k][indices].float().mean(1).cuda() for k in names}
    refs["permuted"]=.5*(refs.pop("permuted_fragments_A")+refs.pop("permuted_fragments_B"))
    refs["native"]=torch.zeros_like(refs["appearance"])
    return refs


def details(target,base,yy,photo_labels):
    # Label comparisons only take place after every query/gallery score has been produced.
    mask=yy[:,None]==photo_labels
    first_pos=base.masked_fill(~mask,-torch.inf).argmax(1,keepdim=True)
    first_neg=base.masked_fill(mask,-torch.inf).argmax(1,keepdim=True)
    target_top=target.argmax(1);base_top=base.argmax(1)
    correct=yy==photo_labels[base_top]
    target_correct=yy==photo_labels[target_top]
    before=(base.gather(1,first_pos)-base.gather(1,first_neg)).squeeze(1)
    after=(target.gather(1,first_pos)-target.gather(1,first_neg)).squeeze(1)
    return {"top":torch.stack((base_top,target_top,first_pos[:,0],first_neg[:,0]),1).int(),
            "correct":torch.stack((correct,target_correct),1),"gap":torch.stack((before,after),1)}


def report(rows,labels,qfiles,pfiles,classes):
    was,now=rows["correct"].T
    before,after=rows["gap"].T;delta=after-before
    repaired=~was&now;lost=was&~now
    counts={}
    for flag,name in ((~was,"original_errors"),(was,"original_correct")):
        counts[name]={"count":int(flag.sum()),"pair_gap_increased":int((flag&(delta>0)).sum()),
                      "pair_gap_decreased":int((flag&(delta<0)).sum()),"correct_pair_after":int((flag&(after>0)).sum())}
    examples={}
    for flag,name in ((repaired,"repair"),(lost,"loss")):
        ids=torch.where(flag)[0].tolist()
        if not ids:continue
        # Select a typical gap change within repairs/losses with an index tie-break.
        idx=sorted(ids,key=lambda k:(float(delta[k]),k))[len(ids)//2]
        b,t,_,_=rows["top"][idx].tolist()
        examples[name]={"id":idx,"query":qfiles[idx],"first_before":pfiles[b],"first_after":pfiles[t],
                        "gap_before":float(before[idx]),"gap_after":float(after[idx])}
    return {"top1_fixed":int(repaired.sum()),"top1_broken":int(lost.sum()),"top1_before":float(was.float().mean()),
            "top1_after":float(now.float().mean()),"fixed_pair":counts,"examples":examples,
            "by_class":{name:[int((repaired&(labels==i)).sum()),int((lost&(labels==i)).sum())] for i,name in enumerate(classes)}}


def relative(p):
    return "/".join(Path(p).parts[-3:])


@torch.inference_mode()
def run(w,ds):
    root=w/"results"/FAMILY
    freeze=read(root/"freeze.json")
    if digest(freeze["config"])!=freeze["sha256"] or freeze["config"]["pure"]["best"]!="geometric/64/0.8/walk_after_ratio":
        raise ValueError("This attribution belongs to the frozen single-vector scorer")
    inputs=dataset(w,ds);bank=load(w,ds,"full")
    _,allq,qs,allp,gs,_,_=inputs
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    for s in qs:
        out=root/("rank-audit-"+s+".json")
        if out.is_file():
            if read(out)["freeze"]!=freeze["sha256"] or checksum(out.with_suffix(".pt"))!=read(out)["per_query_sha256"]:
                raise ValueError("Rank-audit checkpoint mismatch")
            continue
        p=split_context(w,s,ds,inputs,bank,"official",False)
        walk=GalleryWalk(p["gallery"]);pl=p["gl"].cuda()
        raw=defaultdict(lambda:defaultdict(list));geometry=defaultdict(list)
        equiv=0.
        for off in range(0,len(p["ids"]),128):
            sl=slice(off,off+128);query=p["q"][sl];ys=p["labels"][sl].cuda()
            refs=views(bank,p["pos"][sl])
            direct={k:(query-GAMMA*refs[k])@p["gallery"].T for k in REFERENCES}
            propagated={k:walk.log(TAU*value) for k,value in direct.items()}
            for name in COMPARISONS:
                for order,score in zip(ORDERS,(direct,propagated)):
                    x=details(score["appearance"],score[name],ys,pl)
                    for key,val in x.items():raw[name+"/"+order][key].append(val.cpu())
                if name=="native":continue
                diff=refs["appearance"]-refs[name]
                manual=-GAMMA*(diff@p["gallery"].T)
                err=float((direct["appearance"]-direct[name]-manual).abs().amax())
                if err>1e-5:raise ValueError("Score mediation identity failed")
                equiv=max(equiv,err)
                geometry[name].append(torch.stack((F.cosine_similarity(refs["appearance"],refs[name]),
                    diff.norm(dim=1),manual.std(1,unbiased=False)/(direct["native"].std(1,unbiased=False).clamp_min(1e-12))),1).cpu())
        joined={k:{j:torch.cat(v) for j,v in items.items()} for k,items in raw.items()}
        geom={k:torch.cat(v) for k,v in geometry.items()}
        qi,labels=qs[s];gi,_=gs[s]
        qfiles=[relative(allq[int(x)]) for x in qi.tolist()];pfiles=[relative(allp[int(x)]) for x in gi.tolist()]
        summary={k:report(v,labels,qfiles,pfiles,UNSEEN_CLASSES[s]) for k,v in joined.items()}
        save_tensor({"ids":p["ids"],"labels":labels,"pairs":joined,"geometry":geom},out.with_suffix(".pt"))
        atomic_json(out,{"code":revision(),"freeze":freeze["sha256"],"protocol":"same gamma=.8 and tau=64 for every comparison; after-score attribution only",
            "queries":len(qi),"gallery":len(gi),"classes":len(UNSEEN_CLASSES[s]),"pairs":summary,"linear_identity_error":equiv,
            "geometry_mean_cosine_distanceNorm_relativeScoreStd":{k:v.mean(0).tolist() for k,v in geom.items()},"per_query_sha256":checksum(out.with_suffix(".pt"))})
        print("audited",s,len(qi),flush=True)
        del p,walk


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--dataset",choices=("sketchy","tuberlin"),required=True)
    a=parser.parse_args();run(a.workspace,a.dataset)
