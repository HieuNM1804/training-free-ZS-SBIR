"""Build the label-free gallery hierarchy and the direct public-word score used by the resolution experiment."""
import argparse
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,read
from .crossdomain_benchmark import digest,revision,SPLIT_HASHES
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .query_fragment_reference_inputs import load as fragment_bank
from .query_partition_tree import Hierarchy,enumerate_check,folds
from .recoverable_patch_study import locate
from .retrieval_diagnosis import stratified_sample


FAMILY="query-conditioned-gallery-partition"
SOURCES=("lex","native","fragments","wordwalk","regional")
FOLD_SEED=14891


def source_inputs(w,ds):
    cfg,qp,qs,pp,gs,q,p,src,tgt,scale=semantic_inputs(w,ds)
    bank=fragment_bank(w,ds,"full")
    f=bank["modes"]["appearance"].float().mean(1)
    if not torch.equal(bank["ids"],torch.arange(len(q))):raise ValueError("Missing native query response")
    if (1-(bank["native"]*q).sum(1)).abs().amax()>.002:raise ValueError("Different source embeddings")
    return qp,qs,pp,gs,q,p,f,src,tgt,scale


def place(w,s):
    return w/"cache"/FAMILY/s


@torch.inference_mode()
def prepare(w,ds):
    if ds=="tuberlin" and not (w/"results"/FAMILY/"freeze.json").is_file():raise ValueError("No TU before freeze")
    qp,qs,pp,gs,q,p,f,src,tgt,scale=source_inputs(w,ds)
    for s,(qi,y) in qs.items():
        root=place(w,s);root.mkdir(parents=True,exist_ok=True)
        pi,gl=gs[s];g=p[pi].cuda()
        sig=digest({"model":CHECKPOINT,"paths":[pp[j] for j in pi.tolist()],"queries":[qp[j] for j in qi.tolist()],
                    "version":"first_neighbor_all_galleries_v1","dictionary_size":len(src),"score":"log sum shared public word posterior"})
        if (root/"done.json").exists():
            if read(root/"done.json")["signature"]!=sig:raise ValueError("Gallery changed")
            continue
        torch.set_float32_matmul_precision("highest")
        tree=Hierarchy.build(g)
        save_tensor({"ids":[z.cpu() for z in tree.ids],"parents":[z.cpu() for z in tree.parents],"sizes":tree.sizes},root/"tree.pt")
        print("tree",s,tree.sizes,flush=True)
        # Keep the full open vocabulary and the unmodified prompt pair used for the incidence baseline.
        probs=g.new_empty(len(g),len(tgt))
        torch.set_float32_matmul_precision("high")
        for i in range(0,len(g),256):probs[i:i+256]=(scale*(g[i:i+256]@tgt.T)).softmax(1)
        out=torch.empty((len(qi),len(pi)))
        for i in range(0,len(qi),192):
            post=(scale*(q[qi[i:i+192]].cuda()@src.T)).softmax(1)
            out[i:i+192]=(post@probs.T).clamp_min(1e-32).log().cpu()
        save_tensor(out,root/"lex.pt")
        torch.set_float32_matmul_precision("highest")
        if digest(UNSEEN_CLASSES[s])!=SPLIT_HASHES[s]:raise ValueError("Split identity changed")
        atomic_json(root/"done.json",{"signature":sig,"code":revision(),"algebra_check":enumerate_check(),"tree":tree.sizes,
                    "gallery":len(gl),"queries":len(y),"device":torch.cuda.get_device_name()})
        print("saved inputs",s,flush=True)
        del out,probs,g,tree


def context(w,s,source,stage):
    qp,qs,pp,gs,qraw,praw,f,*_=source
    qi,y=qs[s];pi,gl=gs[s]
    root=place(w,s);info=read(root/"done.json")
    tree=torch.load(root/"tree.pt",mmap=True,weights_only=True)
    tree=Hierarchy([x.cuda() for x in tree["ids"]],[x.cuda() for x in tree["parents"]],tree["sizes"])
    if tree.sizes!=info["tree"]:raise ValueError("Changed tree levels")
    idx=stratified_sample(y,36,63241) if stage=="pilot" else torch.arange(len(y))
    load=lambda p:torch.load(p,mmap=True,weights_only=True)
    fixed={"lex":load(root/"lex.pt"),"regional":load(w/"cache/gallery-stable-region-evidence"/(s+"-regional-reference.pt")),
           "wordwalk":load(w/"cache/pragmatic-gallery-competition"/(s+"-named-rgb16.pt"))}
    return {"s":s,"tree":tree,"fixed":fixed,"ix":idx,"qi":qi[idx],"y":y[idx],"gl":gl,"g":praw[pi].cuda(),
            "q":qraw[qi[idx]].cuda(),"corrected":(qraw[qi[idx]]-.8*f[qi[idx]]).cuda(),
            "qp":[qp[i] for i in qi[idx].tolist()],"pp":[pp[i] for i in pi.tolist()],
            "folds":folds(len(pi),torch.device("cuda"),FOLD_SEED),"info":info}


def query_scores(c,sl):
    x={src:table[c["ix"][sl]].cuda() for src,table in c["fixed"].items()}
    x["wordwalk"]=x["wordwalk"].clamp_min(1e-32).log()
    x["native"]=c["q"][sl]@c["g"].T
    x["fragments"]=c["corrected"][sl]@c["g"].T
    return x


@torch.inference_mode()
def diagnosis(w):
    inputs=source_inputs(w,"sketchy");root=w/"results"/FAMILY;root.mkdir(parents=True,exist_ok=True)
    for s in inputs[1]:
        c=context(w,s,inputs,"pilot");tree=c["tree"];gl=c["gl"].cuda();classes=gl.unique()
        # The diagnostic sees test labels only after the hierarchy and original rankings have been fixed.
        cls=torch.nn.functional.one_hot(gl,len(classes)).float()
        level=[]
        for k,ids in zip(tree.sizes,tree.ids,strict=True):
            count=cls.new_zeros(k,len(classes)).index_add_(0,ids,cls)
            purity=float(count.max(1).values.sum()/len(gl))
            matched=0
            for h in c["folds"]:
                keep=torch.ones(len(gl),dtype=torch.bool,device=gl.device);keep[h]=False
                pool=cls.new_zeros(k,len(classes)).index_add_(0,ids[keep],cls[keep])
                seen=pool[ids[h]].sum(1)>0
                matched+=int((seen&(pool[ids[h]].argmax(1)==gl[h])).sum())
            level.append({"clusters":k,"purity":purity,"held_photo_majority_recovery":matched/len(gl)})
        counts={};examples=[]
        for j in range(0,len(c["ix"]),64):
            sl=slice(j,j+64);scores=query_scores(c,sl);labels=c["y"][sl].cuda()
            yes=labels[:,None]==gl;old=scores["regional"];r=torch.arange(len(labels),device=gl.device)
            pos=old.masked_fill(~yes,-torch.inf).argmax(1);neg=old.masked_fill(yes,-torch.inf).argmax(1)
            wrong=(gl[old.argmax(1)]!=labels)
            counts["regional_top1_errors"]=counts.get("regional_top1_errors",0)+int(wrong.sum())
            for src,z in scores.items():
                key=src+"_rescues_regional_wrong_at_top1"
                counts[key]=counts.get(key,0)+int((wrong&(gl[z.argmax(1)]==labels)).sum())
            for top,ids in enumerate(tree.ids):
                key=f"level{top}_original_positive_negative_separate_on_errors"
                counts[key]=counts.get(key,0)+int((wrong&(ids[pos]!=ids[neg])).sum())
            if len(examples)<4:
                for i in torch.where(wrong&(gl[scores["fragments"].argmax(1)]==labels))[0].tolist()[:4-len(examples)]:
                    examples.append({"query":c["qp"][j+i],"wrong_first":c["pp"][int(neg[i])],"positive":c["pp"][int(pos[i])],
                                     "first_corrected_visual":c["pp"][int(scores["fragments"][i].argmax())]})
        atomic_json(root/("diagnosis-"+s+".json"),{"queries":len(c["ix"]),"gallery":len(gl),"levels":level,"original_errors":counts,
                    "examples":examples,"usage":"post-ranking diagnostic, true labels never form a node"})
        print(s,level,counts,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    p.add_argument("--diagnose",action="store_true");a=p.parse_args()
    prepare(a.workspace,a.dataset)
    if a.diagnose:diagnosis(a.workspace)
