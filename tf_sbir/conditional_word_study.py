"""Evaluate lexical path marginalization, its telescoping null, and mass-matched switching controls."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
import os
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,checksum,read
from .conditional_agreement_study import Replay
from .conditional_word_inputs import FAMILY,load
from .conditional_word_path import MODES,Recipe,forward,grid,outputs
from .crossdomain_benchmark import digest,revision
from .crossdomain_inputs import CHECKPOINT
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .query_partition_inputs import place
from .query_partition_study import locked,save_lock,rank_metrics,mean_tables
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import stratified_sample
from .retrieval_evidence import load_kd,paired_delta


def picks(tables,detailed):
    a,b=tables
    if a.keys()!=b.keys():raise ValueError("Different Sketchy candidates")
    def best(names):
        return max(sorted(names),key=lambda k:tuple((a[k][i]+b[k][i])/2 for i in (0,1)))
    parsed={k:Recipe.parse(k) for k in a if not k.startswith("base/")}
    chosen={};short={"base/direct","base/regional"}
    for scope in ("cell","regional","regional-cell"):
        candidates={k:r for k,r in parsed.items() if r.readout==scope}
        for mode in MODES:
            eligible=[k for k,r in candidates.items() if r.mode==mode]
            if not eligible:continue
            pick=best(eligible);chosen[scope+"/"+mode]=pick;short.add(pick)
            if detailed and mode in ("cv","cv-info","stay","new","half","retain80","cv-flat","info-flat"):
                for level in sorted({parsed[k].depth for k in eligible}):
                    short.add(best([k for k in eligible if parsed[k].depth==level]))
        chosen[scope+"/best"]=best(candidates)
        chosen[scope+"/adaptive"]=best([k for k,r in candidates.items() if r.mode in ("cv","cv-info","cv-bg")])
    for name in (chosen["cell/cv-info"],chosen["cell/cv"],chosen["cell/adaptive"],chosen["regional/adaptive"],chosen["regional-cell/adaptive"]):
        r=parsed[name]
        for mode in MODES:
            tag=Recipe(r.size,r.depth,mode,r.readout,r.within,1. if mode=="stay" else r.gain).label()
            if tag in parsed:short.add(tag)
    return chosen,sorted(short)


def selected_for_audit(picked,scope):
    rows={"base/direct","base/regional"}
    highlights=[picked[k] for k in ("cell/cv-info","cell/cv","cell/best","regional/adaptive","regional-cell/adaptive")]
    rows.update(highlights)
    for prefix in ("cell","regional","regional-cell"):
        for m in ("stay","new","half","cv-flat","info-flat","adaptive","best"):
            rows.add(picked[prefix+"/"+m])
    for tag in highlights:
        r=Recipe.parse(tag)
        for mode in ("stay","cv-flat","info-flat","info-new","cv-reverse","cv-shuffle","new"):
            v=Recipe(r.size,r.depth,mode,r.readout,r.within,1. if mode=="stay" else r.gain).label()
            if v in scope:rows.add(v)
    return sorted(rows),list(dict.fromkeys(highlights))


@torch.inference_mode()
def run(w,ds,stage):
    root=w/"results"/FAMILY;root.mkdir(parents=True,exist_ok=True)
    if stage=="lock":
        sl=locked(root/"shortlist.json");p,short=picks(mean_tables(root,"scan-"),False)
        rows,leaders=selected_for_audit(p,set(sl["config"]["scan"]))
        cfg={"rows":rows,"leaders":leaders,"picked":p,"primary":p["cell/cv-info"],"code":revision(),"pilot":sl["sha256"],
             "scan_sha":{s:checksum(root/("scan-"+s+".json")) for s in ("sketchy_1","sketchy_2")},"selection":"equal mean Sketchy mAP then precision, locked before TU"}
        print("frozen",save_lock(root/"freeze.json",cfg),len(rows),flush=True);return
    if ds=="tuberlin" and stage!="official":raise ValueError("TU is reserved for the locked evaluation")
    official=stage in ("pilot-official","official")
    if stage=="pilot":
        rows=grid();spec={"code":revision(),"grid":rows,"subset":"36 per class, seed 63241","model":CHECKPOINT,"KD":KD_FUNCTION_SHA,
              "primary":"cell/cv-info","null":"telescope to same retained word mixture, all nodes","selection":"equal mAP of splits before mean precision"}
        signature=save_lock(root/"protocol.json",spec);prefix="pilot-"
    else:
        lock=locked(root/("freeze.json" if stage=="official" else "shortlist.json"))
        signature=lock["sha256"];rows=lock["config"][{"pilot-official":"audit","scan":"scan","official":"rows"}[stage]]
        prefix="" if stage=="official" else stage+"-"
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Wrong KD evaluator")
    cfg,allq,qs,allp,gs,query,_,source,_,scale=semantic_inputs(w,ds)
    torch.set_float32_matmul_precision("highest")
    for s,(qi,labels) in qs.items():
        dest=root/(prefix+s+".json")
        if dest.is_file():
            prev=read(dest)
            if prev["signature"]!=signature or checksum(dest.with_suffix(".pt"))!=prev["per_query_sha256"]:raise ValueError("Corrupt stored retrieval result")
            continue
        tree,levels,closed,word_mean=load(w,s)
        ix=stratified_sample(labels,36,63241) if stage.startswith("pilot") else torch.arange(len(qi))
        y=labels[ix];pi,glabel=gs[s];gt=glabel.cuda()
        read_tensor=lambda f:torch.load(f,mmap=True,weights_only=True)
        lex=read_tensor(place(w,s)/"lex.pt")
        reg=read_tensor(w/"cache/gallery-stable-region-evidence"/(s+"-regional-reference.pt"))
        if (lex.shape!=reg.shape or lex.shape!=(len(qi),len(pi))):raise ValueError("Wrong baseline sizes")
        scores,top,lead=defaultdict(list),defaultdict(list),defaultdict(list)
        replay=Replay(kd,glabel,s) if official else None
        rng=torch.randperm(len(pi),generator=torch.Generator().manual_seed(77419)).to(gt.device)
        max_kd=0.;max_identity=0.;max_mass=0.;cover=defaultdict(list)
        one=None
        with (ThreadPoolExecutor(max_workers=min(8,len(os.sched_getaffinity(0)))) if replay else nullcontext()) as pool:
            for off in range(0,len(ix),32):
                sl=slice(off,off+32);idx=ix[sl];q=query[qi[idx]].cuda()
                logu=scale*(q@source.T)+word_mean.clamp_min(1e-32).log()[None]
                l=lex[idx].cuda();r=reg[idx].cuda();yy=y[sl];yc=yy.cuda()
                if off==0:
                    ids=logu[:1].argmax(1,keepdim=True);u=logu.new_ones(1,1)
                    panel,e,con=forward(tree,levels,closed,ids,u,{"stay","new","cv","cv-info","half","root-info"},{0,1})
                    one=max(float((v-panel[(depth,"stay")]).abs().amax()) for (depth,mode),v in panel.items())
                    if one>1e-5:raise ValueError("With one word, resampling must leave the ranking unchanged")
                tie={k:rng[x[:,rng].argsort(dim=1,descending=True,stable=True)] for k,x in (("direct",l),("regional",r))}
                for key,mat,base,checks in outputs(tree,levels,closed,logu,l,r,rows):
                    if checks:
                        e,cons,mass=checks;max_identity=max(max_identity,e);max_mass=max(max_mass,cons)
                        if key.startswith(("32/0/stay/","128/0/stay/")):cover[key.split("/")[0]].append(mass)
                    val,order,good=rank_metrics(mat,tie[base],yc,gt,s)
                    if replay:
                        exact=replay.score(order,yy,pool);err=float((exact-val.cpu()).abs().amax())
                        if err>2e-6:raise ValueError("Original KD-SBIR metric mismatch")
                        max_kd=max(max_kd,err);val=exact;lead[key].append(order[:,:5].int().cpu())
                    scores[key].append(val.cpu());top[key].append(good.cpu())
                if off%768==0:print(stage,s,off,len(ix),len(rows),flush=True)
        scores={k:torch.cat(v) for k,v in scores.items()};top={k:torch.cat(v) for k,v in top.items()}
        if set(scores)!=set(rows):raise ValueError("Missing lexical control")
        save_tensor({"ids":ix,"labels":y,"metrics":scores,"top1":top,"first_five":{k:torch.cat(v) for k,v in lead.items()}},dest.with_suffix(".pt"))
        meta={"signature":signature,"code":revision(),"queries":len(ix),"gallery":len(pi),"classes":len(y.unique()),
              "values":{k:v.double().mean(0).tolist() for k,v in scores.items()},"mass_conservation":max_mass,"single_word_max_error":one,
              "closed_form_max_log_error":max_identity,"average_retained_root_mass":{k:sum(z)/len(z) for k,z in cover.items()},
              "original_evaluator":official,"KD_max_error":max_kd if official else None,"per_query_sha256":checksum(dest.with_suffix(".pt"))}
        if official:
            leaders=lock["config"]["leaders"];cls=y.unique(return_inverse=True)[1]
            meta["paired"]={k:{b:paired_delta(scores[k],scores[b],cls) for b in rows} for k in leaders}
            meta["top1"]={k:{b:[int((top[k]&~top[b]).sum()),int((~top[k]&top[b]).sum())] for b in rows} for k in leaders}
        atomic_json(dest,meta);print("saved",stage,s,flush=True)
        del tree,levels,closed
    if stage=="pilot":
        picked,scan=picks(mean_tables(root,prefix),True);audit,leaders=selected_for_audit(picked,set(scan))
        print("shortlist",save_lock(root/"shortlist.json",{"scan":scan,"audit":audit,"leaders":leaders,"picked":picked,"protocol":signature}),len(scan),len(audit),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    p.add_argument("--stage",choices=("pilot","pilot-official","scan","lock","official"),required=True)
    a=p.parse_args();run(a.workspace,a.dataset,a.stage)
