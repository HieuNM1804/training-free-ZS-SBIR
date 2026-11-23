"""Test whether a different visual readout is needed at successive gallery distinctions."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
import os
from pathlib import Path

import torch

from .branch_view_inputs import FAMILY
from .branch_view_policy import POLICIES,Row,grid,outputs
from .concept_benchmark import atomic_json,checksum,read
from .conditional_agreement_study import Replay
from .crossdomain_benchmark import revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .partition_rank_replay import PendingReplay
from .query_fragment_reference_inputs import dataset
from .query_partition_inputs import place
from .query_partition_study import locked,save_lock,rank_metrics,mean_tables
from .query_partition_tree import Hierarchy
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd,paired_delta


def choices(tables,detailed):
    a,b=tables
    if a.keys()!=b.keys():raise ValueError("Split choices differ")
    def best(sub):return max(sorted(sub),key=lambda k:tuple((a[k][i]+b[k][i])/2 for i in (0,1)))
    rs={k:Row.parse(k) for k in a if not k.startswith("base/")}
    selected={};short={"base/direct","base/regional"}
    for kind in ("cell","regional-delta","regional-cell"):
        pool={k:v for k,v in rs.items() if v.readout==kind}
        for policy in POLICIES:
            elig=[k for k,v in pool.items() if v.policy==policy]
            if not elig:continue
            win=best(elig);selected[kind+"/"+policy]=win;short.add(win)
            if detailed and policy in ("cross","commit","uniform","global-commit","global-fresh","same","moved"):
                for depth in sorted({rs[k].depth for k in elig}):short.add(best([k for k in elig if rs[k].depth==depth]))
        selected[kind+"/best"]=best(pool)
    extra=[]
    for k in short:
        if k.startswith("base/") or rs[k].policy!="cross":continue
        r=rs[k]
        for p in POLICIES:
            gamma=1. if p in ("commit","uniform") else r.gamma
            test=Row(r.depth,r.whole,p,gamma,r.readout,r.gain).label()
            if test in rs:extra.append(test)
    return selected,sorted(short|set(extra))


def audit(picked,available):
    leaders=list(dict.fromkeys(picked[x] for x in ("cell/cross","cell/best","regional-cell/cross","regional-delta/cross","regional-cell/best")))
    rows={"base/direct","base/regional",*leaders}
    for name in leaders:
        r=Row.parse(name)
        for p in ("commit","uniform","global-commit","global-fresh","same","moved","wrong-node","agreement","reverse"):
            rows.add(picked[r.readout+"/"+p])
            gamma=1. if p in ("commit","uniform") else r.gamma
            candidate=Row(r.depth,r.whole,p,gamma,r.readout,r.gain).label()
            if candidate in available:rows.add(candidate)
    return sorted(rows),leaders


def literal_report(raw,root,cls):
    valid=raw[:,0,0]>0
    result={"eligible":int(valid.sum()),"classes_with_exact_word":len(cls[valid].unique()),"classes":len(cls.unique())}
    for k in (32,128):
        for name,ranks in (("query",raw),("root",root)):
            one=(ranks[:,0,0]<=k)&valid
            any_view=(ranks.amin((1,2))<=k)&valid
            result[f"{name}_whole_top{k}"]=int(one.sum())
            result[f"{name}_any_view_top{k}"]=int(any_view.sum())
        top_q=raw[:,0,0]<=k;top_root=root[:,0,0]<=k
        result[f"gallery_prior_lost_top{k}"]=int((top_q&~top_root&valid).sum())
        result[f"gallery_prior_added_top{k}"]=int((~top_q&top_root&valid).sum())
    return result


class PrefixReplay(Replay):
    def __init__(self,kd,labels,split):
        super().__init__(kd,labels,split)
        self.capped=split=="sketchy_2"
        if self.capped:self.y=self.y[:200]
    def chunk(self,args):
        order,labels=args
        return super().chunk((order[:,:200] if self.capped else order,labels))


@torch.inference_mode()
def run(w,ds,stage,only):
    root=w/"results"/FAMILY;root.mkdir(parents=True,exist_ok=True)
    if stage=="lock":
        prev=locked(root/"shortlist.json");p,_=choices(mean_tables(root,"scan-"),False);rows,leaders=audit(p,set(prev["config"]["scan"]))
        cfg={"rows":rows,"leaders":leaders,"picked":p,"primary":p["cell/cross"],"code":revision(),"prior":prev["sha256"],
             "scan_sha":{s:checksum(root/("scan-"+s+".json")) for s in ("sketchy_1","sketchy_2")},
             "selection":"both complete exploratory Sketchy splits with equal mAP weights and mean precision as tie-break"}
        print("frozen",save_lock(root/"freeze.json",cfg),len(rows),flush=True);return
    if ds=="tuberlin" and stage!="official":raise ValueError("No TU development")
    ispilot=stage.startswith("pilot");official=stage in ("pilot-official","official")
    if stage=="pilot":
        rows=grid();spec={"code":revision(),"checkpoint":CHECKPOINT,"kd":KD_FUNCTION_SHA,"terms":20931,"grid":rows,
                         "pilot":"36 queries per class, original full galleries","primary":"cell/cross"}
        sign=save_lock(root/"protocol.json",spec)
    else:
        lock=locked(root/("freeze.json" if stage=="official" else "shortlist.json"))
        sign=lock["sha256"];rows=lock["config"][{"official":"rows","pilot-official":"audit","scan":"scan"}[stage]]
    prefix="" if stage=="official" else stage+"-"
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Original KD evaluator changed")
    _,_,qs,_,gs,_,_=dataset(w,ds)
    for s in qs:
        if only and s!=only:continue
        target=root/(prefix+s+".json")
        if target.is_file():
            info=read(target)
            if info["signature"]!=sign or checksum(target.with_suffix(".pt"))!=info["per_query_sha256"]:raise ValueError("Invalid previous result")
            continue
        cache=w/"cache"/FAMILY/s;data=cache/("pilot" if ispilot else "full")
        ready=read(data/"done.json")
        load=lambda path:torch.load(path,weights_only=True,mmap=True)
        g=load(cache/"gallery.pt")
        tree=Hierarchy([x.cuda() for x in g["ids"]],[x.cuda() for x in g["parents"]],g["sizes"])
        lex=load(place(w,s)/"lex.pt");reg=load(w/"cache/gallery-stable-region-evidence"/(s+"-regional-reference.pt"))
        _,yl=qs[s];_,gl=gs[s];gt=gl.cuda()
        perm=torch.randperm(len(gt),generator=torch.Generator().manual_seed(73991)).to(gt.device)
        scores,tops,heads=defaultdict(list),defaultdict(list),defaultdict(list)
        ids=[];lit=[];lroot=[];tracker=defaultdict(list);maxerr=0.;sentinel=0.
        workers=min(8,len(os.sched_getaffinity(0)))
        full_replay=Replay(kd,gl,s) if official and s=="sketchy_2" else None
        with (ThreadPoolExecutor(max_workers=workers) if official else nullcontext()) as pool:
            pending=PendingReplay(PrefixReplay(kd,gl,s),pool,workers) if official else None
            for chunk in sorted(data.glob("[0-9]*.pt")):
                pack=load(chunk)
                lit.append(pack["literal_sketch_rank"]);lroot.append(pack["literal_root_rank"])
                for i in range(0,len(pack["ids"]),64):
                    ix=pack["ids"][i:i+64];ids.append(ix)
                    yy=yl[ix];yc=yy.cuda();l=lex[ix].cuda();r=reg[ix].cuda()
                    tie={n:perm[val[:,perm].argsort(dim=1,descending=True,stable=True)] for n,val in (("direct",l),("regional",r))}
                    trace=None
                    if official:
                        win=Row.parse(lock["config"]["picked"]["cell/cross"]);trace={"policy":(win.whole,"cross",win.gamma)}
                    for k,v,sec in outputs(tree,pack["densities"][i:i+64].cuda(),l,r,rows,trace):
                        value,order,good=rank_metrics(v,tie[sec],yc,gt,s)
                        if pending:
                            pending.submit(order,yy,value,scores[k])
                            if full_replay is not None:
                                # In Ext-2 both official metrics use exactly 200 ranks. Verify a full-size original call for each scorer and batch as well.
                                match=full_replay.chunk((order[:1].int().cpu(),yy[:1]))
                                sentinel=max(sentinel,float((match-value[:1].cpu()).abs().max()))
                                if sentinel>2e-6:raise ValueError("Full-gallery Ext-2 sentinel differed")
                            heads[k].append(order[:,:5].int().cpu())
                        else:scores[k].append(value.cpu())
                        tops[k].append(good.cpu())
                    if trace and "trace" in trace:
                        for depth,v in trace["trace"].items():tracker[depth].append(v)
                if int(chunk.stem)%2304==0:print(stage,s,chunk.stem,ready["queries"],len(rows),flush=True)
            if pending:maxerr=pending.finish()
        m={k:torch.cat(v) for k,v in scores.items()};top={k:torch.cat(v) for k,v in tops.items()}
        ix=torch.cat(ids);yy=yl[ix];rk=torch.cat(lit);rr=torch.cat(lroot)
        if len(ix)!=ready["queries"] or set(m)!=set(rows):raise ValueError("Some rows or queries were missed")
        details={"ids":ix,"labels":yy,"metrics":m,"top1":top,"first_five":{k:torch.cat(v) for k,v in heads.items()},
                 "policy":{k:torch.cat(v) for k,v in tracker.items()}}
        save_tensor(details,target.with_suffix(".pt"))
        meta={"signature":sign,"code":revision(),"queries":len(ix),"gallery":len(gl),"classes":len(yy.unique()),"words":ready["word_count"],
              "values":{k:v.double().mean(0).tolist() for k,v in m.items()},"original_evaluator":official,
              "KD_max_error":max(maxerr,sentinel) if official else None,"full_gallery_sentinel_max_error":sentinel if full_replay else None,
              "posthoc_word_coverage":literal_report(rk,rr,yy),"per_query_sha256":checksum(target.with_suffix(".pt"))}
        if official:
            leaders=lock["config"]["leaders"];cls=yy.unique(return_inverse=True)[1]
            meta["paired"]={k:{b:paired_delta(m[k],m[b],cls) for b in rows} for k in leaders}
            meta["top1"]={k:{b:[int((top[k]&~top[b]).sum()),int((~top[k]&top[b]).sum())] for b in rows} for k in leaders}
            meta["mean_policy_kl_vs_root_full_probability_modal_change"]={k:v.mean(0).tolist() for k,v in details["policy"].items()}
        atomic_json(target,meta);print("saved",stage,s,flush=True)
    if stage=="pilot" and not only:
        p,scan=choices(mean_tables(root,prefix),True);a,leaders=audit(p,set(scan))
        cfg={"picked":p,"scan":scan,"audit":a,"leaders":leaders,"protocol":sign}
        print("shortlist",save_lock(root/"shortlist.json",cfg),len(scan),len(a),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    p.add_argument("--split",choices=("sketchy_1","sketchy_2","tuberlin"))
    p.add_argument("--stage",choices=("pilot","pilot-official","scan","lock","official"),required=True)
    a=p.parse_args();run(a.workspace,a.dataset,a.stage,a.split)
