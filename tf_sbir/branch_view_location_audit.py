"""Keep the exact marginal use of all query views while removing only their dependence on the current gallery branch."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import os
from pathlib import Path

import torch

from .branch_view_inputs import FAMILY
from .branch_view_policy import Row,build_levels,decisions,weights
from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json,checksum,read
from .crossdomain_benchmark import revision
from .encoding import save_tensor
from .partition_rank_replay import PendingReplay
from .query_fragment_reference_inputs import dataset
from .query_partition_inputs import place
from .query_partition_study import locked,rank_metrics
from .query_partition_tree import Hierarchy
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd,paired_delta


SCOPES=("cell","regional-cell","regional-delta")
NULLS=("actual","global","whole","scales")


def constrained(policy,mass,parts):
    # Mean over parent identities using the probability of actually reaching each parent at this step.
    mass=mass/mass.sum(1,keepdim=True).clamp_min(1e-30)
    marginal=(mass[:,:,None]*policy).sum(1,keepdim=True)
    group_mass=policy.new_zeros((*policy.shape[:2],len(parts)))
    group_global=marginal.new_zeros((len(marginal),1,len(parts)))
    result=torch.empty_like(policy)
    for j,group in enumerate(parts):
        group_mass[:,:,j]=policy[:,:,group].sum(2)
        group_global[:,:,j]=marginal[:,:,group].sum(2)
        result[:,:,group]=group_mass[:,:,j,None]*marginal[:,:,group]/group_global[:,:,j,None].clamp_min(1e-30)
    row_error=float((result.sum(2)-1).abs().amax())
    col_error=float(((mass[:,:,None]*result).sum(1,keepdim=True)-marginal).abs().amax())
    group_error=max(float((result[:,:,group].sum(2)-group_mass[:,:,j]).abs().amax()) for j,group in enumerate(parts))
    if max(row_error,col_error,group_error)>3e-5:raise ValueError("Counterfactual policy changed its matching constraints")
    return result,max(row_error,col_error,group_error)


def divergence(p,q):
    return (p*(p.clamp_min(1e-30).log()-q.clamp_min(1e-30).log())).sum(2)


def compare_policies(row,xs,count,rates,signals,tree):
    h=row.whole
    full=xs[0].new_full((1,1,14),(1-h)/13);full[:,:,0]=h
    partitions={"global":[list(range(14))],"whole":[[0],list(range(1,14))],
                "scales":[[0],list(range(1,10)),list(range(10,14))]}
    b=len(xs[0]);states={k:xs[0].new_ones((b,1)) for k in NULLS}
    info=[];bound=0.
    for j in reversed(range(row.depth-1,len(rates))):
        up=tree.parents[j]
        po=weights(full,signals[j]["cross"],row.gamma)
        native_mass=states["actual"]*count[j+1][None]/count[-1]
        native_mass=native_mass/native_mass.sum(1,keepdim=True)
        committed={}
        for k in NULLS:
            if k=="actual":policy=po
            else:
                mass=states[k]*count[j+1][None]/count[-1]
                policy,err=constrained(po,mass,partitions[k]);bound=max(bound,err)
                # Compute information at the original, unaltered incoming probability of the query.
                check,err=constrained(po,native_mass,partitions[k]);bound=max(bound,err)
                committed[k]=(native_mass*divergence(po,check)).sum(1)
            states[k]=states[k][:,up]*(policy[:,up]*rates[j]).sum(2)
            mass=float(((states[k]*count[j][None]/count[-1]).sum(1)-1).abs().amax())
            if mass>2e-4:raise ValueError("Rerouting changed total probability")
        # Chain rule for P -> which window. Conditional information after retaining scale cannot increase.
        if (committed["scales"]-committed["whole"]).max()>1e-5 or (committed["whole"]-committed["global"]).max()>1e-5:
            raise ValueError("Conditional information violates the refinement ordering")
        info.append(torch.stack([committed[k] for k in ("global","whole","scales")],1))
    return {k:v.clamp_min(1e-30).log() for k,v in states.items()},torch.stack(info,1),bound


def readout(row,xs,cells,tree,lex,reg):
    j=row.depth-1;ix=tree.ids[j]
    h=row.whole;prior=xs[j].new_full((1,1,14),(1-h)/13);prior[:,:,0]=h
    committed=(xs[j][:,:,:,:,0].mean(3)*prior).sum(2).clamp_min(1e-30).log()[:,ix]
    whole=xs[j][:,:,0,0,0].clamp_min(1e-30).log()[:,ix]
    val=cells[:,ix]
    if row.readout=="cell":return val+row.gain*(lex-whole)
    if row.readout=="regional-delta":return reg+row.gain*(val-committed)
    ctr=val-val.mean(1,keepdim=True)
    return reg+row.gain*(reg.std(1,keepdim=True,unbiased=False)/ctr.std(1,keepdim=True,unbiased=False).clamp_min(1e-8))*ctr


@torch.inference_mode()
def run(w,ds,split):
    root=w/"results"/FAMILY;freeze=locked(root/"freeze.json");picked=freeze["config"]["picked"]
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Wrong KD evaluator")
    _,qp,qs,pp,gs,_,_=dataset(w,ds)
    for s,(qi,y) in qs.items():
        if split and split!=s:continue
        out=root/("location-"+s+".json")
        if out.is_file():
            data=read(out)
            if data["freeze"]!=freeze["sha256"] or data["per_query_sha256"]!=checksum(out.with_suffix(".pt")):raise ValueError("Changed output")
            continue
        cache=w/"cache"/FAMILY/s
        load=lambda f:torch.load(f,mmap=True,weights_only=True)
        gallery=load(cache/"gallery.pt")
        tree=Hierarchy([v.cuda() for v in gallery["ids"]],[v.cuda() for v in gallery["parents"]],gallery["sizes"])
        full=cache/"full";ready=read(full/"done.json");pi,gl=gs[s];gt=gl.cuda()
        lex=load(place(w,s)/"lex.pt");reg=load(w/"cache/gallery-stable-region-evidence"/(s+"-regional-reference.pt"))
        perm=torch.randperm(len(gl),generator=torch.Generator().manual_seed(73991)).to(gt.device)
        rows={scope:Row.parse(picked[scope+"/cross"]) for scope in SCOPES}
        scores,top,sample,stats=defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
        current=[];before=[];after=[];max_error=0.;constraint_error=0.
        wkr=min(6,len(os.sched_getaffinity(0)))
        with ThreadPoolExecutor(max_workers=wkr) as pool:
            pending=PendingReplay(PrefixReplay(kd,gl,s),pool,wkr)
            for fl in sorted(full.glob("[0-9]*.pt")):
                data=load(fl);before.append(data["literal_root_rank"])
                for i in range(0,len(data["ids"]),64):
                    ix=data["ids"][i:i+64];current.append(ix)
                    yy=y[ix];yc=yy.cuda();l=lex[ix].cuda();r=reg[ix].cuda()
                    x,n,pri=build_levels(data["densities"][i:i+64].cuda(),tree);rates,signals,_=decisions(x,tree,pri)
                    tie={b:perm[z[:,perm].argsort(dim=1,descending=True,stable=True)] for b,z in (("cell",l),("regional",r))}
                    for scope,row in rows.items():
                        maps,information,e=compare_policies(row,x,n,rates,signals,tree)
                        stats[scope].append(information.cpu());constraint_error=max(constraint_error,e)
                        for policy,cells in maps.items():
                            k=scope+"/"+policy
                            v=readout(row,x,cells,tree,l,r)
                            val,rank,correct=rank_metrics(v,tie["cell" if scope=="cell" else "regional"],yc,gt,s)
                            pending.submit(rank,yy,val,scores[k]);top[k].append(correct.cpu());sample[k].append(rank[:,0].int().cpu())
                if int(fl.stem)%3072==0:print(s,fl.stem,len(y),flush=True)
            max_error=pending.finish()
        qix=torch.cat(current)
        if not torch.equal(qix,torch.arange(len(y))) or len(y)!=ready["queries"]:raise ValueError("Incomplete gallery query audit")
        m={k:torch.cat(v) for k,v in scores.items()};t={k:torch.cat(v) for k,v in top.items()};lead={k:torch.cat(v) for k,v in sample.items()}
        incl=torch.cat(before);labeled=incl[:,0,0]>0
        whole=labeled&(incl[:,0,0]<=32);anyv=labeled&(incl.amin((1,2))<=32)
        masks={"whole_top32":whole,"only_additional_view_top32":anyv&~whole,"no_view_top32":labeled&~anyv}
        cls=y.unique(return_inverse=True)[1]
        comparisons={};examples={}
        for scope in SCOPES:
            a=scope+"/actual"
            comparisons[scope]={}
            for policy in NULLS[1:]:
                b=scope+"/"+policy;dif=(m[a]-m[b]).double()*100
                fixed=t[a]&~t[b];broken=t[b]&~t[a]
                comparisons[scope][policy]={"paired":paired_delta(m[a],m[b],cls),"top1_fixed_broken":[int(fixed.sum()),int(broken.sum())],
                    "by_observed_literal_rank":{k:{"n":int(mask.sum()),"delta_mAP_pp":float(dif[mask,0].mean()) if mask.any() else None} for k,mask in masks.items()}}
                if policy=="scales":
                    selected=[]
                    for name,ind in (("fix",fixed),("loss",broken)):
                        ids=torch.where(ind)[0].tolist()
                        if not ids:continue
                        idx=sorted(ids,key=lambda j:(float(dif[j,0]),j))[len(ids)//2]
                        selected.append({"change":name,"query":str(qp[int(qi[idx])]),"old":str(pp[int(pi[lead[b][idx]])]),"new":str(pp[int(pi[lead[a][idx]])]),
                                         "AP_before":float(m[b][idx,0]),"AP_after":float(m[a][idx,0])})
                    examples[scope]=selected
        inf={k:torch.cat(v) for k,v in stats.items()}
        save_tensor({"ids":qix,"labels":y,"metrics":m,"top1":t,"first_photo":lead,"information":inf},out.with_suffix(".pt"))
        atomic_json(out,{"freeze":freeze["sha256"],"code":revision(),"queries":len(y),"gallery":len(gl),"classes":len(cls.unique()),"original_evaluator":True,
                        "constraint_error":constraint_error,"KD_max_error":max_error,"values":{k:v.double().mean(0).tolist() for k,v in m.items()},
                        "comparisons":comparisons,"examples":examples,"information_total_given_global_given_full_given_scale":{k:v.mean(0).tolist() for k,v in inf.items()},
                        "per_query_sha256":checksum(out.with_suffix(".pt"))})
        print("saved",s,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    ap.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    ap.add_argument("--split",choices=("sketchy_1","sketchy_2","tuberlin"))
    a=ap.parse_args();run(a.workspace,a.dataset,a.split)
