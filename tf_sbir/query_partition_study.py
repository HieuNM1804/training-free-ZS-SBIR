"""Select gallery cut and matched controls on Sketchy, then evaluate the signed freeze with KD-SBIR."""
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
from .crossdomain_benchmark import digest,revision
from .crossdomain_inputs import CHECKPOINT
from .encoding import save_tensor
from .query_partition_inputs import context,query_scores,source_inputs,FAMILY,SOURCES
from .query_partition_model import Row,grid,predictions,FIXED,PAIRINGS
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd,paired_delta


def rank_metrics(score,source_order,labels,gl,s):
    local=score.gather(1,source_order).argsort(dim=1,descending=True,stable=True)
    order=source_order.gather(1,local)
    good=labels[:,None]==gl[order]
    upto=good[:,:200] if s=="sketchy_2" else good
    prec=upto.cumsum(1)/torch.arange(1,upto.size(1)+1,device=gl.device)
    ap=(upto*prec/upto.sum(1,keepdim=True).clamp_min(1)).sum(1)
    k=200 if s=="sketchy_2" else 100
    return torch.stack((ap,good[:,:k].float().mean(1)),1),order,good[:,0]


def raw_name(k):return "base/"+k


def reduce_choices(tables,detail):
    # Every selection uses an equal weight on the two complete split metrics available at this stage.
    a,b=tables
    if set(a)!=set(b):raise ValueError("Candidate differences across splits")
    def best(keys):
        return max(sorted(keys),key=lambda k:tuple((a[k][i]+b[k][i])/2 for i in (0,1)))
    rows={k:Row.parse(k) for k in a if not k.startswith("base/")}
    picked={};pool=set()
    for base,other in PAIRINGS:
        for kind in ("pool","shift"):
            group={k:v for k,v in rows.items() if (v.base,v.other,v.kind)==(base,other,kind)}
            for scope in ("other","read","fixed","permgate","permtree","prior","ungrouped"):
                opts=[k for k,v in group.items() if (v.gate in FIXED if scope=="fixed" else v.gate==scope)]
                if not opts:continue
                name=f"{base}/{kind}/{other}/{scope}"
                pick=best(opts);picked[name]=pick;pool.add(pick)
                if detail and scope in ("other","read","fixed"):
                    for gain in sorted({rows[k].gain for k in opts}):
                        pool.add(best([k for k in opts if rows[k].gain==gain]))
    for base in ("lex","regional","fragments","wordwalk"):
        opts=[k for k,v in rows.items() if v.base==base and v.gate=="other"]
        if opts:picked[base+"/cross"]=best(opts)
        opts=[k for k,v in rows.items() if v.base==base]
        if opts:picked[base+"/best"]=best(opts)
    for name in ("lex/pool/native/other","lex/shift/native/other","regional/cross","fragments/cross"):
        win=rows[picked[name]]
        for control in ("read","permgate","permtree","prior",*FIXED):
            v=win.variance if control not in (*FIXED,"prior") else 2.
            k=Row(win.base,win.kind,win.other,control,v,win.scale,win.gain).label()
            if k in rows:pool.add(k)
    return picked,sorted({*pool,*[raw_name(k) for k in SOURCES]})


def locked(file):
    d=read(file)
    if digest(d["config"])!=d["sha256"]:raise ValueError("Changed lock")
    return d


def save_lock(file,cfg):
    sha=digest(cfg)
    if file.exists() and locked(file)["sha256"]!=sha:raise ValueError("Conflicting freeze")
    atomic_json(file,{"config":cfg,"sha256":sha});return sha


def mean_tables(root,prefix):
    return [read(root/(prefix+s+".json"))["values"] for s in ("sketchy_1","sketchy_2")]


@torch.inference_mode()
def run(w,ds,stage):
    root=w/"results"/FAMILY;root.mkdir(parents=True,exist_ok=True)
    torch.set_float32_matmul_precision("highest")
    if stage=="lock":
        shortlist=locked(root/"shortlist.json")
        picked,_=reduce_choices(mean_tables(root,"scan-"),False)
        wants=["lex/pool/native/other","lex/shift/native/other","lex/cross","regional/cross","fragments/cross","wordwalk/cross"]
        wanted={raw_name(k) for k in SOURCES}
        for b in ("lex","fragments","regional","wordwalk"):
            wanted.add(picked[b+"/best"])
        for name in wants:
            winner=Row.parse(picked[name]);stem=f"{winner.base}/{winner.kind}/{winner.other}/"
            wanted.add(winner.label())
            for kind in ("read","fixed","permgate","permtree","prior","ungrouped"):
                if stem+kind in picked:wanted.add(picked[stem+kind])
            for head in ("read","permgate","permtree"):
                matched=Row(winner.base,winner.kind,winner.other,head,winner.variance,winner.scale,winner.gain).label()
                if matched in shortlist["config"]["scan"]:wanted.add(matched)
        cfg={"rows":sorted(wanted),"primary":picked["lex/pool/native/other"],"shift":picked["lex/shift/native/other"],
             "picked":picked,"pilot":shortlist["sha256"],"code":revision(),"scans":{s:checksum(root/("scan-"+s+".json")) for s in ("sketchy_1","sketchy_2")},
             "selection":"two full unseen Sketchy splits weighted equally by mAP, then precision, before held-out TU scores"}
        print("freeze",save_lock(root/"freeze.json",cfg),len(wanted),flush=True);return
    if ds=="tuberlin" and stage!="official":raise ValueError("TU reserved for official transfer")
    stage_sample="pilot" if stage.startswith("pilot") else "full"
    official=stage in ("official","pilot-official")
    if stage=="pilot":
        rows=[*[raw_name(k) for k in SOURCES],*grid()]
        spec={"code":revision(),"checkpoint":CHECKPOINT,"evaluator":KD_FUNCTION_SHA,"rows":rows,"folds":3,"seed":14891,
              "selection":"36 unseen sketches per class, full original galleries; equal split mAP then precision"}
        sig=digest(spec)
        if (root/"protocol.json").exists() and read(root/"protocol.json")["sha256"]!=sig:raise ValueError("Changed pilot")
        atomic_json(root/"protocol.json",{"config":spec,"sha256":sig,"device":torch.cuda.get_device_name()})
        pref="pilot-"
    else:
        f=locked(root/("freeze.json" if stage=="official" else "shortlist.json"))
        sig=f["sha256"];key={"scan":"scan","pilot-official":"audit","official":"rows"}[stage]
        rows=f["config"][key];pref="" if stage=="official" else stage+"-"
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("KD routine changed")
    source=source_inputs(w,ds)
    for s in source[1]:
        dest=root/(pref+s+".json")
        if dest.exists():
            prev=read(dest)
            if prev["signature"]!=sig or checksum(dest.with_suffix(".pt"))!=prev["per_query_sha256"]:raise ValueError("Corrupt checkpoint")
            continue
        c=context(w,s,source,stage_sample);c["shuffled"]=c["tree"].shuffle()
        gl=c["gl"].cuda();replay=Replay(kd,c["gl"],s) if official else None
        m,top,heads=defaultdict(list),defaultdict(list),defaultdict(list)
        err=0.;modeled=[k for k in rows if not k.startswith("base/")]
        # Explicit raw-score secondary order, then seeded photo order for residual ties. Never break cluster ties by category.
        perm=torch.randperm(len(gl),generator=torch.Generator().manual_seed(23941)).to(gl.device)
        workers=min(8,max(1,len(os.sched_getaffinity(0))-1))
        with (ThreadPoolExecutor(max_workers=workers) if official else nullcontext()) as pool:
            for off in range(0,len(c["ix"]),64):
                sl=slice(off,off+64);raw=query_scores(c,sl);yy=c["y"][sl];y=yy.cuda()
                orders={k:perm[v[:,perm].argsort(dim=1,descending=True,stable=True)] for k,v in raw.items()}
                def record(k,mat,order):
                    nonlocal err
                    val,ids,good=rank_metrics(mat,order,y,gl,s)
                    if official:
                        exact=replay.score(ids,yy,pool);err=max(err,float((val.cpu()-exact).abs().max()))
                        if err>2e-6:raise ValueError("Original KD evaluator disagrees")
                        val=exact;heads[k].append(ids[:,:5].int().cpu())
                    m[k].append(val.cpu());top[k].append(good.cpu())
                for k,orig in raw.items():
                    if raw_name(k) in rows:record(raw_name(k),orig,orders[k])
                for name,score,_ in predictions(c,raw,modeled):
                    record(name,score,orders[name.split("/")[0]])
                if off%768==0:print(stage,s,off,len(c["ix"]),len(rows),flush=True)
        m={k:torch.cat(v) for k,v in m.items()};top={k:torch.cat(v) for k,v in top.items()}
        if set(m)!=set(rows):raise ValueError("Some rows were not evaluated")
        save_tensor({"ids":c["ix"],"labels":c["y"],"metrics":m,"top1":top,"first_five":{k:torch.cat(v) for k,v in heads.items()}},dest.with_suffix(".pt"))
        meta={"signature":sig,"code":revision(),"queries":len(c["ix"]),"gallery":len(gl),"classes":len(c["y"].unique()),
              "levels":c["tree"].sizes,"values":{k:v.double().mean(0).tolist() for k,v in m.items()},
              "original_evaluator":official,"KD_max_error":err if official else None,"per_query_sha256":checksum(dest.with_suffix(".pt"))}
        if official:
            chosen=f["config"]
            checkpoints=([chosen["primary"],chosen["shift"],chosen["picked"]["regional/cross"],chosen["picked"]["lex/cross"],chosen["picked"]["fragments/cross"]]
                         if stage=="official" else chosen["audit_primary"])
            classes=c["y"].unique(return_inverse=True)[1]
            meta["paired"]={k:{j:paired_delta(m[k],m[j],classes) for j in rows} for k in dict.fromkeys(checkpoints)}
            meta["top1"]={k:{j:[int((top[k]&~top[j]).sum()),int((~top[k]&top[j]).sum())] for j in rows} for k in dict.fromkeys(checkpoints)}
        atomic_json(dest,meta);print("saved",stage,s,flush=True)
    if stage=="pilot":
        picked,scan=reduce_choices(mean_tables(root,pref),True)
        primary=[picked[k] for k in ("lex/pool/native/other","lex/shift/native/other","lex/cross","regional/cross","fragments/cross")]
        audit={*primary,*[raw_name(k) for k in SOURCES]}
        for b in ("lex","regional","fragments"):
            audit.add(picked[b+"/best"])
        for r in (Row.parse(k) for k in primary):
            for h in ("fixed","read","permgate","prior"):
                k=f"{r.base}/{r.kind}/{r.other}/{h}"
                if k in picked:audit.add(picked[k])
        cfg={"picked":picked,"audit_primary":primary,"audit":sorted(audit),"scan":scan,"protocol":sig}
        print("shortlist",save_lock(root/"shortlist.json",cfg),len(scan),len(audit),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    p.add_argument("--stage",choices=("pilot","pilot-official","scan","lock","official"),required=True)
    a=p.parse_args();run(a.workspace,a.dataset,a.stage)
