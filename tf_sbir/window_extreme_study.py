"""Evaluate pairwise rival-conditioned extremal evidence with matched uncalibrated and independent-window controls."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import inspect
import os
from pathlib import Path

import torch

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json,checksum,read
from .crossdomain_benchmark import revision
from .crossdomain_inputs import CHECKPOINT
from .encoding import save_tensor
from .partition_rank_replay import PendingReplay
from .query_fragment_reference_inputs import dataset
from .query_partition_inputs import place
from .query_partition_study import locked,save_lock,rank_metrics,mean_tables
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd,paired_delta
from .window_extreme_calibration import SPECS
from .window_extreme_inputs import FAMILY


GAMMAS=(-.45,.1,.35,1.,2.5)
READOUT=("cell","regional","whole-cell")
METHODS=(*SPECS,"gain-minus-mixed","gain-minus-whole","gain-minus-pooled")
BASES=("base/direct","base/regional")


@dataclass(frozen=True)
class Row:
    domain:str
    level:int
    read:str
    method:str
    adversary:str
    gain:float

    def encode(self):return f"{self.domain}/{self.level}/{self.read}/{self.method}/{self.adversary}/{self.gain:g}"

    @classmethod
    def parse(cls,k):
        d,n,r,m,a,g=k.split("/")
        return cls(d,int(n),r,m,a,float(g))


def grid():
    out=set(BASES)
    for dom in ("photo","source"):
        for level in (2,3):
            for readout in READOUT:
                out.add(Row(dom,level,readout,"base","query",0.).encode())
                for m in METHODS:
                    for no in ("equal","query"):
                        for gain in GAMMAS:out.add(Row(dom,level,readout,m,no,gain).encode())
    return sorted(out)


def reduce(tables,all_settings):
    a,b=tables
    if a.keys()!=b.keys():raise ValueError("Different cells across Sketchy splits")
    def best(ks):return max(sorted(ks),key=lambda k:tuple((a[k][i]+b[k][i])/2 for i in (0,1)))
    names={k:Row.parse(k) for k in a if k not in BASES}
    chosen={};short=set(BASES)
    for read in READOUT:
        sub={k:r for k,r in names.items() if r.read==read}
        for mode in ("base",*METHODS):
            valid=[k for k,r in sub.items() if r.method==mode and r.gain>=0]
            if valid:chosen[read+"/"+mode]=best(valid);short.add(best(valid))
        chosen[read+"/best"]=best(sub)
        opposite=[k for k,r in sub.items() if r.method=="gain" and r.gain<0]
        if opposite:chosen[read+"/opposite"]=best(opposite);short.add(best(opposite))
        core=[k for k,r in sub.items() if r.method=="gain" and r.domain=="photo" and r.gain>0]
        if core:chosen[read+"/primary"]=best(core);short.add(best(core))
        for k in (chosen[read+"/primary"],chosen[read+"/gain"],chosen[read+"/best"]):short.add(k)
    for k in list(short):
        if k in BASES:continue
        src=names[k]
        if src.method not in ("gain","gain-minus-mixed","gain-minus-whole"):continue
        for m in ("gain","gain-unconditional","gain-centered","gain-mixed","gain-pooled","gain-raw","whole","mean_gain","maximum","single_pose","base"):
            tag=Row(src.domain,src.level,src.read,m,src.adversary if m!="base" else "query",src.gain if m!="base" else 0.).encode()
            if tag in names:short.add(tag)
        if all_settings:
            for dom in ("photo","source"):
                tag=Row(dom,src.level,src.read,src.method,src.adversary,src.gain).encode()
                if tag in names:short.add(tag)
    return chosen,sorted(short)


def verify_set(mapping,choices):
    leaders=list(dict.fromkeys(mapping[k] for k in ("cell/primary","cell/best","regional/primary","regional/best")))
    rows={*leaders,*BASES}
    for read in ("cell","regional"):
        for m in ("base","whole","gain-raw","gain-mixed","gain-pooled","mean_gain","opposite"):rows.add(mapping[read+"/"+m])
    for k in leaders:
        r=Row.parse(k)
        for m in ("base","gain","gain-centered","gain-mixed","gain-raw","gain-unconditional","whole"):
            v=Row(r.domain,r.level,r.read,m,r.adversary if m!="base" else "query",r.gain if m!="base" else 0.).encode()
            if v in choices:rows.add(v)
    return sorted(rows),leaders


def change(table,kind,who):
    if "-minus-" not in kind:return table[kind+"/"+who]
    first,last=kind.split("-minus-")
    other={"mixed":"gain-mixed","whole":"whole","pooled":"gain-pooled"}[last]
    return table[first+"/"+who]-table[other+"/"+who]


@torch.inference_mode()
def run(w,ds,stage):
    root=w/"results"/FAMILY;root.mkdir(parents=True,exist_ok=True)
    if stage=="lock":
        old=locked(root/"shortlist.json");p,_=reduce(mean_tables(root,"scan-"),False);rows,leaders=verify_set(p,set(old["config"]["scan"]))
        cfg={"rows":rows,"leaders":leaders,"picked":p,"primary":p["cell/primary"],"pilot":old["sha256"],"code":revision(),
             "scan_sha":{s:checksum(root/("scan-"+s+".json")) for s in ("sketchy_1","sketchy_2")}}
        print("freeze",save_lock(root/"freeze.json",cfg),len(rows),flush=True);return
    pilot=stage.startswith("pilot");official=stage in ("pilot-official","official")
    if ds=="tuberlin" and stage!="official":raise ValueError("TU is a transfer experiment")
    if stage=="pilot":
        rows=grid();sign=save_lock(root/"protocol.json",{"code":revision(),"model":CHECKPOINT,"KD":KD_FUNCTION_SHA,"rows":rows,
                  "selection":"equal mean mAP then precision on unseen Sketchy 36/class, original galleries","primary":"photographic null of the largest pose-stable within-image gain"})
    else:
        sel=locked(root/("freeze.json" if stage=="official" else "shortlist.json"))
        sign=sel["sha256"];rows=sel["config"][{"pilot-official":"audit","official":"rows","scan":"scan"}[stage]]
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Wrong KD reference")
    _,_,qs,_,gs,_,_=dataset(w,ds);prefix="" if stage=="official" else stage+"-"
    get=lambda f:torch.load(f,mmap=True,weights_only=True)
    for s,(_,y) in qs.items():
        dest=root/(prefix+s+".json")
        if dest.is_file():
            old=read(dest)
            if old["signature"]!=sign or checksum(dest.with_suffix(".pt"))!=old["per_query_sha256"]:raise ValueError("Invalid result receipt")
            continue
        cache=w/"cache"/FAMILY/s/("groups-pilot" if pilot else "groups-full")
        read(cache/"done.json")
        _,gl=gs[s];gt=gl.cuda();idx=get(cache/"photo-2.pt")["ids"];labels=y[idx]
        reg=get(w/"cache/gallery-stable-region-evidence"/(s+"-regional-reference.pt"))[idx]
        direct=get(place(w,s)/"lex.pt")[idx]
        rng=torch.randperm(len(gl),generator=torch.Generator().manual_seed(62511)).cuda()
        specs=defaultdict(list)
        for n in rows:
            if n not in BASES:
                r=Row.parse(n);specs[(r.domain,r.level)].append(r)
        val,top,heads=defaultdict(list),defaultdict(list),defaultdict(list)
        workers=min(7,len(os.sched_getaffinity(0)));maxerr=None
        with (ThreadPoolExecutor(max_workers=workers) if official else nullcontext()) as pool:
            pending=PendingReplay(PrefixReplay(kd,gl,s),pool,workers) if official else None
            def emit(k,matrix,tie,sl):
                v,rank,good=rank_metrics(matrix,tie,labels[sl].cuda(),gt,s)
                if pending:
                    pending.submit(rank,labels[sl],v,val[k]);heads[k].append(rank[:,:5].int().cpu())
                else:val[k].append(v.cpu())
                top[k].append(good.cpu())
            for i in range(0,len(idx),96):
                sl=slice(i,i+96)
                for k,x in zip(BASES,(direct,reg),strict=True):
                    if k in rows:
                        mat=x[sl].cuda();tie=rng[mat[:,rng].argsort(dim=1,descending=True,stable=True)];emit(k,mat,tie,sl)
            for (domain,level),recipes in specs.items():
                data=get(cache/f"{domain}-{level}.pt");table={k:v.cuda() for k,v in data["scores"].items()}
                membership=data["labels"].cuda();hold=torch.zeros_like(membership)
                for fold,ix in enumerate(data["parts"]):hold[ix.cuda()]=fold
                key=hold*table["whole"].size(2)+membership
                for i in range(0,len(idx),96):
                    sl=slice(i,i+96);a=direct[sl].cuda();b=reg[sl].cuda();g={k:v[sl] for k,v in table.items()}
                    ties={r:rng[z[:,rng].argsort(dim=1,descending=True,stable=True)] for r,z in (("cell",a),("regional",b))}
                    for recipe in recipes:
                        kind=recipe.read
                        base=g["whole" if kind=="whole-cell" else "multi"].flatten(1)[:,key] if kind!="regional" else b
                        z=base if recipe.method=="base" else base+recipe.gain*change(g,recipe.method,recipe.adversary).flatten(1)[:,key]
                        emit(recipe.encode(),z,ties["regional" if kind=="regional" else "cell"],sl)
                print(stage,s,domain,level,len(recipes),flush=True)
            if pending:maxerr=pending.finish()
        m={k:torch.cat(v) for k,v in val.items()};t={k:torch.cat(v) for k,v in top.items()}
        if m.keys()!=set(rows):raise ValueError("Incomplete calibration controls")
        save_tensor({"ids":idx,"labels":labels,"metrics":m,"top1":t,"first_five":{k:torch.cat(v) for k,v in heads.items()}},dest.with_suffix(".pt"))
        details={"signature":sign,"code":revision(),"queries":len(idx),"gallery":len(gl),"classes":len(labels.unique()),"original_evaluator":official,
               "KD_max_error":maxerr,"values":{k:v.double().mean(0).tolist() for k,v in m.items()},"per_query_sha256":checksum(dest.with_suffix(".pt")),
               "empirical_calibration":{f"{d}-{n}":read(cache/f"{d}-{n}.json") for d,n in specs}}
        if official:
            labs=labels.unique(return_inverse=True)[1];leaders=sel["config"]["leaders"]
            details["paired"]={k:{b:paired_delta(m[k],m[b],labs) for b in rows} for k in leaders}
            details["top1"]={k:{b:[int((t[k]&~t[b]).sum()),int((~t[k]&t[b]).sum())] for b in rows} for k in leaders}
        atomic_json(dest,details);print("saved",stage,s,flush=True)
    if stage=="pilot":
        p,scan=reduce(mean_tables(root,"pilot-"),True);ar,leads=verify_set(p,set(scan))
        print("shortlist",save_lock(root/"shortlist.json",{"picked":p,"scan":scan,"audit":ar,"leaders":leads,"prior":sign}),len(scan),len(ar),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy");p.add_argument("--stage",choices=("pilot","pilot-official","scan","lock","official"),required=True)
    a=p.parse_args();run(a.workspace,a.dataset,a.stage)
