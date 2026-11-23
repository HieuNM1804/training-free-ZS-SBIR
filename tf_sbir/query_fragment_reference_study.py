"""Compare and freeze source-only visual likelihood ratios and gallery-conditioned propagation, then replay all full ranks via KD-SBIR."""

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
from .crossdomain_benchmark import digest,revision,SPLIT_HASHES
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .map_diagnosis import map_values
from .query_fragment_likelihood import REFS,ORDERS,GalleryWalk,centered,mathematical_checks,pure_grid,read_scores,required_modes
from .query_fragment_reference_inputs import FAMILY,dataset,load
from .quickdraw_locked import KD_FUNCTION_SHA
from .recoverable_patch_study import locate
from .retrieval_diagnosis import stratified_sample
from .retrieval_evidence import load_kd,paired_delta


MIX=(.08,.22,.45,.8)


def mixkey(a,r):
    return f"mix{a:g}|{r}"


def prefix_score(x):
    return x.split("|")[-1]


def rank_key(values,names):
    return max(names,key=lambda k:(values[k][0],values[k][1]))


def choose_pilot(means):
    structured=[k for k in means if len(k.split("/"))==4]
    opt={"by_reference":{},"by_order":{}}
    for r in REFS:
        opt["by_reference"][r]=rank_key(means,[k for k in structured if k.split("/")[0]==r])
    for name in ("appearance","paired"):
        for op in ORDERS:
            opt["by_order"][name+"/"+op]=rank_key(means,[k for k in structured if k.startswith(name+"/") and k.endswith("/"+op)])
    opt["primary"]=opt["by_order"]["paired/ratio_after_walk"]
    opt["best_any"]=rank_key(means,means)
    opt["plain_graph"]=rank_key(means,[k for k in means if k.startswith("walk/")])
    r,t,g,o=opt["primary"].split("/")
    matched=[f"{kind}/{t}/{g}/{o}" for kind in REFS]
    matched += [f"{r}/{t}/{g}/{order}" for order in ORDERS]
    pool=sorted({"native","t4_source","t4_linear",opt["plain_graph"],opt["best_any"],*opt["by_reference"].values(),*opt["by_order"].values(),*matched})
    return opt,matched,pool


def select_final(means):
    pure=[k for k in means if not k.startswith("mix") and k!="regional"]
    fused=[k for k in means if k.startswith("mix")]
    def with_ref(ref,scope):
        return [k for k in scope if prefix_score(k).startswith(ref+"/")]
    def by_order(scope,ref,order):
        return [k for k in with_ref(ref,scope) if k.endswith("/"+order)]
    def selector(scope):
        return {"best":rank_key(means,scope),
                "primary":rank_key(means,by_order(scope,"paired","ratio_after_walk")),
                "appearance_after":rank_key(means,by_order(scope,"appearance","ratio_after_walk")),
                "appearance_direct":rank_key(means,by_order(scope,"appearance","direct")),
                "appearance_before":rank_key(means,by_order(scope,"appearance","walk_after_ratio")),
                "empty":rank_key(means,with_ref("empty_token",scope)+with_ref("mean_colour",scope)+with_ref("fixed_white",scope)+with_ref("fixed_neutral",scope)),
                "permuted_whole":rank_key(means,with_ref("scrambled_whole",scope)),
                "area_control":rank_key(means,with_ref("area_shuffle",scope)),
                "walk":rank_key(means,with_ref("walk",scope)),
                "global_mean":rank_key(means,with_ref("gallery_mean",scope))}
    p=selector(pure);b=selector(fused)
    used={"native","regional","t4_source","t4_linear",*p.values(),*b.values()}
    prim=p["primary"]
    _,t,g,o=prim.split("/")
    for ref in ("appearance","empty_token","area_shuffle","scrambled_whole"):
        k=f"{ref}/{t}/{g}/{o}"
        if k in means:used.add(k)
    return p,b,sorted(used)


def split_context(w,s,ds,inputs,bank,stage,need_regional):
    _,_,qs,_,gs,q,p=inputs
    qi,labels=qs[s];pi,gl=gs[s]
    if digest(UNSEEN_CLASSES[s])!=SPLIT_HASHES[s]:raise ValueError("Benchmark categories changed")
    indices=stratified_sample(labels,36,63241) if stage.startswith("pilot") else torch.arange(len(qi))
    pos=locate(bank["ids"],qi[indices])
    if float((1-(bank["native"][pos]*q[qi[indices]]).sum(1)).abs().max())>.002:raise ValueError("Query embedding mismatch")
    reg=None
    if need_regional:
        f=w/"cache/gallery-stable-region-evidence"/(s+"-regional-reference.pt")
        reg=torch.load(f,mmap=True,weights_only=True)
        if reg.shape!=(len(labels),len(gl)):raise ValueError("Full regional reference has different sizes")
        if len(indices)!=len(labels):reg=reg[indices]
    return {"ids":indices,"labels":labels[indices],"pos":pos,"gallery":p[pi].cuda(),"gl":gl,
            "q":q[qi[indices]].cuda(),"reg":reg}


def evaluate(s,bank,p,rows,kd,workers):
    pure={prefix_score(k) for k in rows if k!="regional"}
    integrations=defaultdict(list)
    for tag in rows:
        if tag.startswith("mix"):
            prefix,core=tag.split("|");integrations[core].append((tag,float(prefix[3:])))
    need_reg=p["reg"] is not None
    required=required_modes(pure)
    graph=GalleryWalk(p["gallery"])
    graph_check=graph.check(8*(p["q"][:4]@p["gallery"].T))
    gt=p["gl"].cuda()
    replay=Replay(kd,p["gl"],s) if kd else None
    scores,tops,errs,lead=defaultdict(list),defaultdict(list),{},defaultdict(list)
    # Chunk size does not influence a query's references, graph or normalization.
    with (ThreadPoolExecutor(max_workers=workers) if kd else nullcontext()) as pool:
        for off in range(0,len(p["ids"]),64):
            sl=slice(off,off+64)
            yy=p["labels"][sl]
            reg=p["reg"][sl].cuda() if need_reg else None
            spread=reg.std(1,keepdim=True,unbiased=False) if need_reg else None
            def record(name,matrix):
                if not torch.isfinite(matrix).all():raise ValueError("Non-finite reference score")
                v,order,r,_=map_values(matrix,yy.cuda(),gt,s)
                got=v.cpu()
                if replay:
                    exact=replay.score(order,yy,pool)
                    err=float((exact-got).abs().max())
                    if err>2e-6:raise ValueError("Different KD-SBIR evaluator or rank")
                    errs[name]=max(errs.get(name,0.),err)
                    got=exact
                    lead[name].append(order[:,:5].int().cpu())
                scores[name].append(got)
                tops[name].append(r[:,0].cpu())
            if "regional" in rows:record("regional",reg)
            use=p["pos"][sl]
            feats={k:bank["modes"][k][use].cuda() for k in required}
            for name,val in read_scores(p["gallery"],p["q"][sl],feats,pure,graph):
                if name in rows:record(name,val)
                if name in integrations:
                    term=spread*centered(val)
                    for tag,gain in integrations[name]:record(tag,reg+gain*term)
            if off%768==0:print(s,"official" if kd else "scan",off,len(p["ids"]),flush=True)
    return ({k:torch.cat(v) for k,v in scores.items()},{k:torch.cat(v) for k,v in tops.items()},
            {k:torch.cat(v) for k,v in lead.items()},errs,graph_check)


def save_lock(path,config):
    sha=digest(config)
    if path.is_file() and read(path)["sha256"]!=sha:raise ValueError("Conflicting frozen design")
    atomic_json(path,{"config":config,"sha256":sha})
    return sha


def get_lock(p):
    x=read(p)
    if digest(x["config"])!=x["sha256"]:raise ValueError("Bad config hash")
    return x


def tables(root,prefix):
    return [read(root/(prefix+s+".json"))["values"] for s in ("sketchy_1","sketchy_2")]


def average(both):
    if both[0].keys()!=both[1].keys():raise ValueError("Different choices on the two Sketchy splits")
    return {k:[sum(t[k][i] for t in both)/2 for i in (0,1)] for k in both[0]}


@torch.inference_mode()
def run(w,ds,stage):
    root=w/"results"/FAMILY
    root.mkdir(parents=True,exist_ok=True)
    if stage=="lock":
        shortlist=get_lock(root/"shortlist.json")
        means=average(tables(root,"scan-"))
        pure,mixed,rows=select_final(means)
        cfg={"rows":rows,"pure":pure,"regional_plus_visual":mixed,"input_modes":required_modes(rows),
             "shortlist":shortlist["sha256"],"scan_sha256":{s:checksum(root/("scan-"+s+".json")) for s in ("sketchy_1","sketchy_2")},
             "selection":"locked on complete unseen Sketchy Ext-1 and Ext-2 with equal mAP weights before TU-Berlin scores",
             "code":revision()}
        print("final lock",save_lock(root/"freeze.json",cfg),pure,mixed,flush=True)
        return
    if ds!="sketchy" and stage!="official":raise ValueError("No TU development stage")
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Wrong original evaluator")
    inputs=dataset(w,ds)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    bank=load(w,ds,"pilot" if stage.startswith("pilot") else "full")
    base="regional"
    if stage=="pilot":
        rows=pure_grid()
        prefix="pilot-";locked=None
        spec={"model":CHECKPOINT,"KD":KD_FUNCTION_SHA,"device":torch.cuda.get_device_name(),"code":revision(),"grid":rows,
              "sample_seed":63241,"queries_per_class":36,"selection":"exploratory two unseen splits, equal mean AP then equally weighted precision",
              "checks":mathematical_checks()}
        signature=digest(spec)
        if (root/"protocol.json").exists() and read(root/"protocol.json")["signature"]!=signature:raise ValueError("Changed pilot")
        atomic_json(root/"protocol.json",spec|{"signature":signature})
    elif stage in ("pilot-official","scan"):
        locked=get_lock(root/"shortlist.json");signature=locked["sha256"]
        rows=locked["config"]["pilot_official"] if stage=="pilot-official" else locked["config"]["full_sketchy_scan"]
        prefix="pilot-official-" if stage=="pilot-official" else "scan-"
    else:
        locked=get_lock(root/"freeze.json");signature=locked["sha256"];rows=locked["config"]["rows"];prefix=""
    official=stage in ("official","pilot-official")
    for s in inputs[2]:
        dest=root/(prefix+s+".json")
        if dest.is_file():
            old=read(dest)
            if old["signature"]!=signature or checksum(dest.with_suffix(".pt"))!=old["per_query_sha256"]:raise ValueError("Checkpoint differs")
            continue
        p=split_context(w,s,ds,inputs,bank,stage,any(k=="regional" or k.startswith("mix") for k in rows))
        workers=min(10,max(1,len(os.sched_getaffinity(0))-2))
        m,top,lead,err,gerr=evaluate(s,bank,p,rows,kd if official else None,workers)
        if set(m)!=set(rows):raise ValueError("Some readouts were not scored")
        save_tensor({"ids":p["ids"],"labels":p["labels"],"metrics":m,"top1":top,"first_five":lead},dest.with_suffix(".pt"))
        summary={"signature":signature,"queries":len(p["ids"]),"gallery":len(p["gl"]),"classes":len(p["labels"].unique()),
                 "values":{k:v.double().mean(0).tolist() for k,v in m.items()},"original_evaluator":official,
                 "effective_RGB_operator_max_error":gerr,"KD_max_error":max(err.values()) if err else None,
                 "per_query_sha256":checksum(dest.with_suffix(".pt"))}
        if official:
            labs=p["labels"].unique(return_inverse=True)[1]
            choices=([locked["config"]["pure"]["primary"],locked["config"]["pure"]["best"],locked["config"]["regional_plus_visual"]["best"]]
                     if stage=="official" else [locked["config"]["pilot_choice"]["primary"],locked["config"]["pilot_choice"]["best_any"]])
            pairs={k:{other:paired_delta(m[k],m[other],labs) for other in m} for k in dict.fromkeys(choices)}
            summary["paired"]=pairs
            summary["top1"]={k:{b:{"fixed":int((~top[b]&top[k]).sum()),"broken":int((top[b]&~top[k]).sum())}
                                for b in ("native","regional") if b in top} for k in dict.fromkeys(choices)}
        atomic_json(dest,summary)
        print(stage,s,len(p["ids"]),"saved",flush=True)
        del p
    if stage=="pilot":
        means=average(tables(root,prefix));opts,matched,short=choose_pilot(means)
        scan=sorted({"regional",*short,*(mixkey(a,r) for a in MIX for r in short)})
        primary=opts["primary"];app=opts["by_reference"]["appearance"]
        audit_rows=sorted({"native","t4_source","t4_linear",opts["plain_graph"],primary,opts["best_any"],app,
            opts["by_reference"]["empty_token"],opts["by_reference"]["mean_colour"],opts["by_reference"]["fixed_white"],
            opts["by_reference"]["fixed_neutral"],opts["by_reference"]["area_shuffle"],opts["by_reference"]["scrambled_whole"],
            opts["by_order"]["appearance/ratio_after_walk"],opts["by_order"]["appearance/walk_after_ratio"],
            *[x for x in matched if x.startswith(("appearance/","empty_token/"))]})
        cfg={"pilot_choice":opts,"mean":{k:means[k] for k in short},"matched":matched,"pilot_official":audit_rows,
             "full_sketchy_scan":scan,"input_modes":required_modes(scan),"pilot_signature":signature}
        print("shortlist",save_lock(root/"shortlist.json",cfg),primary,opts["best_any"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    p.add_argument("--stage",choices=("pilot","pilot-official","scan","lock","official"),required=True)
    a=p.parse_args();run(a.workspace,a.dataset,a.stage)
