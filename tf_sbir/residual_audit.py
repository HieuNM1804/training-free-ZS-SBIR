"""Inspect locked trials, paired class changes and exact full-gallery metric equivalence."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from importlib import import_module
from pathlib import Path

import torch
import tf_sbir

from .benchmark import merge_inputs
from .concept_benchmark import atomic_json, checksum, read
from .crossdomain_benchmark import digest
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .encoding import global_cache_name
from .residual_benchmark import INCUMBENT, load_lock
from .residual_confirm import inputs, where
from .retrieval_evidence import check_record, load_kd, paired_delta, tensor
from .verify_locked_ranks import replay


def write_csv(path,header,lines):
    with path.open("w",newline="",encoding="utf8") as f:
        writer=csv.writer(f,lineterminator="\n")
        writer.writerow(header)
        writer.writerows(lines)


def confusers(old,new,ql,pl,names):
    before,after=pl[old[:,0]],pl[new[:,0]]
    errors=before!=ql
    counts=Counter(zip(ql[errors].tolist(),before[errors].tolist()))
    transitions=[]
    for (source,wrong),n in counts.most_common(10):
        mask=(ql==source)&(before==wrong)
        transitions.append({"class":names[source],"old_wrong":names[wrong],"old_count":n,
                            "correct_now":int((after[mask]==source).sum()),
                            "still_same":int((after[mask]==wrong).sum())})
    return {"old_wrong":int(errors.sum()),"new_wrong":int((after!=ql).sum()),
            "rescued":int(((after==ql)&errors).sum()),"regressed":int(((after!=ql)&~errors).sum()),
            "confusion_pairs":transitions}


@torch.inference_mode()
def run(w):
    root=w/"results"; dest=root/"residual-study"
    if not (dest/"heldout.done").is_file():
        raise ValueError("Trials incomplete")
    joint=read(dest/"final-selection.json")
    signature=joint.pop("sha256")
    if digest(joint)!=signature:
        raise ValueError("Modified selection")
    families=joint["families"]; primary=joint["primary"]
    methods,locks={},{}
    for name,data in families.items():
        if name==INCUMBENT:
            continue
        checkout=w/"worktrees"/("residual-"+data["revision"][:12])/"tf_sbir"
        if not checkout.is_dir():
            raise FileNotFoundError(checkout)
        tf_sbir.__path__.insert(0,str(checkout))
        methods[name]=import_module("tf_sbir."+name.replace("-","_"))
        lock=load_lock(root/name/"confirm",data["revision"])
        if lock["freeze_sha256"]!=data["freeze_sha256"] or data["tu_rows_prelock"]!=0:
            raise ValueError("Invalid freeze")
        for file,sha in lock["sketchy_rows"].items():
            if checksum(root/name/file)!=sha:
                raise ValueError("Sketchy source record changed")
        rdir=root/name/"confirm/heldout"
        held=[read(file) for file in rdir.glob("tuberlin-*.json")]
        assert {digest(r["parameters"]) for r in held}=={digest(c) for c in lock["controls"].values()}
        if any(r["freeze_sha256"]!=data["freeze_sha256"] or r["revision"]!=data["revision"] or r["exploratory_unseen"] for r in held):
            raise ValueError("Heldout setting differs from lock")
        locks[name]=lock
    reference=read(root/"inquiry-report/verified-rankings.json")
    kd,kd_hash=load_kd(w)
    if kd_hash!=reference["kd_source_sha256"]:
        raise ValueError("Reference KD function changed")
    audit={"primary":primary,"joint_freeze_sha256":signature,"kd_source_sha256":kd_hash,"splits":{}}
    config_rows,classes=[],[]
    for s in (*SKETCHY_SPLITS,"tuberlin"):
        held=s=="tuberlin"
        origin=root/INCUMBENT/("heldout" if held else "sketchy")
        order=tensor(origin/f"{s}-labels.pt")
        ql,pl=order["query"],order["gallery"]
        names=UNSEEN_CLASSES[s]
        ref_name=families[INCUMBENT]["scores"][0]["method"]
        old=check_record(read(origin/f"{s}-{ref_name}.json"),origin,ql,pl,s)
        old_top=tensor(origin/"top10"/f"{s}-{ref_name}.pt").long()
        native=check_record(read(origin/f"{s}-native-clip.json"),origin,ql,pl,s)
        details={INCUMBENT:old,"native":native}
        top={INCUMBENT:old_top}
        tables={}
        for fam,method in methods.items():
            folder=root/fam
            lock=locks[fam]
            c=method.Config(**lock["winner"])
            entries={}
            for alias,param in lock["controls"].items():
                choice=method.Config(**param)
                path=where(folder,s,choice,held)
                labels=tensor(path/f"{s}-labels.pt")
                if not torch.equal(ql,labels["query"]) or not torch.equal(pl,labels["gallery"]):
                    raise ValueError("Dataset order mismatch")
                r=read(path/f"{s}-{choice.name}.json")
                if r["revision"]!=lock["revision"] or r["parameters"]!=param:
                    raise ValueError("Different method scored")
                key=fam+"/"+alias
                details[key]=check_record(r,path,ql,pl,s)
                entries[alias]={"method":choice.name,"precision":r["precision"],"mAP":r["mAP"]}
                if choice==c:
                    top[fam]=tensor(path/"top10"/f"{s}-{c.name}.pt").long()
                    details[fam]=details[key]
            if held:
                directories=[folder/"confirm/heldout"]
            else:
                directories=[folder/"sketchy",folder/"confirm/sketchy"]
            for path in directories:
                for f in sorted(path.glob(s+"-*.json")):
                    r=read(f)
                    check_record(r,path,ql,pl,s)
                    config_rows.append([s,fam,r["method"],r["precision"],r["mAP"],r["external_vocabulary"],r["revision"],r["freeze_sha256"]])
            tables[fam]={"method":c.name,"controls":entries,"vs_previous":paired_delta(details[fam],old,ql),
                         "top1":confusers(old_top,top[fam],ql,pl,names)}
        selected=details[primary]
        for k,n in enumerate(names):
            mask=ql==k
            classes.append([s,n,int(mask.sum()),*old[mask].mean(0).tolist(),*selected[mask].mean(0).tolist()])
        pairs={k:paired_delta(selected,v,ql) for k,v in details.items() if "/" not in k or k.startswith(primary+"/")}
        audit["splits"][s]={"queries":len(ql),"gallery":len(pl),"classes":len(names),"selection_AP_P":selected.mean(0).tolist(),
            "incumbent_AP_P":old.mean(0).tolist(),"native_AP_P":native.mean(0).tolist(),"paired":pairs,"methods":tables}
    if primary==INCUMBENT:
        audit["incumbent_original_replay"]=reference
    else:
        method=methods[primary]
        conf=method.Config(**locks[primary]["winner"])
        for dataset in ("sketchy","tuberlin"):
            for args,s,q,ql,p,pl,index in inputs(w,method,dataset):
                path=where(root/primary,s,conf,s=="tuberlin")
                metrics=tensor(path/"per-query"/f"{s}-{conf.name}.pt")
                with ThreadPoolExecutor(max_workers=6) as pool:
                    checks,_=replay(kd,pool,index.scores(q,conf),ql,pl,metrics,s)
                if max(checks["kd_minus_pipeline_max_abs_same_order_AP_P"])>1e-7:
                    raise ValueError("Metric disagreement at fixed ranks")
                sample=q[:7]
                own=next(index.scores(sample[:1],conf))[0]
                together=next(index.scores(sample,conf))[0]
                checks["single_vs_batch_score_max"]=float((own-together).abs().max())
                audit["splits"][s]["kd_replay"]=checks
                print(s,checks,flush=True)
    spectral="spectral-object-retrieval"
    if spectral in methods:
        method=methods[spectral]
        lock=locks[spectral]
        for dataset in ("sketchy","tuberlin"):
            from .crossdomain_inputs import runtime
            args=runtime(w,dataset)
            subsections=SKETCHY_SPLITS if dataset=="sketchy" else ("tuberlin",)
            paths,queries=merge_inputs(args.root,subsections,"sketch")
            file=args.cache/global_cache_name(args.root,paths,"sketch","spectral-clip-box-v1","fp16",args.clip_checkpoint)
            boxes=torch.load(file,weights_only=True,map_location="cpu",mmap=True)["boxes"]
            c=method.Config(**lock["winner"])
            for s,(qi,ql) in queries.items():
                box=boxes[qi,method.MODES.index(c.mode)].float()
                area=(box[:,2:]-box[:,:2]).prod(1)
                rows=[]
                for ids in area.argsort(stable=True).tensor_split(4):
                    row={"count":len(ids),"mean_box_area_fraction":float(area[ids].mean())}
                    for name in ("primary","both_full_frames","same_size_center","same_size_opposite"):
                        setting=method.Config(**lock["controls"][name])
                        folder=where(root/spectral,s,setting,s=="tuberlin")
                        values=tensor(folder/"per-query"/f"{s}-{setting.name}.pt")
                        row[name+"_AP_P"]=values[ids].mean(0).tolist()
                    rows.append(row)
                audit["splits"][s]["proposal_area_quartiles"]=rows
    write_csv(dest/"all-configurations.csv",("split","family","configuration","precision","mAP","public_words","revision","freeze_sha256"),config_rows)
    write_csv(dest/"by-class.csv",("split","class","queries","previous_AP","previous_P","selected_AP","selected_P"),classes)
    atomic_json(dest/"audit.json",audit)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
