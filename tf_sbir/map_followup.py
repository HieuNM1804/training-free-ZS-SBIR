"""Lock the new Sketchy selection, run held-out TU and verify matched AP changes."""

import argparse
import csv
from datetime import datetime,timezone
from pathlib import Path
import os
import subprocess
import sys
import time
import zipfile

import torch

from .concept_benchmark import atomic_json,checksum,read
from .conditional_semantic_index import SIZES
from .crossdomain_benchmark import digest,revision
from .data import SKETCHY_SPLITS,UNSEEN_CLASSES,collect_paths
from .map_study import score_key
from .map_transfer_audit import load_result
from .retrieval_evidence import paired_delta,tensor


METHODS={
    "neighborhood-calibrated-likelihood":"a8a73530a993",
    "winclip-concept-likelihood":"24b9cae9603c",
    "freedom-gallery-transfer":"5f874b5570ec",
}
OLD="cross-diffusion-map"
OLD_LOCK="90eee74f8730e3e76fe17ac93fef4b48648f91b04bcc8371e6c296cbcca7158d"


def signed(path):
    item=read(path)
    expected=item.pop("sha256")
    if digest(item)!=expected:
        raise ValueError(f"Incorrect signed record {path}")
    return item,expected


def lock_result(root,path):
    if path.is_file():
        return signed(path)[0]
    incumbent,old_sig=signed(root/"map-followup/incumbent-freeze.json")
    if old_sig!=OLD_LOCK or incumbent["primary"]!=OLD:
        raise ValueError("Wrong incumbent")
    entries={OLD:incumbent["families"][OLD]}
    entries[OLD]["historical"]=True
    for name,commit in METHODS.items():
        candidate=root/name
        lock,sig=signed(candidate/"freeze.json")
        if (not lock["revision"].startswith(commit) or lock["tu_rows_before_freeze"]!=0
                or any((candidate/"heldout").glob("tuberlin-*.json"))):
            raise ValueError(f"TU measured before the prospective lock for {name}")
        rows=[]
        for split in SKETCHY_SPLITS:
            file=candidate/"sketchy"/f"{split}-{lock['winner_id']}.json"
            data=read(file)
            h=checksum(file)
            if h!=lock["sources"][file.name] or data["kd_check"]["max_per_query_abs_AP_P"]!=[0.,0.]:
                raise ValueError("Source metrics changed or KD did not reproduce them")
            rows.append({k:data[k] for k in ("mAP","precision")}|{"split":split,"sha256":h})
        entries[name]={"scores":rows,"name":lock["winner_id"],"freeze":sig,
                       "historical":False,"tu_rows_prelock":0,"parameters":lock["winner"]}
    winner=max(entries,key=lambda n:score_key(entries[n]["scores"]))
    record={"revision":revision(),"utc":datetime.now(timezone.utc).isoformat(),"primary":winner,"families":entries,
            "selection":"equal mean official Sketchy mAP, tie break equal mean precision","selection_data":list(SKETCHY_SPLITS)}
    atomic_json(path,record|{"sha256":digest(record)})
    return record


def audit(w,joint):
    results=w/"results"
    out=results/"map-followup"
    table=[]
    comparisons={}
    reference={"native":("conditional-semantic-index","native"),"concepts-uniform":("shared-concept-likelihood","uniform"),
               "concepts-mle":("shared-concept-likelihood","named-likelihood")}
    reference.update({k:(k,v["name"]) for k,v in joint["families"].items()})
    for split in (*SKETCHY_SPLITS,"tuberlin"):
        sub="heldout" if split=="tuberlin" else "sketchy"
        dataset="tuberlin-data.json" if split=="tuberlin" else "ready.json"
        original=Path(read(w/dataset)["root"])
        cats=UNSEEN_CLASSES[split]
        ql=collect_paths(original,cats,"sketch")[1]
        pl=collect_paths(original,cats,"photo")[1]
        if (len(ql),len(pl),len(ql.unique()))!=SIZES[split]:
            raise ValueError("Unexpected split size")
        scores={}
        details={}
        controls={}
        for name,(family,cid) in reference.items():
            locked=read(results/family/"freeze.json") if family in METHODS else None
            row,point=load_result(results,family,split,cid,ql,pl,locked)
            scores[name]=row
            details[name]=point
            table.append({"split":split,"family":name,"control":"primary",**row})
            if locked is None:
                continue
            saved=tensor(results/family/sub/f"{split}-labels.pt")
            if not torch.equal(ql,saved["query"]) or not torch.equal(pl,saved["gallery"]):
                raise ValueError("The preserved sketch and photo label orders differ")
            controls[name]={}
            for control,parameters in locked["controls"].items():
                row,point=load_result(results,family,split,digest(parameters)[:16],ql,pl,locked)
                table.append({"split":split,"family":name,"control":control,**row})
                controls[name][control]=row|{"gain_of_primary":paired_delta(details[name],point,ql),"parameters":parameters}
        paired={}
        for name,values in details.items():
            if name==OLD:
                continue
            paired[name]={"versus_incumbent":paired_delta(values,details[OLD],ql),
                          "versus_uniform":paired_delta(values,details["concepts-uniform"],ql)}
        comparisons[split]={"scores":scores,"paired":paired,"controls":controls}
        if split=="tuberlin":
            distinct=set(cats)-set(UNSEEN_CLASSES["sketchy_1"])-set(UNSEEN_CLASSES["sketchy_2"])
            mask=torch.tensor([k in distinct for k in cats])[ql]
            comparisons[split]["name_disjoint_transfer"]={
                "classes":len(distinct),"queries":int(mask.sum()),
                "mAP":{name:float(x[mask,0].mean()) for name,x in details.items()}}
        print(split,{k:round(v["mAP"],6) for k,v in scores.items()},flush=True)
    stamp=signed(results/"map-study/joint-freeze.json")[1]
    atomic_json(out/"comparison.json",{"revision":revision(),"lock_sha256":stamp,"primary":joint["primary"],"splits":comparisons})
    with (out/"metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)
    files=[p for directory in (out,*(results/n for n in METHODS)) for p in directory.rglob("*") if p.is_file()]
    files.append(results/"map-study/joint-freeze.json")
    export=w/"map-followup-evidence.zip"
    with zipfile.ZipFile(export,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            archive.write(p,p.relative_to(w))
    with zipfile.ZipFile(export) as archive:
        if archive.testzip() is not None or len(archive.namelist())!=len(files):
            raise ValueError("Invalid evidence archive")
    atomic_json(w/"map-followup-archive-info.json",{"size":export.stat().st_size,"sha":checksum(export),"count":len(files)})


def run(w):
    root=w/"results"
    while not all((root/name/"freeze.json").is_file() for name in METHODS):
        ready=[name for name in METHODS if (root/name/"freeze.json").is_file()]
        print("Completed Sketchy studies",ready,flush=True)
        time.sleep(20)
    joint=lock_result(root,root/"map-study/joint-freeze.json")
    print("Locked Sketchy winner",joint["primary"],flush=True)
    for name,sha in METHODS.items():
        source=w/"worktrees"/("map-"+sha)
        subprocess.run([sys.executable,"-u","-m","tf_sbir.map_study","--family",name,"--stage","heldout"],cwd=source,
                       env=os.environ|{"OMP_NUM_THREADS":"1"},check=True)
    audit(w,joint)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
