"""Confirm competitive ablations on Sketchy and lock one decision before new TU measurements.

Run this script by absolute path with PYTHONPATH set to the measured method worktree.
"""

import argparse
from dataclasses import asdict
from datetime import datetime,timezone
from importlib import import_module
from pathlib import Path

import torch
from torch.nn import functional as F

from tf_sbir.concept_benchmark import atomic_json, checksum, read
from tf_sbir.crossdomain_benchmark import SPLIT_HASHES, digest, revision
from tf_sbir.crossdomain_inputs import CHECKPOINT
from tf_sbir.data import SKETCHY_SPLITS, UNSEEN_CLASSES
from tf_sbir.encoding import save_tensor
from tf_sbir.foundation import load_words
from tf_sbir.residual_benchmark import INCUMBENT, SIZES, get_views, load_lock, ordered
from tf_sbir.retrieval_diagnosis import global_inputs, stratified_sample
from tf_sbir.reverse_benchmark import pilot_score, record_config
from tf_sbir.semantic import read_words


def eligible(row):
    p=row["parameters"]
    return p.get("pairing") not in ("shuffled","random_groups") and p.get("location")!="opposite"


def available(folder,split):
    scores={}
    for sub in ("sketchy","confirm/sketchy"):
        for file in (folder/sub).glob(split+"-*.json"):
            row=read(file)
            if eligible(row):
                scores[row["method"]]=row
    return list(scores.values())


def full_ranking(folder):
    listings=[available(folder,s) for s in SKETCHY_SPLITS]
    names=set.intersection(*({r["method"] for r in block} for block in listings))
    return ordered([[r for r in block if r["method"] in names] for block in listings])


def where(folder,split,c,held=False):
    for path in ([folder/"confirm/heldout"] if held else [folder/"sketchy",folder/"confirm/sketchy"]):
        if (path/f"{split}-{c.name}.json").exists():
            return path
    return None


def new_lock(folder,method,selected,rev):
    base=load_lock(folder,rev)
    if any((folder/"confirm/heldout").glob("*.json")):
        raise ValueError("TU was already scored before the new freeze")
    controls=method.controls(selected)
    checks={}
    for split in SKETCHY_SPLITS:
        for c in controls.values():
            loc=where(folder,split,c)
            f=loc/f"{split}-{c.name}.json"
            r=read(f)
            if r["revision"]!=rev or r["split_sha256"]!=SPLIT_HASHES[split]:
                raise ValueError("Invalid official row")
            checks[str(f.relative_to(folder))]=checksum(f)
    lock={"utc":datetime.now(timezone.utc).isoformat(),"revision":rev,"initial_lock":base["freeze_sha256"],
          "method":selected.name,"winner":asdict(selected),"controls":{n:asdict(v) for n,v in controls.items()},
          "sketchy_rows":checks,"tu_score_count":0,"rule":"equal mean Sketchy precision, then mean mAP"}
    atomic_json(folder/"confirm/freeze.json",lock|{"freeze_sha256":digest(lock)})


def join(root,studies):
    path=root/"residual-study/final-selection.json"
    if path.exists():
        record=read(path)
        signature=record.pop("sha256")
        if signature!=digest(record) or set(record["families"])!=set(studies)|{INCUMBENT}:
            raise ValueError("Conflicting joint selection")
        return record["primary"]
    original=read(root/"inquiry-joint-freeze.json")
    original_hash=original.pop("sha256")
    if digest(original)!=original_hash or original["primary"]!=INCUMBENT:
        raise ValueError("Invalid reference")
    entries={INCUMBENT:original["families"][INCUMBENT]|{"historical":True}}
    for family in studies:
        folder=root/family
        lock=load_lock(folder/"confirm")
        if any((folder/"heldout").glob("*.json")) or any((folder/"confirm/heldout").glob("*.json")):
            raise ValueError("New TU row before joint selection")
        scores=[]
        for s in SKETCHY_SPLITS:
            f=next(f for f in (folder/"sketchy"/f"{s}-{lock['method']}.json",folder/"confirm/sketchy"/f"{s}-{lock['method']}.json") if f.is_file())
            row=read(f)
            if checksum(f)!=lock["sketchy_rows"][str(f.relative_to(folder))]:
                raise ValueError("Frozen Sketchy score changed")
            scores.append({k:row[k] for k in ("split","method","mAP","precision")})
        entries[family]=dict(revision=lock["revision"],freeze_sha256=lock["freeze_sha256"],scores=scores,
                             historical=False,tu_rows_prelock=0)
    def key(family):
        return tuple(sum(r[m] for r in entries[family]["scores"])/2 for m in ("precision","mAP"))
    primary=max(entries,key=key)
    decision=dict(utc=datetime.now(timezone.utc).isoformat(),families=entries,primary=primary,
                  rule="equal mean Sketchy precision, then equal mean mAP",selection_splits=list(SKETCHY_SPLITS))
    atomic_json(path,decision|{"sha256":digest(decision)})
    return primary


def inputs(w,method,dataset):
    args,qp,qs,pp,gs,qraw,praw=global_inputs(w,dataset)
    words=read_words(args.vocabulary)
    if len(words)!=20931:
        raise ValueError("Wrong public vocabulary")
    source,target=(x.cuda() for x in load_words(args,words))
    get=getattr(method,"get_views",get_views)
    qv=get(args,qp,"sketch") if method.USE_VIEWS else None
    pv=get(args,pp,"photo") if method.USE_VIEWS else None
    for split,(qi,ql) in qs.items():
        pi,pl=gs[split]
        if (len(qi),len(pi),len(ql.unique()))!=SIZES[split] or len(pl.unique())!=SIZES[split][2] or digest(UNSEEN_CLASSES[split])!=SPLIT_HASHES[split]:
            raise ValueError("Invalid class set or dataset size")
        q,p=qraw[qi].cuda(),praw[pi].cuda()
        if qv is not None:
            q=torch.cat((q[:,None],F.normalize(qv[qi].cuda().float(),dim=-1)),1)
            p=torch.cat((p[:,None],F.normalize(pv[pi].cuda().float(),dim=-1)),1)
        yield args,split,q,ql,p,pl,method.Index(p,source,target)


@torch.inference_mode()
def run(w,family,stage):
    folder=w/"results"/family
    method=import_module("tf_sbir."+family.replace("-","_"))
    rev=revision()
    original=load_lock(folder,rev)
    base=method.Config(**full_ranking(folder)[0])
    lock={}
    if stage=="pilot":
        cfgs=[method.Config(**r) for r in full_ranking(folder)[:6]]+method.refine(base)
    elif stage=="full":
        rows=[read(folder/f"confirm/pilot-{s}.json")["scores"] for s in SKETCHY_SPLITS]
        cfgs=[method.Config(**p) for p in ordered(rows)[:5]]
    elif stage=="controls":
        cfgs=list(method.controls(base).values())
    elif stage=="freeze":
        # Controls might include an even better deployable setting. A further
        # controls pass is run by the supervisor if the selected name changes.
        if not all(where(folder,s,c) for s in SKETCHY_SPLITS for c in method.controls(base).values()):
            raise ValueError("New winner requires its own matched controls")
        if not (folder/"confirm/freeze.json").exists():
            new_lock(folder,method,base,rev)
        return
    else:
        lock=load_lock(folder/"confirm",rev)
        j=read(w/"results/residual-study/final-selection.json")
        sig=j.pop("sha256")
        if digest(j)!=sig or j["families"][family]["freeze_sha256"]!=lock["freeze_sha256"]:
            raise ValueError("Invalid heldout decision")
        cfgs=[method.Config(**v) for v in lock["controls"].values()]
    held=stage=="heldout"
    for args,s,q,ql,p,pl,index in inputs(w,method,"tuberlin" if held else "sketchy"):
        meta=dict(revision=rev,backbone="OpenAI CLIP ViT-B/32",checkpoint_sha256=CHECKPOINT,
                  gpu=torch.cuda.get_device_name(),split_sha256=SPLIT_HASHES[s],classes=SIZES[s][2],
                  study=family,exploratory_unseen=not held,freeze_sha256=lock.get("freeze_sha256"))
        if stage=="pilot":
            ids=stratified_sample(ql,96,92113)
            rows=[]
            for c in dict.fromkeys(cfgs):
                values=pilot_score(index,q[ids],c,ql[ids].cuda(),pl.cuda(),200 if s=="sketchy_2" else 100)
                rows.append(dict(method=c.name,parameters=asdict(c),**values))
            atomic_json(folder/f"confirm/pilot-{s}.json",meta|{"scores":rows,"queries":len(ids),"seed":92113,
                        "initial_lock":original["freeze_sha256"]})
        else:
            out=folder/("confirm/heldout" if held else "confirm/sketchy")
            save_tensor({"query":ql,"gallery":pl},out/f"{s}-labels.pt")
            args.metric_workers=5 if s!="sketchy_2" else 3
            for c in dict.fromkeys(cfgs):
                if not where(folder,s,c,held):
                    record_config(args,out,s,c,index,q,ql,pl,meta|{"external_vocabulary":method.vocab_size(c)})
        print(s,stage,len(cfgs),flush=True)
        del index,p,q


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    ap.add_argument("--study",nargs="+",required=True)
    ap.add_argument("--stage",choices=("pilot","full","controls","freeze","joint","heldout"),required=True)
    a=ap.parse_args()
    if a.stage=="joint":
        print(join(a.workspace/"results",a.study))
    else:
        run(a.workspace,a.study[0],a.stage)
