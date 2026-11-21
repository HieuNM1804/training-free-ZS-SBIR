"""Refine the dense assignment bandwidth on Sketchy, on the original assignment branch."""

import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
from pathlib import Path
import time

from .concept_benchmark import atomic_json,checksum,read
from .crossdomain_benchmark import digest
from .data import SKETCHY_SPLITS
from .image_retrieval_round import verified
from .map_study import config_id,run,score_key
from .retrieval_evidence import paired_delta,tensor
from .retrieval_soft_assignment import Config,Index,cache_key,DECISION as INITIAL

FAMILY="retrieval-assignment-bandwidth"
DECISION=FAMILY+"/decision.json"
FULL_FINALISTS=2


def canonical():
    return Config(1.,1.,0,"mass")


def candidates():
    return [Config(q,p,0,"mass") for q in (.625,.75,.875,1.,1.125,1.25)
            for p in (.75,.875,1.,1.125,1.25,1.5)]


def controls(c):
    return {"primary":c,"checkpoint":canonical(),"both_query_width":replace(c,photo_scale=c.query_scale),
            "both_photo_width":replace(c,query_scale=c.photo_scale)}


def study(w):
    root=w/"results";folder=root/FAMILY
    run(w,FAMILY,"pilot")
    pilots=[read(folder/f"pilot-{s}.json")["settings"] for s in SKETCHY_SPLITS]
    best=max(candidates(),key=lambda c:score_key([r[config_id(c)] for r in pilots]))
    selected=score_key([r[config_id(best)] for r in pilots])
    baseline=score_key([r[config_id(canonical())] for r in pilots])
    qualified=selected>baseline
    screen={"decision_data":list(SKETCHY_SPLITS),"qualified":qualified,"best":asdict(best),
            "best_mAP_P":selected,"checkpoint_mAP_P":baseline,"new_TU_at_screen":0}
    atomic_json(folder/"screen.json",screen)
    if qualified:
        for step in ("sketchy","controls","freeze"):
            run(w,FAMILY,step)
    atomic_json(folder/"study-complete.json",screen)
    print(screen,flush=True)


def transfer(w):
    root=w/"results";folder=root/FAMILY
    while not (folder/"study-complete.json").is_file() or not (root/INITIAL).is_file():
        time.sleep(3)
    screen=read(folder/"screen.json")
    if not screen["qualified"]:
        atomic_json(folder/"summary.json",screen)
        return
    choice=verified(root/INITIAL)
    lock=verified(folder/"freeze.json")
    if not (root/DECISION).exists():
        if any((folder/"heldout").glob("tuberlin-*.json")):
            raise ValueError("TU scored before bandwidth selection")
        scores=[]
        for s in SKETCHY_SPLITS:
            f=folder/"sketchy"/f"{s}-{lock['winner_id']}.json"
            if checksum(f)!=lock["sources"][f.name]:
                raise ValueError(f"Stale score {f}")
            row=read(f)
            scores.append({"split":s,"mAP":row["mAP"],"precision":row["precision"],"sha256":checksum(f)})
        entries=dict(choice["families"])
        entries[FAMILY]={"scores":scores,"scope":"pairwise","name":lock["winner_id"],
                   "freeze":digest(lock),"settings":lock["winner"],"revision":lock["revision"]}
        primary=max((n for n in entries if entries[n]["scope"]=="pairwise"),key=lambda n:score_key(entries[n]["scores"]))
        best_all=max(entries,key=lambda n:score_key(entries[n]["scores"]))
        decision={"utc":datetime.now(timezone.utc).isoformat(),"parent":digest(choice),"families":entries,
                  "primary":best_all,"pairwise":primary,"selection_data":list(SKETCHY_SPLITS),"new_TU_rows_at_lock":0}
        atomic_json(root/DECISION,decision|{"sha256":digest(decision)})
    decision=verified(root/DECISION)
    run(w,FAMILY,"heldout")
    results={}
    for s in (*SKETCHY_SPLITS,"tuberlin"):
        sub=folder/("heldout" if s=="tuberlin" else "sketchy")
        labels=tensor(sub/f"{s}-labels.pt")["query"]
        current=tensor(sub/"per-query"/f"{s}-{lock['winner_id']}.pt")
        item={}
        for label,params in lock["controls"].items():
            id=digest(params)[:16]
            p=sub/f"{s}-{id}.json";r=read(p)
            val=tensor(sub/"per-query"/(p.stem+".pt"))
            item[label]={k:r[k] for k in ("mAP","precision","queries","gallery","classes")}
            item[label]["delta_vs_this_control"]=paired_delta(current,val,labels)["mAP"]
            if label=="primary":
                item[label]["kd"]=r["kd_check"]
                if max(r["kd_check"]["kd_minus_pipeline_max_abs_same_order_AP_P"])!=0:
                    raise ValueError("KD disagreement")
        results[s]=item
    atomic_json(folder/"summary.json",screen|{"signed_choice":digest(decision),"primary_pairwise_after_refine":decision["pairwise"],
          "config":lock["winner"],"revision":lock["revision"],"splits":results})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--phase",required=True,choices=("study","transfer"))
    args=p.parse_args()
    (study if args.phase=="study" else transfer)(args.workspace)
