"""Screen a cross-language relevance model against FreeDom on identical Sketchy queries."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import zipfile

import torch

from .concept_benchmark import atomic_json,checksum,read
from .crossdomain_benchmark import digest,revision
from .data import SKETCHY_SPLITS
from .map_followup import signed
from .map_study import run as measure, score_key
from .retrieval_evidence import paired_delta,tensor


FAMILY="cross-language-relevance"
REFERENCE="freedom-gallery-transfer"


def run(workspace):
    root=workspace/"results"
    folder=root/FAMILY
    decision,_=signed(root/"map-study/joint-freeze.json")
    free=decision["families"][REFERENCE]["name"]
    measure(workspace,FAMILY,"pilot")
    pilots=[read(folder/f"pilot-{s}.json") for s in SKETCHY_SPLITS]
    previous=[read(root/REFERENCE/f"pilot-{s}.json") for s in SKETCHY_SPLITS]
    for a,b in zip(pilots,previous,strict=True):
        for key in ("seed","queries","classes","gallery","dataset_sha256","checkpoint_sha256","external_vocabulary"):
            if a[key]!=b[key]:
                raise ValueError("Pilot cohorts differ")
    ids=pilots[0]["settings"]
    values={key:score_key([p["settings"][key] for p in pilots]) for key in ids}
    winner=max(values,key=values.get)
    old_score=score_key([p["settings"][free] for p in previous])
    qualified=values[winner]>=old_score
    report={"utc":datetime.now(timezone.utc).isoformat(),"revision":revision(),"qualified":qualified,
            "rule":"shared Sketchy pilot mean mAP, then mean P, relative to the frozen FreeDom variant",
            "reference_id":free,"pilot_winner":pilots[0]["settings"][winner]["parameters"],
            "reference":[p["settings"][free] for p in previous],"best":[p["settings"][winner] for p in pilots],
            "threshold":old_score,"pilot_score":values[winner]}
    atomic_json(folder/"screen.json",report)
    print("Sketchy screen",qualified,values[winner],old_score,flush=True)
    if qualified:
        for stage in ("sketchy","controls","freeze"):
            measure(workspace,FAMILY,stage)
        own,own_sha=signed(folder/"freeze.json")
        incumbent=decision["primary"]
        previous_entry=decision["families"][incumbent]
        candidate=[]
        for split in SKETCHY_SPLITS:
            path=folder/"sketchy"/f"{split}-{own['winner_id']}.json"
            row=read(path)
            if checksum(path)!=own["sources"][path.name] or row["kd_check"]["max_per_query_abs_AP_P"]!=[0.,0.]:
                raise ValueError("Invalid Sketchy replay")
            candidate.append({k:row[k] for k in ("mAP","precision")}|{"split":split,"sha256":checksum(path)})
        final_family=FAMILY if score_key(candidate)>score_key(previous_entry["scores"]) else incumbent
        fate=folder/"additional-joint.json"
        if fate.is_file():
            checked,_=signed(fate)
            if checked["primary"]!=final_family or checked["families"][FAMILY]["freeze"]!=own_sha:
                raise ValueError("Inconsistent additional lock")
        else:
            if any((folder/"heldout").glob("tuberlin-*.json")):
                raise ValueError("TU data precede the new lock")
            record={"utc":datetime.now(timezone.utc).isoformat(),"primary":final_family,
                    "selection_data":list(SKETCHY_SPLITS),"families":{incumbent:previous_entry,
                         FAMILY:{"name":own["winner_id"],"freeze":own_sha,"scores":candidate,"tu_rows_before_freeze":0}}}
            atomic_json(fate,record|{"sha256":digest(record)})
        measure(workspace,FAMILY,"heldout",heldout_decision=fate)
        report["primary_after_full_Sketchy"]=final_family
        report["parameters"]=own["winner"]
        report["full_scores"]={}
        for split in (*SKETCHY_SPLITS,"tuberlin"):
            part="heldout" if split=="tuberlin" else "sketchy"
            key=own["winner_id"]
            row=read(folder/part/f"{split}-{key}.json")
            obs=tensor(folder/part/"per-query"/f"{split}-{key}.pt")
            ref=tensor(root/incumbent/part/"per-query"/f"{split}-{previous_entry['name']}.pt")
            ql=tensor(folder/part/f"{split}-labels.pt")["query"]
            if row["kd_check"]["max_per_query_abs_AP_P"]!=[0.,0.] or not torch.isfinite(obs).all():
                raise ValueError("Invalid official evaluation")
            controls={}
            for c,parameters in own["controls"].items():
                r=read(folder/part/f"{split}-{digest(parameters)[:16]}.json")
                controls[c]={k:r[k] for k in ("mAP","precision")}
            report["full_scores"][split]={**{k:row[k] for k in ("mAP","precision","queries","gallery","classes")},
                    "versus_incumbent":paired_delta(obs,ref,ql),"controls":controls}
    atomic_json(folder/"result.json",report)
    artifact=workspace/"map-relevance-evidence.zip"
    with zipfile.ZipFile(artifact,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(p for p in folder.rglob("*") if p.is_file()):
            archive.write(p,p.relative_to(workspace))
    with zipfile.ZipFile(artifact) as z:
        if z.testzip() is not None:raise ValueError("Corrupt relevance archive")
        count=len(z.namelist())
    atomic_json(workspace/"map-relevance-archive-info.json",{"size":artifact.stat().st_size,"sha":checksum(artifact),"count":count})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
