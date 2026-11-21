"""Freeze a new image-retrieval study after the complete Sketchy results and before TU."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

from .concept_benchmark import atomic_json, checksum, read
from .crossdomain_benchmark import digest
from .data import SKETCHY_SPLITS
from .map_study import score_key


DECISION = "image-retrieval-round/joint-freeze.json"
FAMILIES = {"retrieval-semantic-whitening":"pairwise","retrieval-vocabulary-tree":"pairwise",
            "retrieval-democratic-regions":"pairwise","retrieval-reciprocal-encoding":"gallery",
            "retrieval-composed-modality":"pairwise"}
HISTORY = {"public-semantic-translation":"pairwise","neighborhood-calibrated-likelihood":"gallery"}


def verified(path):
    row=read(path)
    if row.pop("sha256")!=digest(row):
        raise ValueError(f"Bad lock {path}")
    return row


def joint(w):
    root=w/"results"
    output=root/DECISION
    if output.exists():
        return verified(output)
    entries={}
    for name,scope in (HISTORY|FAMILIES).items():
        folder=root/name
        fresh=name in FAMILIES
        if fresh and any((folder/"heldout").glob("tuberlin-*.json")):
            raise ValueError(f"TU preceded the freeze for {name}")
        lock=verified(folder/"freeze.json")
        cid=lock["winner_id"]
        rows=[]
        for split in SKETCHY_SPLITS:
            file=folder/"sketchy"/f"{split}-{cid}.json"
            sha=checksum(file)
            if sha!=lock["sources"][file.name]:
                raise ValueError(f"Changed Sketchy data for {name}")
            row=read(file)
            rows.append({"split":split,"sha256":sha,"mAP":row["mAP"],"precision":row["precision"]})
        entries[name]={"scope":scope,"fresh":fresh,"name":cid,"freeze":digest(lock),
                       "settings":lock["winner"],"revision":lock["revision"],"scores":rows}
    def best(group):
        options=[name for name,r in entries.items() if group=="all" or r["scope"]==group]
        return max(options,key=lambda n:score_key(entries[n]["scores"]))
    record={"utc":datetime.now(timezone.utc).isoformat(),"families":entries,
            "selection_data":list(SKETCHY_SPLITS),"new_tu_rows":0,"primary":best("all"),
            "pairwise":best("pairwise"),"gallery":best("gallery"),
            "criterion":"mean Sketchy mAP@all and mAP@200 equally, then mean precision"}
    atomic_json(output,record|{"sha256":digest(record)})
    return record


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--wait-for-sketchy",action="store_true")
    opts=parser.parse_args()
    while opts.wait_for_sketchy and not all((opts.workspace/"results"/f/"freeze.json").is_file() for f in FAMILIES):
        time.sleep(3)
    decision=joint(opts.workspace)
    print({k:decision[k] for k in ("primary","pairwise","gallery")})
