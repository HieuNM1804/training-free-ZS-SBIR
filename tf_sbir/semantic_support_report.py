"""Combine disjoint development and confirmation predictions, preserving benchmark query order."""
import argparse
import json
from pathlib import Path

import torch

from .benchmark import merge_inputs
from .concept_benchmark import atomic_json
from .crossdomain_inputs import runtime
from .data import SKETCHY_SPLITS
from .depiction_path_diagnostic import class_interval


def verify_rows(rows, summary):
    for arm,v in rows.items():
        torch.testing.assert_close(100*v.double().mean(0),
                                   torch.tensor(summary["metrics"][arm],dtype=torch.double),
                                   atol=1e-8,rtol=0)


def run(family,selected):
    work=Path("/content/sbir")
    folder=work/"results"/family
    cfg=runtime(work,"sketchy")
    _,queries=merge_inputs(cfg.root,SKETCHY_SPLITS,"sketch")
    _,photos=merge_inputs(cfg.root,SKETCHY_SPLITS,"photo")
    report={}
    for split,(ids,labels) in queries.items():
        stages=[]
        metadata=[]
        for name in ("pilot","confirmation"):
            path=folder/(split+"-"+name)
            record=torch.load(path.with_suffix(".pt"),weights_only=True)
            summary=json.loads(path.with_suffix(".json").read_bytes())
            assert summary["queries"]==len(record["qi"]) and summary["seed"]==42
            assert torch.equal(record["pi"],photos[split][0])
            verify_rows(record["rows"],summary)
            stages.append(record)
            metadata.append(summary)
        assert metadata[0]["evaluator"]==metadata[1]["evaluator"]
        join=torch.cat([r["qi"] for r in stages])
        assert len(join)==len(ids)==len(join.unique()) and torch.isin(ids,join).all()
        position={qid:i for i,qid in enumerate(join.tolist())}
        reorder=torch.tensor([position[qid] for qid in ids.tolist()])
        assert torch.equal(torch.cat([r["labels"] for r in stages])[reorder],labels)
        arms=tuple(stages[1]["rows"])
        rows={a:torch.cat([r["rows"][a] for r in stages])[reorder].double() for a in arms}
        results={}
        for a,v in rows.items():
            delta=100*(rows[selected]-v)
            results[a]=dict(metrics=(100*v.mean(0)).tolist(),
                            selected_minus=delta.mean(0).tolist(),
                            class_ap=class_interval(delta[:,0],labels,seed=42))
        report[split]=dict(queries=len(ids),gallery=len(photos[split][0]),classes=int(labels.unique().numel()),
                          pilot_commit=metadata[0]["commit"],confirmation_commit=metadata[1]["commit"],arms=results)
    atomic_json(folder/"complete-sketchy.json",dict(selected=selected,evaluator=metadata[0]["evaluator"],splits=report))
    print({s:r["arms"][selected]["metrics"] for s,r in report.items()},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("family",choices=("semantic-predictive-support","semantic-predictive-diffusion"))
    p.add_argument("selected")
    a=p.parse_args()
    run(a.family,a.selected)
