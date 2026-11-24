"""Assess decision identifiability while matching the sketch or matching the two rival photographs."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .depiction_path_diagnostic import class_interval


MEASURES = ("margin", "answer", "separation")


def compare(preferred, alternative):
    return np.array([(1+np.sign(abs(preferred[k])-abs(alternative[k])))/2 for k in MEASURES])


def summarize(scores, clusters, counts):
    result = dict(counts=counts, comparisons=len(scores), clusters=len(set(clusters)))
    if not scores:
        return result
    values = torch.tensor(np.stack(scores),dtype=torch.double)
    groups = torch.tensor(clusters)
    result["concordance"] = dict(zip(MEASURES, values.mean(0).tolist()))
    result["certificate_minus_chance"] = class_interval(values[:,0]-.5,groups,seed=42)
    result["certificate_minus_absolute_vote"] = class_interval(values[:,0]-values[:,1],groups,seed=42)
    return result


def boundary(rows, name):
    by_query = defaultdict(list)
    for r in rows:
        by_query[r["q"]].append(r)
    scores, classes = [], []
    counts = defaultdict(int)
    for pair in by_query.values():
        assert len(pair)==2 and pair[0]["mode"]==pair[1]["mode"]
        a,b = (r[name] for r in pair)
        if a["acc"]==.5 or b["acc"]==.5:
            counts["tied_decision"] += 1
        elif a["acc"]==b["acc"]:
            counts["same_correctness"] += 1
        else:
            scores.append(compare(a,b) if a["acc"] else compare(b,a))
            classes.append(pair[0]["mode"])
            counts["one_correct"] += 1
    return summarize(scores, classes, dict(counts))


def redraw(rows, name, mode):
    by_pair = defaultdict(lambda:defaultdict(dict))
    for r in rows:
        if r["mode"]==mode:
            by_pair[r["group"]][r["pos"]][r["view"]] = r[name]
    scores, pairs = [], []
    counts = defaultdict(int)
    for group, sources in by_pair.items():
        if len(sources)!=2:
            counts["incomplete_pair"] += 1
            continue
        a,b = sources.values()
        assert set(a)==set(b)=={0,1}
        if any(s[i]["acc"]==.5 for s in (a,b) for i in (0,1)):
            counts["tied_decision"] += 1
            continue
        a_stable = a[0]["acc"]==a[1]["acc"]
        b_stable = b[0]["acc"]==b[1]["acc"]
        if a_stable==b_stable:
            counts["same_stability"] += 1
            continue
        preferred,alternative = (a,b) if a_stable else (b,a)
        picks = np.mean([compare(preferred[i],alternative[i]) for i in (0,1)],axis=0)
        assert picks[2]==.5
        scores.append(picks)
        pairs.append(group)
        counts["one_stable"] += 1
    return summarize(scores, pairs, dict(counts))


def run():
    root = Path("/content/sbir/results/semantic-invariant-question")
    results = {}
    for split in ("sketchy_1","sketchy_2"):
        rows = json.loads((root/(split+"-pairs.json")).read_bytes())
        results[split] = {name:dict(boundary=boundary(rows["boundary"],name),
                                    same_class=redraw(rows["sources"],name,0),
                                    different_class=redraw(rows["sources"],name,1))
                          for name in ("invariant","pooled")}
    atomic_json(root/"confidence.json", dict(commit=revision(),seed=42,results=results))
    print(results, flush=True)


if __name__=="__main__":
    run()
