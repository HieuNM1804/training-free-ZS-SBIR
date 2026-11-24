"""Test frozen whole-sketch semantic decisions on source redraws and category-boundary pairs."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import revision
from .depiction_path_diagnostic import class_interval
from .gallery_panel_resume import semantic_inputs
from .semantic import read_words
from .semantic_invariant_question import Question, validate


FAMILY = "semantic-invariant-question"
ARMS = ("invariant", "pooled")


def select(ref, source):
    meta = ref["meta"].numpy()
    rng = np.random.default_rng(42)
    selected = []
    if source:
        for mode in np.unique(meta[:, 4]):
            part = meta[meta[:, 4] == mode]
            pairs = np.unique(part[:, 5])
            chosen = rng.choice(pairs, min(24, len(pairs)), replace=False)
            for query, redraw, photo, rival, _, group in part[np.isin(part[:, 5], chosen)]:
                for view, q in enumerate((query, redraw)):
                    selected.append((q, photo, rival, mode, group, view))
    else:
        ids, counts = np.unique(meta[:, 0], return_counts=True)
        both = meta[np.isin(meta[:, 0], ids[counts == 2])]
        for label in np.unique(both[:, 5]):
            part = both[both[:, 5] == label]
            ids = np.unique(part[:, 0])
            chosen = rng.choice(ids, min(5, len(ids)), replace=False)
            for q, a, b, direction, _, cls in part[np.isin(part[:, 0], chosen)]:
                pos, neg = (a,b) if direction else (b,a)
                selected.append((q,pos,neg,cls,q,direction))
    return np.asarray(selected, dtype=np.int64)


def word_rows(features, ids, heads, scale):
    rows = {}
    for batch in torch.tensor(sorted(ids)).split(128):
        feature = F.normalize(features[batch].cuda().float(), dim=-1)
        readings = torch.stack([(scale * (feature @ head.T)).double().softmax(-1) for head in heads], dim=1)
        rows.update(zip(batch.tolist(), readings.cpu().numpy()))
    return rows


def score_pair(pair, state, queries, photos):
    a, b = (photos[i] for i in pair)
    output = {}
    for name in ARMS:
        test = Question(a[:2], b[:2], invariant=(name == "invariant"))
        g = test.best.answer
        fitted = np.asarray([a[:2].mean(0) @ g, b[:2].mean(0) @ g])
        held = np.asarray([a[2] @ g, b[2] @ g])
        output[name] = dict(bare_gap=float(np.abs(held-fitted).mean()),
                            fitted=fitted.tolist(), held_bare=held.tolist(),
                            answers={str(q):test.read(queries[q][0]) for q in sorted(state)})
    output["direct"] = {str(q):float(queries[q][0] @ (a[1]-b[1])) for q in state}
    return output


def accuracy(score):
    return float((1+np.sign(score))/2)


def describe(trials, records, source):
    table = []
    for q, pos, neg, mode, group, view in trials.tolist():
        pair = tuple(sorted((pos, neg)))
        sign = 1 if pos == pair[0] else -1
        record = records[pair]
        row = dict(q=q, pos=pos, neg=neg, group=group, mode=mode, view=view,
                   direct=accuracy(sign * record["direct"][str(q)]))
        for name in ARMS:
            method = record[name]
            ans = method["answers"][str(q)]
            response = sign * ans["response"]
            target = sign * method["fitted"][pair.index(pos)]
            row[name] = dict(acc=accuracy(sign*ans["signed_certificate"]), answer=response,
                margin=sign*ans["signed_certificate"], separation=2*ans["decision_accuracy"]-1,
                calibration=abs(response-target), bare_gap=method["bare_gap"],
                gap=ans["primal_dual_gap"], feasibility=ans["equality_residual"])
        table.append(row)
    out = {}
    modes = (("same_source_class", 0), ("different_source_class", 1)) if source else (("category", None),)
    for label, mode in modes:
        part = [r for r in table if mode is None or r["mode"] == mode]
        group = torch.tensor([r["group"] if source else r["mode"] for r in part])
        report = dict(decisions=len(part), queries=len(set(r["q"] for r in part)),
                      photo_pairs=len(set(tuple(sorted((r["pos"],r["neg"]))) for r in part)),
                      direct_accuracy=np.mean([r["direct"] for r in part]).item())
        values = {}
        for name in ARMS:
            values[name] = {k:torch.tensor([r[name][k] for r in part],dtype=torch.double) for k in part[0][name]}
            v = values[name]
            arm = {k:float(v[k].mean()) for k in ("acc","calibration","bare_gap","separation","margin")}
            arm.update(ties=int((v["margin"] == 0).sum()), max_gap=float(v["gap"].max()),
                       max_equality_residual=float(v["feasibility"].max()))
            if source:
                by_source = {}
                for r in part:
                    by_source.setdefault((r["group"],r["pos"]),{})[r["view"]] = np.sign(r[name]["margin"])
                signs = np.asarray([[x[0],x[1]] for x in by_source.values()])
                arm["strict_redraw_agreement"] = float((signs[:,0]*signs[:,1] > 0).mean())
            report[name] = arm
        effect = {}
        for measure in ("acc","bare_gap","calibration","separation"):
            delta = values["invariant"][measure] - values["pooled"][measure]
            effect[measure] = dict(mean=float(delta.mean()), interval=class_interval(delta,group,seed=42))
        report["invariant_minus_pooled"] = effect
        delta = values["invariant"]["acc"]-values["pooled"]["acc"]
        report["positive_switch_mass"] = float(delta.clamp_min(0).sum())
        report["negative_switch_mass"] = float((-delta).clamp_min(0).sum())
        out[label] = report
    return table, out


@torch.inference_mode()
def run():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    validate()
    work = Path("/content/sbir")
    out = work / "results" / FAMILY
    out.mkdir(parents=True, exist_ok=True)
    cfg, _, qs, _, gs, qf, pf, src, tgt, scale = semantic_inputs(work, "sketchy")
    bare, _ = bare_words(cfg, read_words(cfg.vocabulary))
    stages = {"sources":"regional-semantic-interactions", "boundary":"gcwe-action-category-transfer"}
    trials = {}
    for split in qs:
        for stage, reference in stages.items():
            path = work / "results" / reference / (split + ".pt")
            ref = torch.load(path, mmap=True, weights_only=True)
            assert torch.equal(ref["pi"], gs[split][0])
            if stage == "sources":
                assert torch.equal(ref["qi"], qs[split][0])
            trials[split,stage] = select(ref, stage == "sources")
    grouped = {}
    for rows in trials.values():
        for q, a, b in rows[:, :3].tolist():
            grouped.setdefault(tuple(sorted((a,b))), set()).add(q)
    qids = set.union(*grouped.values())
    pids = set(i for pair in grouped for i in pair)
    queries = word_rows(qf, qids, [src], scale)
    photos = word_rows(pf, pids, [src,tgt,bare], scale)
    commit = revision()
    # Include the exact input readouts so restored or regenerated caches cannot silently mix.
    records = {}
    for number, (pair, qset) in enumerate(sorted(grouped.items()),1):
        digest = hashlib.sha256(commit.encode())
        for i in pair:
            digest.update(photos[i].tobytes())
        for q in sorted(qset):
            digest.update(q.to_bytes(8,"little"))
            digest.update(queries[q].tobytes())
        path = work / "cache" / FAMILY / (digest.hexdigest()+".json")
        if not path.is_file():
            atomic_json(path, score_pair(pair, qset, queries, photos))
        records[pair] = json.loads(path.read_text())
        if number % 50 == 0:
            print(number, "of", len(grouped), "photo pairs", flush=True)
    for split in qs:
        paired, report = {}, {}
        for stage in stages:
            paired[stage], report[stage] = describe(trials[split,stage],records,stage=="sources")
        atomic_json(out / (split+"-pairs.json"),paired)
        atomic_json(out / (split+".json"),dict(seed=42,commit=commit,gpu=torch.cuda.get_device_name(),assessment=report))
        print(split,report["boundary"]["category"]["invariant_minus_pooled"],flush=True)


if __name__ == "__main__":
    run()
