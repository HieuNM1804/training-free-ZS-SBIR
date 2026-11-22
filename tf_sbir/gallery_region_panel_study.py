"""Measure whether a region chosen under one gallery fit survives independent readout."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
from pathlib import Path
import time

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES as SPLIT_SIZES, bare_words
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .encoding import save_tensor
from .foundation import load_words
from .gallery_depiction_study import PRIOR_RUN, views
from .gallery_region_panels import FAMILY, FOLDS, GROUPS, SIZES, read_regions
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs, stratified_sample
from .retrieval_evidence import load_kd, paired_delta
from .semantic import read_words


TEMPERATURES = (.35, 1., 2.5)
BLENDS = (.25, .65, 1.)
GAINS = (.35, 1., 3.)
MEASUREMENTS = ("cross", "same", "average", "wrong")
DIFFERENCES = ("cross_diff", "wrong_diff", "reverse_diff")


def key(kind, size, mode, temp, weight):
    return f"{kind}/{size}/{mode}/{temp:g}/{weight:g}"


def parse(name):
    kind, size, mode, temp, weight = name.split("/")
    return kind, int(size), mode, float(temp), float(weight)


def grid():
    rows = ["regional"]
    for kind in GROUPS:
        for size in SIZES:
            rows.extend(key(kind, size, "uniform", 0, b) for b in BLENDS)
            for temp in TEMPERATURES:
                rows.extend(key(kind, size, m, temp, b) for m in MEASUREMENTS for b in BLENDS)
                rows.extend(key(kind, size, m, temp, g) for m in DIFFERENCES for g in GAINS)
    return rows


def select(means):
    def best(modes):
        return max((k for k in means if k != "regional" and parse(k)[2] in modes), key=means.get)

    main = best(("cross", "cross_diff"))
    kind, size, mode, temp, weight = parse(main)
    if mode == "cross":
        matched = [key(kind, size, m, temp, weight) for m in MEASUREMENTS]
        matched.append(key(kind, size, "uniform", 0, weight))
    else:
        matched = [key(kind, size, m, temp, weight) for m in DIFFERENCES]
    return {"nominee": main, "matched": matched,
            "best_same": best(("same",)), "best_average": best(("average",)),
            "best_uniform": best(("uniform",)), "best_wrong": best(("wrong", "wrong_diff")),
            "best_reverse": best(("reverse_diff",)), "best_cross_replacement": best(("cross",)),
            "best_cross_difference": best(("cross_diff",))}


def audited_rows(lock):
    return sorted({"regional", *lock["matched"], *(v for k, v in lock.items() if k != "matched")})


def evaluate(w, split, queries, gallery, src, scale, view, ids, rows, kd, workers, batch_size=32):
    qi, labels = queries
    _, truth = gallery
    if (len(qi), len(truth), len(truth.unique())) != SPLIT_SIZES[split]:
        raise ValueError("Changed original test split")
    hist = torch.load(w/"cache/pragmatic-gallery-competition"/(split+"-named-rgb16.pt"), mmap=True, weights_only=True)
    full = w/"cache"/PRIOR_RUN/"full"/(split+".pt")
    if full.is_file():
        locked = torch.load(full, mmap=True, weights_only=True)["regional_score"]
    else:
        locked = torch.load(w/"cache"/FAMILY/(split+"-regional-reference.pt"), mmap=True, weights_only=True)
    cpu = torch.load(w/"cache"/FAMILY/(split+"-donor-indices.pt"), mmap=True, weights_only=True)["banks"]
    groups = defaultdict(list)
    for row in rows:
        if row != "regional":
            kind, size, mode, temp, weight = parse(row)
            groups[kind, size].append((row, mode, temp, weight))
    banks = {group: [[v.cuda() for v in cpu[(*group, j)]] for j in range(FOLDS)] for group in groups}
    metrics, top, effects = defaultdict(list), defaultdict(list), defaultdict(list)
    errors = {}
    truth_gpu = truth.cuda()
    evaluator = Replay(kd, truth, split) if kd else None
    with (ThreadPoolExecutor(max_workers=workers) if kd else nullcontext()) as pool:
        for step, ix in enumerate(ids.split(batch_size)):
            crop = view(qi[ix]).float().contiguous()
            base = locked[ix].cuda()
            global_log = hist[ix].cuda().clamp_min(1e-30).log()
            native_region = (base - .65*global_log)/.35
            batch_truth = labels[ix].cuda()

            def record(name, score):
                val, rank, rel, _ = map_values(score, batch_truth, truth_gpu, split)
                val = val.cpu()
                if evaluator:
                    exact = evaluator.score(rank, labels[ix], pool)
                    err = float((exact-val).abs().max())
                    if err > 2e-6:
                        raise ValueError(f"KD replay differs on {name} by {err}")
                    errors[name] = max(err, errors.get(name, 0.))
                    val = exact
                metrics[name].append(val)
                top[name].append(rel[:, 0].cpu())

            record("regional", base)
            for group, arms in groups.items():
                temps = sorted({t for _, m, t, _ in arms if m != "uniform"})
                measured, pooled = read_regions(crop, src, banks[group], scale, temps)
                for temp, scores in measured.items():
                    difference = scores["cross"]-scores["same"]
                    wrong = scores["wrong"]-scores["same"]
                    di = difference-difference.mean(1, keepdim=True)
                    wi = wrong-wrong.mean(1, keepdim=True)
                    corr = (di*wi).mean(1)/(di.square().mean(1)*wi.square().mean(1)).clamp_min(1e-25).sqrt()
                    effects[f"{group[0]}/{group[1]}/{temp:g}"].append(torch.stack(
                        (di.square().mean(1).sqrt(), scores["cross"].std(1), corr), 1).cpu())
                for name, mode, temp, weight in arms:
                    if mode in DIFFERENCES:
                        sc = measured[temp]
                        source = sc["wrong"] if mode=="wrong_diff" else sc["cross"]
                        sign = -1. if mode == "reverse_diff" else 1.
                        score = base + sign*weight*(source-sc["same"])
                    else:
                        region = pooled if mode == "uniform" else measured[temp][mode]
                        score = base + .35*weight*(region-native_region)
                    record(name, score)
            if step%8 == 0:
                print("KD" if kd else "scan", split, step*batch_size, len(ids), flush=True)

    if set(metrics) != set(rows):
        raise ValueError("Missing pilot condition")
    return ({k: torch.cat(v) for k, v in metrics.items()}, {k: torch.cat(v) for k, v in top.items()},
            {k: torch.cat(v) for k, v in effects.items()}, errors)


@torch.inference_mode()
def run(w, workers):
    start = time.monotonic()
    cfg, paths, queries, _, galleries, _, _ = global_inputs(w, "sketchy")
    vocab = read_words(cfg.vocabulary)
    if len(vocab) != 20931:
        raise ValueError("Public vocabulary changed")
    src = load_words(cfg, vocab)[0].cuda()
    _, scale = bare_words(cfg, vocab)
    kd, _ = load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("Original KD-SBIR evaluator changed")
    view = views(w, "sketchy", paths)
    torch.backends.cuda.matmul.allow_tf32 = True
    root = w/"results"/FAMILY
    rows = grid()
    refs = {s: read(w/"cache"/FAMILY/(s+"-donor-indices.json")) for s in queries}
    protocol = {"revision": revision(), "checkpoint": CHECKPOINT, "vocabulary": checksum(cfg.vocabulary),
                "KD": KD_FUNCTION_SHA, "sample_seed": 63241, "queries_per_class": 36,
                "rows": rows, "references": refs, "selection": "equal split mAP then precision"}
    signature = digest(protocol)
    if (root/"protocol.json").exists() and read(root/"protocol.json")["sha256"] != signature:
        raise ValueError("Study differs from the existing pilot")
    atomic_json(root/"protocol.json", protocol | {"sha256": signature})
    samples = {s: stratified_sample(y, 36, 63241) for s, (_, y) in queries.items()}
    captures = {}
    for s, ids in samples.items():
        dest = root/("pilot-"+s+".json")
        if dest.exists():
            previous = read(dest)
            if previous["signature"] != signature or previous["per_query_sha256"] != checksum(dest.with_suffix(".pt")):
                raise ValueError("Stale pilot artifacts")
            captures[s] = previous["values"]
            continue
        metrics, top, effects, _ = evaluate(w, s, queries[s], galleries[s], src, scale, view, ids, rows, None, workers)
        save_tensor({"query_ids": ids, "query_labels": queries[s][1][ids], "metrics": metrics,
                     "top1_correct": top, "effects": effects}, dest.with_suffix(".pt"))
        captures[s] = {k:v.mean(0).tolist() for k,v in metrics.items()}
        atomic_json(dest, {"signature": signature, "queries":len(ids), "gallery":SPLIT_SIZES[s][1],
                           "values":captures[s], "effect_columns":["centered cross-minus-same RMS",
                           "cross log-score SD", "correlation with wrong-minus-same"],
                           "effects":{k:v.mean(0).tolist() for k,v in effects.items()},
                           "per_query_sha256": checksum(dest.with_suffix(".pt"))})
    means = {k:[sum(v[k][i] for v in captures.values())/len(captures) for i in (0,1)] for k in rows}
    chosen = select(means)
    selected = audited_rows(chosen)
    audited, paired, error, top1 = {}, {}, {}, {}
    for s, ids in samples.items():
        metric, top, _, err = evaluate(w, s, queries[s], galleries[s], src, scale, view, ids, selected, kd, workers)
        saved = torch.load(root/("pilot-"+s+".pt"), mmap=True, weights_only=True)
        discrepancy = max(float((metric[k]-saved["metrics"][k]).abs().max()) for k in metric)
        if discrepancy > 2e-6:
            raise ValueError(f"Original evaluator disagrees with scan {discrepancy}")
        main = chosen["nominee"]
        audited[s] = {k:v.mean(0).tolist() for k,v in metric.items()}
        paired[s] = {k:paired_delta(metric[main], v, saved["query_labels"]) for k,v in metric.items()}
        top1[s] = {"fixed":int((~top["regional"] & top[main]).sum()), "broken":int((top["regional"] & ~top[main]).sum())}
        error[s] = {"original_queries_per_condition": len(ids), "max_error": max(err.values()), "replay_error": discrepancy}
    atomic_json(root/"pilot.json", chosen | {"means": means, "audited": audited, "paired": paired, "top1": top1,
                                             "KD": error, "GPU": torch.cuda.get_device_name(),
                                             "seconds": time.monotonic()-start,"peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20})
    print("selected", chosen["nominee"], "means", means[chosen["nominee"]], means["regional"], flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    cli.add_argument("--workers", type=int, default=8)
    args = cli.parse_args()
    run(args.workspace, args.workers)
