"""Evaluate word-conditional visual verification against separately tuned simple fusions."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .encoding import save_tensor
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import stratified_sample
from .retrieval_evidence import load_kd, paired_delta
from .verification_inputs import FAMILY, get_inputs
from .word_conditional_verifier import fit_gallery, lexical_groups, partition, visual_score, within_group_update, zscore

SPLITS = ("sketchy_1", "sketchy_2")
SIMPLE = ("raw_native", "detail", "cross_view", "centered_cross")
OTHER = ("global", "cross_only", "pooled_center")
WEIGHTS = (.04, .12, .35, .8)
BASE = ("base", 0, 0., 0.)


def key(c):
    return json.dumps(c, separators=(",", ":"))


def grid():
    cells = [(kind, k, r, a) for kind in ("joint", *OTHER) for k in (16, 64, 192) for r in (.05, .5) for a in WEIGHTS]
    return [BASE, *cells, *((kind, 0, 0., a) for kind in SIMPLE for a in WEIGHTS)]


def best(values, choices, n):
    keys = sorted((k for k in values if json.loads(k)[0] in choices), key=values.get, reverse=True)
    return [tuple(json.loads(k)) for k in keys[:n]]


def index_setting(kind):
    return kind if kind in ("global", "permuted", "sham") else "lexical"


def simple_views(qa, qb, a, b):
    return {"raw_native": qa @ a.T, "detail": qb @ b.T, "cross_view": .5 * (qa @ b.T + qb @ a.T),
            "centered_cross": .5 * (F.normalize(qa - a.mean(0)) @ F.normalize(b - b.mean(0)).T
                                   + F.normalize(qb - b.mean(0)) @ F.normalize(a - a.mean(0)).T)}


def evaluate(split, data, y, g, ids, cells, kd, primary, workers):
    a, b = data["gallery_native"].cuda().float(), data["gallery_detail"].cuda().float()
    lex = lexical_groups(data["public_words"].cuda())
    folds = partition(lex)
    models = {}
    for kind, _, reg, _ in cells:
        if kind not in (*SIMPLE, "base"):
            table = (index_setting(kind), reg)
            if table not in models:
                models[table] = fit_gallery(a, b, lex, folds, *table)
    diags = {str(k): [v.diagnostics for v in bank] for k, bank in models.items()}
    hidden = sorted({c[:3] for c in cells})
    rows, errors, details = defaultdict(list), {}, defaultdict(list)
    reference = Replay(kd, g, split) if kd else None
    truth = g.cuda()
    with (ThreadPoolExecutor(max_workers=workers) if kd else nullcontext()) as pool:
        for ids_batch in ids.split(48):
            qa = data["query_native"][ids_batch].cuda().float()
            qb = data["query_detail"][ids_batch].cuda().float()
            base = zscore(data["regional_score"][ids_batch].cuda())
            simple = simple_views(qa, qb, a, b)
            labels = y[ids_batch]
            native_metrics, _, _, _ = map_values(simple["raw_native"], labels.cuda(), truth, split)
            base_metrics, _, base_hits, _ = map_values(base, labels.cuda(), truth, split)
            details["native_metrics"].append(native_metrics.cpu())
            details["base_metrics"].append(base_metrics.cpu())
            details["base_top1"].append(base_hits[:, 0].cpu())
            for kind, k, ridge in hidden:
                if kind in SIMPLE:
                    aux = zscore(simple[kind])
                elif kind == "base":
                    aux = None
                else:
                    fitted = models[index_setting(kind), ridge]
                    mode = kind if kind in ("pooled_center", "cross_only") else "joint"
                    aux = zscore(visual_score(fitted, qa, qb, len(g), k, mode))
                for c in (x for x in cells if x[:3] == (kind, k, ridge)):
                    nm = key(c)
                    if kind == "base":
                        score = base
                    elif kind == "within_mass":
                        score = within_group_update(base, aux, lex, c[3])
                    elif kind == "between_mass":
                        score = base + c[3] * aux - within_group_update(base, aux, lex, c[3]) + base
                    else:
                        score = base + c[3] * aux
                    quick, rank, hits, _ = map_values(score, labels.cuda(), truth, split)
                    if reference:
                        actual = reference.score(rank, labels, pool)
                        err = float((actual - quick.cpu()).abs().max())
                        if err > 2e-6:
                            raise ValueError("Original KD-SBIR per-query replay differs")
                        errors[nm] = max(errors.get(nm, 0.), err)
                        if nm == primary:
                            details["top25"].append(rank[:, :25].int().cpu())
                            details["new_top1"].append(hits[:, 0].cpu())
                    else:
                        actual = quick.cpu()
                    rows[nm].append(actual)
    return {k: torch.cat(v) for k, v in rows.items()}, errors, diags, {k: torch.cat(v) for k,v in details.items()}


@torch.inference_mode()
def run(w, phase, workers):
    root = w / "results" / FAMILY
    rev = revision()
    spec = {"revision": rev, "grid": grid(), "pilot_seed": 63241, "pilot_per_class": 36, "prior_word_groups": 64,
            "folds": 4, "primary": "joint", "selection": "mean full unseen Sketchy mAP, then mean precision"}
    sig = digest(spec)
    if (root / "protocol.json").exists() and read(root / "protocol.json")["sha256"] != sig:
        raise ValueError("Conflicting protocol")
    atomic_json(root / "protocol.json", spec | {"sha256": sig})
    kd, _ = load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("Unexpected evaluator")
    stages = {"pilot": ("pilot",), "lock": ("shortlist",), "audit": ("final",), "all": ("pilot", "shortlist", "final")}[phase]
    for stage in stages:
        primary = None
        if stage == "pilot":
            cells = grid()
        elif stage == "shortlist":
            vals = read(root / "pilot.json")["means"]
            cells = sorted({BASE, *best(vals, {"joint"}, 4), *best(vals, SIMPLE, 4),
                            *[cell for kind in OTHER for cell in best(vals, {kind}, 2)]})
        else:
            freeze = read(root / "freeze.json")
            if freeze["revision"] != rev or freeze["shortlist_sha256"] != checksum(root / "shortlist.json"):
                raise ValueError("Invalid Sketchy lock")
            chosen, simple, contrast = (tuple(freeze[k]) for k in ("primary", "simple", "contrast"))
            primary = key(chosen)
            _, k, reg, weight = chosen
            cells = sorted({BASE, chosen, simple, contrast, ("raw_native", 0, 0., weight), ("centered_cross", 0, 0., weight),
                            *[(mode, k, reg, weight) for mode in (*OTHER, "permuted", "sham", "within_mass", "between_mass")]})
        table = {}
        for ds in (("sketchy", "tuberlin") if stage == "final" else ("sketchy",)):
            for s, (data, y, g) in get_inputs(w, ds).items():
                file = root / ((s if stage == "final" else stage + "-" + s) + ".json")
                if file.exists():
                    old = read(file)
                    if old["signature"] != sig or (stage in ("pilot", "final") and old["per_query_sha256"] != checksum(file.with_suffix(".pt"))):
                        raise ValueError("Stale split checkpoint")
                    table[s] = old["values"]
                    continue
                ids = stratified_sample(y, 36, 63241) if stage == "pilot" else torch.arange(len(y))
                result, err, checks, detail = evaluate(s, data, y, g, ids, cells, kd if stage == "final" else None, primary, workers)
                vals = {name: v.mean(0).tolist() for name, v in result.items()}
                info = {"signature": sig, "queries": len(ids), "gallery": len(g), "values": vals, "gallery_checks": checks}
                native_can_help = (detail["native_metrics"][:,0] - detail["base_metrics"][:,0]) > .05
                info["native_advantage_queries"] = int(native_can_help.sum())
                if stage in ("pilot", "final"):
                    tensor = file.with_suffix(".pt")
                    save_tensor({"metrics": result, "query_labels": y[ids], "split_ids": ids} | detail, tensor)
                    info["per_query_sha256"] = checksum(tensor)
                if stage == "final":
                    same = result[key(BASE)]
                    changed = result[primary]
                    info.update({"protocol": {"split": s, "queries": len(y), "gallery": len(g), "classes": len(g.unique()), "checkpoint": CHECKPOINT,
                                             "vocabulary_size": 20931, "kd_sha256": KD_FUNCTION_SHA, "revision": rev, "gpu": torch.cuda.get_device_name(),
                                             "map_k": 200 if s == "sketchy_2" else 0, "p_k": 200 if s == "sketchy_2" else 100},
                                 "scores": {n: dict(zip(("mAP","precision"), value)) for n,value in vals.items()},
                                 "primary": primary, "baseline": key(BASE), "matched_base": key(simple), "best_contrast": key(contrast),
                                 "vs_controls": {n: paired_delta(changed, val, y) for n, val in result.items()}, "kd_max_abs_error": err,
                                 "original_kd_queries_per_condition": len(y), "native_advantage_diagnostic": {
                                     "count": int(native_can_help.sum()),
                                     "change_when_native_ahead_pp": (100 * (changed - same)[native_can_help].mean(0)).tolist() if native_can_help.any() else None,
                                     "change_other_queries_pp": (100 * (changed - same)[~native_can_help].mean(0)).tolist()},
                                 "top1_changes": {"fixed_errors": int((~detail["base_top1"] & detail["new_top1"]).sum()),
                                                  "broken_hits": int((detail["base_top1"] & ~detail["new_top1"]).sum())}})
                table[s] = vals
                atomic_json(file, info)
                selected = best(vals, {"joint"}, 1)[0]
                print(stage, s, selected, vals[key(selected)], "highest", max(vals, key=vals.get), vals[max(vals, key=vals.get)], flush=True)
        if stage != "final":
            means = {k: [sum(table[s][k][i] for s in SPLITS) / 2 for i in (0, 1)] for k in table[SPLITS[0]]}
            atomic_json(root / (stage + ".json"), {"means": means, "splits": table})
            if stage == "shortlist":
                lock = {"revision": rev, "shortlist_sha256": checksum(root / "shortlist.json"),
                        "primary": best(means, {"joint"}, 1)[0], "simple": best(means, SIMPLE, 1)[0],
                        "contrast": best(means, set(OTHER) | set(SIMPLE), 1)[0]}
                atomic_json(root / "freeze.json", lock | {"sha256": digest(lock)})
                print("locked", lock, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    p.add_argument("--phase", required=True, choices=("pilot", "lock", "audit", "all"))
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()
    run(args.workspace, args.phase, args.workers)
