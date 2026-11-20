"""Select a common phase-appearance retrieval setting on both Sketchy splits and test it on the 30 TU classes."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from .benchmark import merge_inputs
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT, projection, runtime
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .domain_prompt_gallery import visual_rows
from .encoding import load_global_inputs
from .phase_agreement import Config, evaluate, grid, name
from .phase_views import phase_bank
from .transfer_benchmark import write_result


BASE = "phase-regions-rgb-graph"
LOCK = "phase-sketchy-to-tuberlin-lock.json"
PROTOCOL = {
    "sketchy_1": (14179, 17100, 100, 0),
    "sketchy_2": (12694, 12552, 200, 200),
    "tuberlin": (2130, 27262, 100, 0),
}


def entries():
    return {BASE: None} | grid()


def params(value):
    if value is None:
        return {"views": 5, "gallery_k": 192, "channels": "images"}
    c, m = value
    return asdict(c) | {"fusion": m}


def read(directory, split, n, value, sig=None):
    r = json.loads((directory / (split + "-" + n + ".json")).read_text())
    per = torch.load(directory / "per-query" / (split + "-" + n + ".pt"), map_location="cpu", weights_only=True)
    if (
        r["method"] != n
        or r["parameters"] != params(value)
        or r["revision"] != revision()
        or r["external_vocabulary"] != 0
        or r["split"] != split
        or r["classes"] != len(UNSEEN_CLASSES[split])
        or r["checkpoint_sha256"] != CHECKPOINT
        or r["exploratory_unseen"] != (split != "tuberlin")
        or r["freeze_sha256"] != sig
        or tuple(r[k] for k in ("queries", "gallery", "p_k", "map_k")) != PROTOCOL[split]
        or per.shape != (r["queries"], 2)
        or not torch.allclose(per.mean(0), torch.tensor([r["mAP"], r["precision"]]), rtol=0, atol=5e-7)
    ):
        raise ValueError(f"Invalid phase result for {split} {n}")
    return r


def freeze(workspace):
    folder = workspace / "results/phase-sketchy"
    catalog = entries()
    ledger = {}
    for n, value in catalog.items():
        ledger[n] = {}
        for s in SKETCHY_SPLITS:
            r = read(folder, s, n, value)
            ledger[n][s] = {k: r[k] for k in ("mAP", "precision", "parameters")}

    def score(n):
        return tuple(sum(ledger[n][s][k] for s in SKETCHY_SPLITS) / 2 for k in ("precision", "mAP"))

    best_phase = max((n for n, item in grid().items() if not item[0].shuffled), key=score)
    best = max((BASE, best_phase), key=score)
    c, m = catalog[best_phase]
    shuffled = replace(c, shuffled=True)
    held = {BASE: None, best_phase: (c, m), name(c, "direct"): (c, "direct"), name(shuffled, m): (shuffled, m)}
    lock = {
        "revision": revision(),
        "checkpoint_sha256": CHECKPOINT,
        "selection_splits": list(SKETCHY_SPLITS),
        "selection_metric": "equal mean precision, followed by equal mean mAP",
        "tu_split_sha256": SPLIT_HASHES["tuberlin"],
        "tu_classes": 30,
        "sketchy_count": len(catalog),
        "ledger_sha256": digest(ledger),
        "winner_including_ablation": best,
        "phase_winner": best_phase,
        "heldout": {n: None if v is None else {"config": asdict(v[0]), "mode": v[1]} for n, v in held.items()},
        "selected_sketchy": {n: ledger[n] for n in (BASE, best_phase)},
    }
    f = workspace / "results" / LOCK
    if f.exists():
        prev = json.loads(f.read_text())
        unsigned = prev.copy()
        sig = unsigned.pop("freeze_sha256")
        if digest(unsigned) != sig or {k: v for k, v in unsigned.items() if k != "utc"} != lock:
            raise ValueError("Lock changed")
        return prev
    if list((workspace / "results/phase-tuberlin-frozen").glob("tuberlin-*.json")):
        raise ValueError("TU started without lock")
    lock["utc"] = datetime.now(timezone.utc).isoformat()
    lock["freeze_sha256"] = digest(lock)
    tmp = f.with_suffix(".pending")
    tmp.write_text(json.dumps(lock, indent=2) + "\n")
    tmp.replace(f)
    return lock


@torch.inference_mode()
def run(workspace, stage, split):
    if stage == "freeze":
        print(json.dumps(freeze(workspace), indent=2))
        return
    held = stage == "heldout"
    if held:
        f = workspace / "results" / LOCK
        lock = json.loads(f.read_text())
        copy = lock.copy()
        sig = copy.pop("freeze_sha256")
        if (
            digest(copy) != sig
            or lock["revision"] != revision()
            or lock["selection_splits"] != list(SKETCHY_SPLITS)
            or lock["tu_classes"] != 30
            or digest(UNSEEN_CLASSES["tuberlin"]) != lock["tu_split_sha256"]
        ):
            raise ValueError("Unverified TU settings")
        catalog = {
            n: None if v is None else (Config(**v["config"]), v["mode"]) for n, v in lock["heldout"].items()
        }
    else:
        sig = None
        catalog = entries()
    splits = ("tuberlin",) if held else SKETCHY_SPLITS
    if split and split not in splits:
        raise ValueError(split)
    args = runtime(workspace, "tuberlin" if held else "sketchy", batch=64, workers=2, crop_batch=192)
    args.output = workspace / "results" / ("phase-tuberlin-frozen" if held else "phase-sketchy")
    args.score_batch = 128
    args.output.mkdir(parents=True, exist_ok=True)
    qp, qi = merge_inputs(args.root, splits, "sketch")
    pp, pi = merge_inputs(args.root, splits, "photo")
    q, p, _ = load_global_inputs(args, qp, pp, False)
    q_views = phase_bank(args, qp, "sketch")
    p_views = phase_bank(args, pp, "photo")
    iso = projection(args)
    for s in splits:
        if split and s != split:
            continue
        if digest(UNSEEN_CLASSES[s]) != SPLIT_HASHES[s]:
            raise ValueError(s)
        qids, ql = qi[s]
        pids, pl = pi[s]
        if held and (ql.unique().numel(), pl.unique().numel()) != (30, 30):
            raise ValueError("TU requires all 30 classes")
        args.metric_workers = 2 if s == "sketchy_2" else 4
        todo = {}
        for n, value in catalog.items():
            if (args.output / (s + "-" + n + ".json")).exists():
                read(args.output, s, n, value, sig)
            else:
                todo[n] = value
        if not todo:
            continue
        raw, photos = q[qids].cuda(), p[pids].cuda()
        qv, pv = q_views[qids].cuda(), p_views[pids].cuda()
        meta = {
            "backbone": "OpenAI CLIP ViT-B/32",
            "checkpoint_sha256": CHECKPOINT,
            "revision": revision(),
            "gpu": torch.cuda.get_device_name(),
            "torch": str(torch.__version__),
            "study": "phase-appearance-agreement",
            "exploratory_unseen": not held,
            "external_vocabulary": 0,
            "freeze_sha256": sig,
            "classes": len(UNSEEN_CLASSES[s]),
        }
        if BASE in todo:
            visual_rows(
                args,
                s,
                torch.cat((raw[:, None], qv[:, :4].float()), 1),
                photos,
                iso,
                [BASE],
                (ql, pl),
                meta,
                write_result,
            )
            del todo[BASE]
        evaluate(args, s, raw, photos, qv, pv, iso, todo, (ql, pl), meta)
        for n, value in catalog.items():
            read(args.output, s, n, value, sig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--stage", choices=("sketchy", "freeze", "heldout"), required=True)
    parser.add_argument("--split", choices=(*SKETCHY_SPLITS, "tuberlin"))
    a = parser.parse_args()
    run(a.workspace, a.stage, a.split)
