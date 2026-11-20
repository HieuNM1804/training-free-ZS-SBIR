"""Verify the phase sweep and compare both newly frozen studies on all three official benchmarks."""

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

from colab_paired import CHECKPOINT, checkout
from colab_region_studies import JOINT, PHASE, PHASE_REV, TU_FOLDERS, combined_lock
from colab_saliency_report import PRIOR, class_validation

REPO = Path(__file__).resolve().parents[1]


def extra_sketchy(workspace, held):
    from tf_sbir.benchmark import merge_inputs
    from tf_sbir.crossdomain_inputs import projection, runtime
    from tf_sbir.data import SKETCHY_SPLITS, UNSEEN_CLASSES
    from tf_sbir.encoding import load_global_inputs
    from tf_sbir.phase_agreement import evaluate
    from tf_sbir.phase_benchmark import entries, read
    from tf_sbir.phase_views import phase_bank

    missing = {n: v for n, v in held.items() if n not in entries()}
    if not missing:
        return
    args = runtime(workspace, "sketchy", batch=64, workers=2, crop_batch=192)
    args.output = workspace / "results/phase-sketchy-checks"
    args.output.mkdir(exist_ok=True)
    args.score_batch = 128
    qp, qi = merge_inputs(args.root, SKETCHY_SPLITS, "sketch")
    pp, pi = merge_inputs(args.root, SKETCHY_SPLITS, "photo")
    q, p, _ = load_global_inputs(args, qp, pp, False)
    qv, pv = phase_bank(args, qp, "sketch"), phase_bank(args, pp, "photo")
    iso = projection(args)
    for s in SKETCHY_SPLITS:
        todo = {n: v for n, v in missing.items() if not (args.output / (s + "-" + n + ".json")).is_file()}
        if todo:
            qids, ql = qi[s]
            pids, pl = pi[s]
            args.metric_workers = 2 if s == "sketchy_2" else 4
            meta = {
                "backbone": "OpenAI CLIP ViT-B/32",
                "checkpoint_sha256": CHECKPOINT,
                "revision": PHASE_REV,
                "gpu": torch.cuda.get_device_name(),
                "torch": str(torch.__version__),
                "study": "phase-matched-control",
                "exploratory_unseen": True,
                "external_vocabulary": 0,
                "freeze_sha256": None,
                "classes": len(UNSEEN_CLASSES[s]),
            }
            evaluate(
                args,
                s,
                q[qids].cuda(),
                p[pids].cuda(),
                qv[qids].cuda(),
                pv[pids].cuda(),
                iso,
                todo,
                (ql, pl),
                meta,
            )
        for n, v in missing.items():
            read(args.output, s, n, v)


@torch.inference_mode()
def run(workspace):
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or not Path("/content").is_dir():
        raise RuntimeError("Run on the Colab GPU")
    folder = workspace / "results"
    out = folder / "region-comparison"
    out.mkdir(exist_ok=True)
    if not (folder / JOINT).is_file():
        raise ValueError("Both new study locks are required before comparing TU")
    joint = combined_lock(folder)
    plan = json.loads((folder / "saliency-phase-comparison-plan.json").read_text())
    if plan["tu_rows"] != {n: 0 for n in TU_FOLDERS} or plan["selection_splits"] != ["sketchy_1", "sketchy_2"]:
        raise ValueError("No predeclared two-Sketchy selection")
    tree = checkout(workspace, REPO, PHASE, PHASE_REV)
    sys.path.insert(0, str(tree))
    from tf_sbir.data import SKETCHY_SPLITS, UNSEEN_CLASSES, collect_paths
    from tf_sbir.phase_agreement import Config
    from tf_sbir.phase_benchmark import BASE, entries, freeze, read
    from tf_sbir.transfer_summary import intervals
    from colab_crossdataset_report import write_tu_classes

    sal = json.loads((folder / "saliency-report/summary.json").read_text())
    phase = freeze(workspace)
    if (
        phase["freeze_sha256"] != joint["locks"]["phase"]
        or sal["freeze"]["freeze_sha256"] != joint["locks"]["saliency"]
    ):
        raise ValueError("Selection hash mismatch")
    held = {n: None if v is None else (Config(**v["config"]), v["mode"]) for n, v in phase["heldout"].items()}
    extra_sketchy(workspace, held)
    all_splits = (*SKETCHY_SPLITS, "tuberlin")
    catalog = entries()
    qroot = {
        s: Path(
            json.loads((workspace / ("tuberlin-data.json" if s == "tuberlin" else "ready.json")).read_text())[
                "root"
            ]
        )
        for s in all_splits
    }
    labels = {s: collect_paths(qroot[s], UNSEEN_CLASSES[s], "sketch")[1] for s in all_splits}
    vals = {s: {} for s in all_splits}
    measured = {s: {} for s in all_splits}
    table = []
    sal_names = list(sal["freeze"]["winners"].values())
    top = phase["phase_winner"]
    for s in all_splits:
        tu = s == "tuberlin"
        root = folder / ("phase-tuberlin-frozen" if tu else "phase-sketchy")
        regular = held if tu else catalog
        if {p.stem[len(s) + 1 :] for p in root.glob(s + "-*.json")} != set(regular):
            raise ValueError(f"Incomplete ledger {s}")
        for n, v in (regular if tu else catalog | held).items():
            d = root if n in regular else folder / "phase-sketchy-checks"
            row = read(d, s, n, v, phase["freeze_sha256"] if tu else None)
            table.append(row)
            measured[s][n] = row
            vals[s][n] = torch.load(
                d / "per-query" / (s + "-" + n + ".pt"), map_location="cpu", weights_only=True
            )
        for n in (*sal_names, *PRIOR):
            measured[s][n] = sal["table"][n][s]
            if n in sal_names:
                d = folder / ("saliency-tuberlin-frozen" if tu else "saliency-sketchy")
            else:
                family = PRIOR[n]
                if tu:
                    family = "structure-tuberlin-frozen" if family == "structure" else "domain-tuberlin-frozen"
                elif family in ("domain", "structure"):
                    family += "-sketchy"
                d = folder / family
            artifact = d / "per-query" / (s + "-" + n + ".pt")
            if artifact.is_file():
                t = torch.load(artifact, map_location="cpu", weights_only=True)
                if t.shape != (len(labels[s]), 2) or not torch.allclose(
                    t.mean(0),
                    torch.tensor([measured[s][n]["mAP"], measured[s][n]["precision"]]),
                    atol=5e-7,
                    rtol=0,
                ):
                    raise ValueError(f"Historical measurements changed for {n}")
                vals[s][n] = t

    def key(n):
        return tuple(sum(measured[s][n][k] for s in SKETCHY_SPLITS) / 2 for k in ("precision", "mAP"))

    chosen = max([*sal_names, phase["winner_including_ablation"]], key=key)
    image = max((sal["freeze"]["winners"]["image"], phase["winner_including_ablation"]), key=key)
    combined = max((chosen, "domain-retain0-joint", "struct-both-k320-w010"), key=key)
    controls = [n for n in held if n != top]
    names = list(dict.fromkeys([chosen, *sal_names, top, BASE, *held, *PRIOR]))
    report = {
        "winner_this_round": chosen,
        "image_winner_this_round": image,
        "best_general_across_rounds_from_sketchy": combined,
        "joint_freeze": joint,
        "plan": plan,
        "phase_freeze": phase,
        "saliency_winners": sal["freeze"]["winners"],
        "selection_data": list(SKETCHY_SPLITS),
        "table": {
            n: {s: {k: measured[s][n][k] for k in ("precision", "mAP", "map_k", "p_k")} for s in all_splits}
            for n in names
        },
        "paired": {},
        "bootstrap_draws": 5000,
        "bootstrap_seed": 571,
        "tu_breadth": {},
    }
    eligible = [n for n, v in catalog.items() if v is None or not v[0].shuffled]
    report["phase_class_validation"] = class_validation(SKETCHY_SPLITS, UNSEEN_CLASSES, labels, eligible, vals)
    contrasts = [
        (top, b)
        for b in [*controls, "native-clip", "struct-both-k320-w010", "visual-graph-k64-g1-r1.5", sal_names[0]]
    ]
    contrasts.extend(
        (chosen, b) for b in ("native-clip", "domain-retain0-joint", "route-neighbors100-k128-q75")
    )
    rng = torch.Generator(device="cuda").manual_seed(571)
    for s in all_splits:
        pairs = [(a, b) for a, b in dict.fromkeys(contrasts) if a != b and a in vals[s] and b in vals[s]]
        report["paired"][s] = intervals(vals[s], labels[s], pairs, rng)
    tu_names = [n for n in names if n in vals["tuberlin"]]
    report["tu_coverage"] = write_tu_classes(
        out, qroot["tuberlin"], labels["tuberlin"], tu_names, vals["tuberlin"]
    )
    for n in tu_names:
        previous = "native-clip"
        if previous in vals["tuberlin"]:
            diff = vals["tuberlin"][n][:, 1] - vals["tuberlin"][previous][:, 1]
            changes = {
                c: float(diff[labels["tuberlin"] == i].mean()) for i, c in enumerate(UNSEEN_CLASSES["tuberlin"])
            }
            report["tu_breadth"][n] = {
                "classes_higher_precision_than_native": sum(v > 0 for v in changes.values()),
                "class_deltas": changes,
            }
    samples = folder / "saliency-resource-samples.jsonl"
    if samples.is_file():
        data = [json.loads(line) for line in samples.read_text().splitlines() if line.strip()]
        if data:
            report["runtime_partial_samples"] = {
                "first_utc": data[0]["utc"],
                "last_utc": data[-1]["utc"],
                "count": len(data),
                "median_gpu_pct": float(torch.tensor([x["gpu_percent"] for x in data]).median()),
                "max_gpu_pct": max(x["gpu_percent"] for x in data),
                "max_vram_mib": max(x["vram_mib"] for x in data),
            }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / "phase-scores.csv").open("w", newline="") as f:
        keys = (
            "split",
            "method",
            "precision",
            "mAP",
            "p_k",
            "map_k",
            "queries",
            "gallery",
            "classes",
            "external_vocabulary",
            "parameters",
            "revision",
            "freeze_sha256",
        )
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in table:
            writer.writerow(
                {k: json.dumps(row[k], sort_keys=True) if k == "parameters" else row[k] for k in keys}
            )
    print(chosen, "verified", len(table), "phase measurements", report["tu_coverage"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    run(p.parse_args().workspace)
