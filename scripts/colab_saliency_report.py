"""Validate saliency-transfer measurements, transfer to TU, and audit tuning by leaving out each class."""

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

from colab_paired import CHECKPOINT, checkout
from colab_saliency import CONTROL_REV, METHOD, REV

REPO = Path(__file__).resolve().parents[1]
PRIOR = {
    "native-clip": "domain",
    "domain-retain0-joint": "domain",
    "struct-both-k320-w010": "structure",
    "visual-graph-k64-g1-r1.5": "visual-graph",
    "route-neighbors100-k128-q75": "gallery-routing",
}


def class_validation(splits, classes, labels, eligible, values):
    names = sorted(eligible)
    per_class = {}
    for s in splits:
        stack = torch.stack([values[s][n] for n in names]).double()
        per_class[s] = {c: stack[:, labels[s] == i].sum(1) for i, c in enumerate(classes[s])}
    totals = {s: sum(per_class[s].values()) for s in splits}
    counts = {s: {c: int((labels[s] == i).sum()) for i, c in enumerate(classes[s])} for s in splits}
    report = {s: {"chosen_by_class": {}, "weighted_sum": torch.zeros(2, dtype=torch.float64)} for s in splits}
    for omitted in sorted(set().union(*(set(classes[s]) for s in splits))):
        remaining = sum(
            (totals[s] - per_class[s].get(omitted, 0)) / (len(labels[s]) - counts[s].get(omitted, 0))
            for s in splits
        )
        pick = max(range(len(names)), key=lambda i: (float(remaining[i, 1]), float(remaining[i, 0])))
        for s in splits:
            if omitted in per_class[s]:
                report[s]["chosen_by_class"][omitted] = names[pick]
                report[s]["weighted_sum"] += per_class[s][omitted][pick]
    for s in splits:
        ap, p = (report[s].pop("weighted_sum") / len(labels[s])).tolist()
        report[s].update(mAP=ap, precision=p)
    return report


@torch.inference_mode()
def run(workspace):
    if not torch.cuda.is_available() or not Path("/content").is_dir():
        raise RuntimeError("Use a Colab GPU")
    torch.set_num_threads(1)
    method = checkout(workspace, REPO, METHOD, REV)
    sys.path.insert(0, str(method))
    from tf_sbir.crossdomain_benchmark import digest
    from tf_sbir.data import SKETCHY_SPLITS, UNSEEN_CLASSES, collect_paths
    from tf_sbir.saliency_benchmark import COARSE, EXPECTED, LOCK, choices, freeze
    from tf_sbir.saliency_transfer import FAMILIES, Setting, grid, report_parameters
    from tf_sbir.transfer_summary import intervals
    from colab_crossdataset_report import write_tu_classes

    folder = workspace / "results"
    out = folder / "saliency-report"
    out.mkdir(exist_ok=True)
    if not (folder / LOCK).is_file():
        raise FileNotFoundError(LOCK)
    lock = freeze(workspace)
    plan = json.loads((folder / "saliency-control-plan.json").read_text())
    if plan["prior_tu_rows_for_this_method"] != 0 or plan["control_revision"] != CONTROL_REV:
        raise ValueError("Incorrect registered region-location control")
    all_splits = (*SKETCHY_SPLITS, "tuberlin")
    roots = {
        s: Path(
            json.loads((workspace / ("tuberlin-data.json" if s == "tuberlin" else "ready.json")).read_text())[
                "root"
            ]
        )
        for s in all_splits
    }
    labels = {s: collect_paths(roots[s], UNSEEN_CLASSES[s], "sketch")[1] for s in all_splits}
    configs = grid() | choices(workspace)
    held = {n: (Setting(**v["config"]), v["mode"]) for n, v in lock["heldout"].items()}
    winners = lock["winners"]
    published = json.loads((REPO / "docs/results/sketchy-tu-summary.json").read_text())["table"]
    rows = {s: {} for s in all_splits}
    values = {s: {} for s in all_splits}
    table = []
    for s in all_splits:
        tu = s == "tuberlin"
        ds = "tuberlin" if tu else "sketchy"
        root = folder / ("saliency-tuberlin-frozen" if tu else "saliency-sketchy")
        requested = held if tu else configs
        expected = {"saliency-tuberlin-frozen": held, "saliency-sketchy": configs}[root.name]
        if {p.stem[len(s) + 1 :] for p in root.glob(s + "-*.json")} != set(expected):
            raise ValueError(f"Invalid fixed inventory for {s}")
        sources = {n: (root, REV, c, m) for n, (c, m) in requested.items()}
        for n, (c, m) in held.items():
            if not tu and n not in sources:
                sources[n] = (folder / "saliency-sketchy-controls", CONTROL_REV, c, m)
        for n in winners.values():
            sources["center-" + n] = (folder / ("saliency-" + ds + "-center"), CONTROL_REV, *held[n])
        for n, (directory, rev, c, mode) in sources.items():
            row = json.loads((directory / (s + "-" + n + ".json")).read_text())
            t = torch.load(
                directory / "per-query" / (s + "-" + n + ".pt"), map_location="cpu", weights_only=True
            )
            expected_params = report_parameters(c, mode, {"all": 7, "core": 2, "full": 1}[c.views])
            if (
                row["method"] != n
                or row["split"] != s
                or row["revision"] != rev
                or row["checkpoint_sha256"] != CHECKPOINT
                or row["parameters"] != expected_params
                or row["external_vocabulary"] != (20931 if c.family == "public" else 0)
                or row["freeze_sha256"] != (lock["freeze_sha256"] if tu else None)
                or row["exploratory_unseen"] == tu
                or tuple(row[k] for k in ("queries", "gallery", "p_k", "map_k")) != EXPECTED[s]
                or row["classes"] != len(UNSEEN_CLASSES[s])
                or t.shape != (len(labels[s]), 2)
                or not torch.allclose(
                    t.mean(0), torch.tensor([row["mAP"], row["precision"]]), rtol=0, atol=5e-7
                )
            ):
                raise ValueError(f"Invalid {s} {n}")
            if n.startswith("center-") and row["region_location"] != "image-center":
                raise ValueError(n)
            rows[s][n] = row
            values[s][n] = t
            table.append(row)
        for n, study in PRIOR.items():
            rows[s][n] = published[n][s]
            priorfolder = folder / (
                "domain-tuberlin-frozen"
                if tu
                else study
                if study not in ("domain", "structure")
                else study + "-sketchy"
            )
            if tu and study == "structure":
                priorfolder = folder / "structure-tuberlin-frozen"
            f = priorfolder / "per-query" / (s + "-" + n + ".pt")
            if f.is_file():
                t = torch.load(f, map_location="cpu", weights_only=True)
                if t.shape != (len(labels[s]), 2) or not torch.allclose(
                    t.mean(0), torch.tensor([rows[s][n]["mAP"], rows[s][n]["precision"]]), rtol=0, atol=5e-7
                ):
                    raise ValueError("Historical scores differ")
                values[s][n] = t
    for n in (*held, *("center-" + n for n in winners.values())):
        if any(rows[s][n]["parameters"] != rows["tuberlin"][n]["parameters"] for s in SKETCHY_SPLITS):
            raise ValueError("Retuned on TU")
    best = max(
        winners.values(),
        key=lambda n: tuple(sum(rows[s][n][k] for s in SKETCHY_SPLITS) for k in ("precision", "mAP")),
    )
    selected = list(
        dict.fromkeys([*winners.values(), *held, *("center-" + n for n in winners.values()), *PRIOR])
    )
    report = {
        "selected_general_method": best,
        "selection_revision": REV,
        "control_revision": CONTROL_REV,
        "freeze": lock,
        "seed": json.loads((folder / COARSE).read_text()),
        "crop_control_plan": plan,
        "bootstrap_draws": 5000,
        "table": {
            n: {s: {k: rows[s][n][k] for k in ("precision", "mAP", "p_k", "map_k")} for s in all_splits}
            for n in selected
        },
        "class_validation": {},
        "paired": {},
        "sketchy_best_per_split": {},
    }
    for family in FAMILIES:
        eligible = [n for n, (c, m) in configs.items() if c.family == family and c.paired]
        report["class_validation"][family] = class_validation(
            SKETCHY_SPLITS, UNSEEN_CLASSES, labels, eligible, values
        )
        report["sketchy_best_per_split"][family] = {
            s: max(eligible, key=lambda n: (rows[s][n]["precision"], rows[s][n]["mAP"])) for s in SKETCHY_SPLITS
        }
    pairs = []
    for family, n in winners.items():
        pairs.extend(
            (n, other)
            for other in (
                "native-clip",
                lock["controls"][family + "_frame"],
                "center-" + n,
                "domain-retain0-joint" if family == "public" else "struct-both-k320-w010",
                "route-neighbors100-k128-q75" if family == "public" else "visual-graph-k64-g1-r1.5",
            )
        )
    pairs.extend(
        ((winners["public"], lock["controls"]["public_shuffle"]), (winners["public"], winners["image"]))
    )
    generator = torch.Generator(device="cuda").manual_seed(341)
    report["bootstrap_seed"] = 341
    for s in all_splits:
        chosen = [(a, b) for a, b in pairs if a != b and a in values[s] and b in values[s]]
        report["paired"][s] = intervals(values[s], labels[s], chosen, generator)
    cname = list(
        dict.fromkeys(
            [*held, *("center-" + n for n in winners.values()), *(n for n in PRIOR if n in values["tuberlin"])]
        )
    )
    report["tu_coverage"] = write_tu_classes(
        out, roots["tuberlin"], labels["tuberlin"], cname, values["tuberlin"]
    )
    report["ledger_check"] = digest(
        {
            n: {s: {k: rows[s][n][k] for k in ("mAP", "precision", "parameters")} for s in SKETCHY_SPLITS}
            for n in configs
        }
    )
    if report["ledger_check"] != lock["ledger_sha256"]:
        raise ValueError("Frozen Sketchy ledger differs")
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / "scores.csv").open("w", newline="") as f:
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
    print("Verified", len(table), "measurements", best, report["tu_coverage"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
