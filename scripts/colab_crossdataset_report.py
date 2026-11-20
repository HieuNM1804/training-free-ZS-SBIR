"""Verify frozen cross-dataset measurements and calculate paired intervals on Google Colab."""

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

from colab_crossdataset import LOCKS, STUDIES, TOGETHER, bind_selections, heldout_done, register_plan
from colab_paired import CHECKPOINT

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tf_sbir.data import SKETCHY_SPLITS, UNSEEN_CLASSES, collect_paths
from tf_sbir.transfer_summary import intervals

TU = "tuberlin"
SPLITS = (*SKETCHY_SPLITS, TU)
PROTOCOL = {"sketchy_1": (14179, 17100, 100, 0), "sketchy_2": (12694, 12552, 200, 200), TU: (2130, 27262, 100, 0)}


def read_scores(directory, split, name, revision, freeze=None):
    base = split + "-" + name
    row = json.loads((directory / (base + ".json")).read_text())
    tensor = torch.load(directory / "per-query" / (base + ".pt"), map_location="cpu", weights_only=True)
    size = tuple(row[k] for k in ("queries", "gallery", "p_k", "map_k"))
    if (
        row["method"] != name
        or row["split"] != split
        or size != PROTOCOL[split]
        or row["revision"] != revision
        or row["checkpoint_sha256"] != CHECKPOINT
        or row["exploratory_unseen"] != (split != TU)
        or row.get("freeze_sha256") != freeze
        or tensor.shape != (size[0], 2)
        or not torch.allclose(tensor.mean(0), torch.tensor([row["mAP"], row["precision"]]), atol=5e-7, rtol=0)
    ):
        raise ValueError(f"Invalid official results for {directory.name} / {base}")
    return row, tensor


def normalize_name(name):
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


@torch.inference_mode()
def run(workspace):
    if not torch.cuda.is_available() or not Path("/content").is_dir():
        raise RuntimeError("Calculate the scientific results on Google Colab")
    torch.set_num_threads(1)
    folder = workspace / "results"
    if not (folder / TOGETHER).is_file():
        raise ValueError("Both Sketchy selections must be frozen before calculating these tables")
    register_plan(folder)
    locks = bind_selections(folder)
    if not all(heldout_done(folder, name, lock) for name, lock in locks.items()):
        raise ValueError("Complete the frozen TU experiments")
    dom, struct = locks["domain"], locks["structure"]
    visual, semantic = (dom["winners"][kind]["name"] for kind in ("general_visual", "general_semantic"))
    base = struct["heldout_controls"][0]
    shape = struct["best_using_structure"]
    visual_old, semantic_old = dom["previous_sketchy_winners"]
    names = list(dict.fromkeys(("native-clip", visual, semantic, base, shape, visual_old, semantic_old)))
    historical = {visual_old: "visual-graph", semantic_old: "gallery-routing"}
    with (REPO / "docs/results/sketchy-colab-l4.csv").open(newline="") as f:
        prior = {row["method"]: row for row in csv.DictReader(f) if row["method"] in historical}
    all_rows, details = {s: {} for s in SPLITS}, {s: {} for s in SPLITS}
    table = []
    for study, lock in locks.items():
        for split in SPLITS:
            tu = split == TU
            directory = folder / (study + ("-tuberlin-frozen" if tu else "-sketchy"))
            files = sorted(directory.glob(split + "-*.json"))
            if not tu and len(files) != {"domain": 36, "structure": 23}[study]:
                raise ValueError(f"Incomplete tuning results in {directory.name}")
            for file in files:
                name = file.stem[len(split) + 1 :]
                row, t = read_scores(
                    directory, split, name, lock["revision"], lock["freeze_sha256"] if tu else None
                )
                details[split][name] = t
                all_rows[split][name] = row
                table.append(row)
    for name, study in historical.items():
        prev = prior[name]
        params = json.loads(prev["parameters"])
        if all_rows[TU][name]["parameters"] != params:
            raise ValueError(f"Historical control changed on TU: {name}")
        for n, split in enumerate(SKETCHY_SPLITS, 1):
            if (folder / study / (split + "-" + name + ".json")).is_file():
                row, t = read_scores(folder / study, split, name, prev["revision"])
                if (
                    abs(100 * row["precision"] - float(prev[f"ext{n}_precision_pct"])) > 1e-5
                    or abs(100 * row["mAP"] - float(prev[f"ext{n}_map_pct"])) > 1e-5
                ):
                    raise ValueError("Historical Sketchy scores disagree with their published archive")
                details[split][name] = t
            else:
                q, p, pk, mk = PROTOCOL[split]
                row = dict(
                    split=split,
                    method=name,
                    queries=q,
                    gallery=p,
                    p_k=pk,
                    map_k=mk,
                    study=study,
                    mAP=float(prev[f"ext{n}_map_pct"]) / 100,
                    precision=float(prev[f"ext{n}_precision_pct"]) / 100,
                    external_vocabulary=int(prev["public_concepts"]),
                    parameters=params,
                    revision=prev["revision"],
                    exploratory_unseen=True,
                    checkpoint_sha256=CHECKPOINT,
                    gpu=prev["gpu"],
                )
            all_rows[split][name] = row
            table.append(row)
    for name in names:
        if any(all_rows[s][name]["parameters"] != all_rows[TU][name]["parameters"] for s in SKETCHY_SPLITS):
            raise ValueError(f"The selected parameter settings changed across datasets for {name}")

    def score(name):
        return tuple(sum(all_rows[s][name][key] for s in SKETCHY_SPLITS) / 2 for key in ("precision", "mAP"))

    primary = max((visual, semantic, struct["best_including_ablation"]), key=score)
    roots = {
        s: Path(json.loads((workspace / ("tuberlin-data.json" if s == TU else "ready.json")).read_text())["root"])
        for s in SPLITS
    }
    labels = {s: collect_paths(roots[s], UNSEEN_CLASSES[s], "sketch")[1] for s in SPLITS}
    report = {
        "selection_splits": list(SKETCHY_SPLITS),
        "selected_general_method": primary,
        "vocabulary_method": semantic,
        "general_visual": visual,
        "matched_four_view_control": base,
        "structure_method": shape,
        "general_candidates_sketchy": {
            n: dict(zip(("precision", "mAP"), score(n)))
            for n in (visual, semantic, struct["best_including_ablation"])
        },
        "locks": {
            n: {"filename": LOCKS[n], "freeze_sha256": v["freeze_sha256"], "revision": STUDIES[n][1]}
            for n, v in locks.items()
        },
        "joint_freeze": json.loads((folder / TOGETHER).read_text()),
        "analysis_plan": json.loads((folder / "crossdataset-analysis-plan.json").read_text()),
        "table": {
            n: {
                s: {k: all_rows[s][n][k] for k in ("precision", "mAP", "p_k", "map_k", "external_vocabulary")}
                for s in SPLITS
            }
            for n in names
        },
        "paired_intervals": {},
        "sketchy_best_per_split": {},
        "bootstrap_replicates": 5000,
        "bootstrap_seed": 51387,
    }
    for s in SKETCHY_SPLITS:
        report["sketchy_best_per_split"][s] = {}
        for family, prefix in (("domain", "domain-"), ("structure", "struct-")):
            eligible = [
                r
                for n, r in all_rows[s].items()
                if n.startswith(prefix) and "shuffled" not in n and "unpaired" not in n
            ]
            r = max(eligible, key=lambda r: (r["precision"], r["mAP"]))
            report["sketchy_best_per_split"][s][family] = {k: r[k] for k in ("method", "precision", "mAP")}
    contrasts = [(n, "native-clip") for n in (visual, semantic, base, shape, visual_old, semantic_old)] + [
        (shape, base),
        (primary, visual_old),
        (primary, semantic_old),
        (semantic, visual),
    ]
    generator = torch.Generator(device="cuda").manual_seed(51387)
    for s in SPLITS:
        pairs = list(contrasts)
        if s != TU:
            pairs.extend(
                [
                    ("domain-default-joint", "domain-neutral-joint"),
                    ("domain-default-joint", "domain-unpaired-joint"),
                    ("domain-default-separate", "domain-neutral-separate"),
                    ("domain-default-separate", "domain-unpaired-separate"),
                    ("struct-both-k192-w030", "struct-shuffled-k192-w030"),
                ]
            )
        available = [(a, b) for a, b in dict.fromkeys(pairs) if a != b and a in details[s] and b in details[s]]
        report["paired_intervals"][s] = intervals(details[s], labels[s], available, generator)
    known = {normalize_name(n) for s in SKETCHY_SPLITS for n in UNSEEN_CLASSES[s]}
    report["tu_subgroups"] = {}
    for shared in (False, True):
        indices = [i for i, c in enumerate(UNSEEN_CLASSES[TU]) if (normalize_name(c) in known) == shared]
        mask = torch.isin(labels[TU], torch.tensor(indices))
        report["tu_subgroups"]["same_name" if shared else "distinct_name"] = {
            "classes": [UNSEEN_CLASSES[TU][i] for i in indices],
            "queries": int(mask.sum()),
            "full_gallery": PROTOCOL[TU][1],
            "methods": {n: dict(zip(("mAP", "precision"), details[TU][n][mask].mean(0).tolist())) for n in names},
        }
    dest = folder / "sketchy-tu-report"
    dest.mkdir(exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    keys = (
        "split",
        "method",
        "study",
        "external_vocabulary",
        "exploratory_unseen",
        "precision",
        "mAP",
        "queries",
        "gallery",
        "p_k",
        "map_k",
        "parameters",
        "revision",
        "freeze_sha256",
        "gpu",
        "checkpoint_sha256",
    )
    with (dest / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in table:
            writer.writerow(
                {k: json.dumps(row[k], sort_keys=True) if k == "parameters" else row.get(k) for k in keys}
            )
    print(primary, "verified", len(table), "rows", dest, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
