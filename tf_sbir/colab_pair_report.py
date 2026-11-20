"""Report precision-first full-gallery Sketchy comparisons from a single Colab runtime."""

import argparse
import csv
import json
from pathlib import Path

import torch

from .data import SKETCHY_SPLITS, UNSEEN_CLASSES, collect_paths
from .transfer_summary import intervals
from .transfer_validation import include_candidate, load_query_results, run as validate_classes


STUDIES = {"visual-graph": (16, 0, "c268df56c1eda43e0ff288abcb9ecb46e5189072"),
           "gallery-balance": (28, 20931, "7f411f8f9d56479b45054d105c5ac184438574ae")}
CHECKPOINT = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
VISUAL = "visual-graph-k64-g1-r1"
REFERENCE = "balance-index2-graph2-text35"
SHARP = "balance-index2-graph2-sharp"


@torch.inference_mode()
def report(workspace):
    torch.set_num_threads(1)
    root = Path(json.loads((workspace / "ready.json").read_text())["root"])
    results = workspace / "results"
    output = results / "precision-report"
    output.mkdir(exist_ok=True)
    values = {s: {} for s in SKETCHY_SPLITS}
    summary = {"selection": "Exploratory unseen; equal mean split precision, then equal mean split mAP.",
               "families": {}, "bootstrap": {"samples": 5000, "seed": 907}}
    table = []
    gpu = torch.cuda.get_device_name()
    for study, (expected, vocab, revision) in STUDIES.items():
        folder = results / study
        per_query = load_query_results(folder)
        names = sorted(per_query[SKETCHY_SPLITS[0]])
        if len(names) != expected or set(names) != set(per_query[SKETCHY_SPLITS[1]]):
            raise ValueError(f"Incomplete runs in {study}")
        group = []
        for name in names:
            row = {"study": study, "method": name, "public_concepts": vocab, "eligible": include_candidate(name),
                   "revision": revision, "gpu": gpu}
            for i, split in enumerate(SKETCHY_SPLITS, 1):
                j = json.loads((folder / f"{split}-{name}.json").read_text())
                details = per_query[split][name]
                protocol = (14179, 17100, 100, 0) if i == 1 else (12694, 12552, 200, 200)
                if ((j["queries"], j["gallery"], j["p_k"], j["map_k"]) != protocol or
                        not j["exploratory_unseen"] or j["external_vocabulary"] != vocab or j["revision"] != revision or
                        j["gpu"] != gpu or j["checkpoint_sha256"] != CHECKPOINT or j["backbone"] != "OpenAI CLIP ViT-B/32" or
                        j["split"] != split or j["method"] != name or details.shape != (protocol[0], 2) or
                        not torch.allclose(details.mean(0), torch.tensor([j["mAP"], j["precision"]]), atol=5e-7, rtol=0)):
                    raise ValueError(f"Invalid measurements for {split} {name}")
                row[f"ext{i}_precision_pct"] = 100 * j["precision"]
                row[f"ext{i}_map_pct"] = 100 * j["mAP"]
                values[split][name] = details
                if i == 1:
                    row["parameters"] = json.dumps(j["parameters"], sort_keys=True)
                elif json.loads(row["parameters"]) != j["parameters"]:
                    raise ValueError(f"Parameters differ across splits for {name}")
            row["mean_precision_pct"] = (row["ext1_precision_pct"] + row["ext2_precision_pct"]) / 2
            row["mean_map_pct"] = (row["ext1_map_pct"] + row["ext2_map_pct"]) / 2
            group.append(row)
        eligible = [r for r in group if r["eligible"]]
        summary["families"][study] = {
            "tested": len(group), "eligible": len(eligible),
            "common": max(eligible, key=lambda r: (r["mean_precision_pct"], r["mean_map_pct"])),
            "per_split": {s: max(eligible, key=lambda r: (r[f"ext{i}_precision_pct"], r[f"ext{i}_map_pct"]))
                          for i, s in enumerate(SKETCHY_SPLITS, 1)}}
        table.extend(group)
    validate_classes(root, [results / study for study in STUDIES], output / "class-validation.json")
    vocab = summary["families"]["gallery-balance"]["common"]["method"]
    vision = summary["families"]["visual-graph"]["common"]["method"]
    summary["comparisons"] = {}
    summary["class_changes_vs_equal_gallery_weights"] = {}
    rng = torch.Generator(device="cuda").manual_seed(907)
    for split in SKETCHY_SPLITS:
        _, labels = collect_paths(root, UNSEEN_CLASSES[split], "sketch")
        best = summary["families"]["gallery-balance"]["per_split"][split]["method"]
        control = lambda method: "balance-index2-graph2-" + method.rsplit("-", 1)[-1]
        comparisons = [(vocab, REFERENCE), (vocab, control(vocab)), (best, control(best)), (vocab, vision),
                       (REFERENCE, VISUAL), (SHARP, REFERENCE), (vision, VISUAL),
                       (VISUAL, "visual-graph-full"), (VISUAL, "visual-graph-cosine-eight"),
                       ("visual-graph-full", "visual-graph-cosine-full"), (VISUAL, "visual-graph-k64-g0-r1"),
                       (VISUAL, "visual-graph-shuffled-errors"),
                       ("balance-index0-graph2-text35", REFERENCE), ("balance-index2-graph0-text35", REFERENCE),
                       ("balance-index0-graph0-text35", REFERENCE)]
        pairs = list(dict.fromkeys((a, b) for a, b in comparisons if a != b))
        summary["comparisons"][split] = intervals(values[split], labels, pairs, rng)
        difference = values[split][vocab] - values[split][REFERENCE]
        summary["class_changes_vs_equal_gallery_weights"][split] = {
            c: dict(zip(("delta_map", "delta_precision"), difference[labels.eq(i)].mean(0).tolist(), strict=True))
            for i, c in enumerate(UNSEEN_CLASSES[split])}
    summary["checkpoint_sha256"] = CHECKPOINT
    summary["gpu"] = gpu
    with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(sorted(table, key=lambda r: (-r["mean_precision_pct"], -r["mean_map_pct"])))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({study: f["common"]["method"] for study, f in summary["families"].items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    report(parser.parse_args().workspace)
