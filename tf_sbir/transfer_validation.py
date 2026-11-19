"""Check transfer of the selected hyperparameters using completed, per-query official metrics."""

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from .benchmark import merge_inputs
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES


def choose(scores):
    """Precision first, mAP second; deterministically retain the first exact tie."""
    return max(range(len(scores)), key=lambda i: (scores[i][1], scores[i][0]))


def numbers(scores):
    return dict(zip(("mAP", "precision"), scores.tolist(), strict=True))


def check_selection(names, sums, counts, classes):
    full = sums.sum(dim=2) / counts.sum(dim=1)[:, None, None]
    common = choose(full.mean(dim=0).tolist())
    selected = []
    for i in range(len(classes)):
        other_sum = sums.sum(dim=2) - sums[:, :, i]
        other_count = counts.sum(dim=1) - counts[:, i]
        training = (other_sum / other_count[:, None, None]).mean(dim=0)
        selected.append(choose(training.tolist()))
    ids = torch.arange(len(classes), device=sums.device)
    heldout_scores = sums[:, selected, ids].sum(dim=1) / counts.sum(dim=1)[:, None]
    report = {"candidates": len(names), "common": names[common], "selection_by_held_out_class": dict(zip(classes, (names[i] for i in selected), strict=True)),
              "selection_counts": dict(Counter(names[i] for i in selected))}
    for i, split in enumerate(SKETCHY_SPLITS):
        other = 1 - i
        unique_to_donor = (counts[other] > 0) & (counts[i] == 0)
        donor = sums[other][:, unique_to_donor].sum(dim=1) / counts[other, unique_to_donor].sum()
        choice = choose(donor.tolist())
        report[split] = {"common": numbers(full[i, common]), "held_out_class": numbers(heldout_scores[i]),
                         "donor_split": SKETCHY_SPLITS[other], "donor_classes": int(unique_to_donor.sum()),
                         "donor_method": names[choice], "donor_choice_on_target": numbers(full[i, choice])}
    return report


def include_candidate(name):
    if "unpaired" in name or "shuffled" in name or name.startswith("ground-cosine-"):
        return False
    return "canvas" not in name and name != "multitransfer-two-views"


def load_query_results(study):
    joined = study / "per-query-metrics.pt"
    if joined.is_file():
        return torch.load(joined, map_location="cpu", weights_only=True)
    separate = study / "per-query"
    results = {split: {p.stem.removeprefix(split + "-"): torch.load(p, map_location="cpu", weights_only=True)
                       for p in separate.glob(f"{split}-*.pt")} for split in SKETCHY_SPLITS}
    if not all(results.values()):
        raise ValueError(f"Missing per-query results in {study}")
    return results


@torch.inference_mode()
def run(root, studies, output):
    classes = sorted(set().union(*(UNSEEN_CLASSES[s] for s in SKETCHY_SPLITS)))
    _, query_index = merge_inputs(root, SKETCHY_SPLITS, "sketch")
    class_ids = []
    for split in SKETCHY_SPLITS:
        mapping = torch.tensor([classes.index(c) for c in UNSEEN_CLASSES[split]], device="cuda")
        class_ids.append(mapping[query_index[split][1].cuda()])
    counts = torch.stack([torch.bincount(labels, minlength=len(classes)) for labels in class_ids])
    results, all_names, all_sums = {}, [], []
    for study in studies:
        rows = load_query_results(study)
        names = sorted(n for n in rows[SKETCHY_SPLITS[0]] if include_candidate(n))
        parts = []
        for split, labels in zip(SKETCHY_SPLITS, class_ids, strict=True):
            scores = torch.stack([rows[split][n] for n in names]).to(device="cuda", dtype=torch.float64)
            if scores.shape[1:] != (len(labels), 2):
                raise ValueError(f"Query metrics and split disagree for {study.name} on {split}")
            parts.append(scores.new_zeros((len(names), len(classes), 2)).index_add_(1, labels, scores))
        summaries = torch.stack(parts)
        results[study.name] = check_selection(names, summaries, counts, classes)
        all_names.extend(names)
        all_sums.append(summaries)
    results["all_candidates"] = check_selection(all_names, torch.cat(all_sums, dim=1), counts, classes)
    results["overlap"] = [c for i, c in enumerate(classes) if (counts[:, i] > 0).all()]
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--studies", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    run(options.root, options.studies, options.output)
