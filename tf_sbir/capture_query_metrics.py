"""Run the method in the current checkout and retain the official metric values for each query."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch


def evaluate_with_details(rows, query_labels, gallery_labels, split, description, workers):
    from tf_sbir import metrics

    batches = metrics.metric_batches
    values = []

    def record(*params, **options):
        for batch in batches(*params, **options):
            values.append(batch)
            yield batch

    with patch.object(metrics, "metric_batches", record):
        summary = metrics.evaluate_rows(rows, query_labels, gallery_labels, split, description, workers)
    return summary, torch.cat(values)


def main():
    # Running this file by path allows the same collector to be used on other method branches.
    sys.path.insert(0, str(Path.cwd()))
    from tf_sbir import semantic_benchmark

    args = semantic_benchmark.arguments()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    collected = {}
    def evaluate(rows, query_labels, gallery_labels, split, description, workers):
        summary, details = evaluate_with_details(rows, query_labels, gallery_labels, split, description, workers)
        name = description.split(" ", 1)[1]
        collected.setdefault(split, {})[name] = details
        return summary

    with TemporaryDirectory(dir=output) as work, patch.object(semantic_benchmark, "evaluate_rows", evaluate):
        args.output = Path(work)
        semantic_benchmark.run(args)
        for file in args.output.glob("sketchy*.json"):
            existing = output / file.name
            if existing.is_file():
                before = json.loads(existing.read_text())
                after = json.loads(file.read_text())
                if any(abs(before[key] - after[key]) > 1e-5 for key in ("mAP", "precision")):
                    raise ValueError(f"Saved evaluation differs for {file.name}")
            else:
                file.replace(existing)
        saved = args.output / "per-query-metrics.pt"
        torch.save(collected, saved)
        saved.replace(output / saved.name)


if __name__ == "__main__":
    main()
