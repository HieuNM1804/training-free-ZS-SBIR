"""Run the method in the current checkout and retain the official metric values for each query."""

from pathlib import Path
import sys
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
    from tf_sbir.encoding import save_tensor

    args = semantic_benchmark.arguments()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    args.per_query_dir = output / "per-query"

    def evaluate(rows, query_labels, gallery_labels, split, description, workers):
        summary, details = evaluate_with_details(rows, query_labels, gallery_labels, split, description, workers)
        name = description.split(" ", 1)[1]
        save_tensor(details, args.per_query_dir / f"{split}-{name}.pt")
        return summary

    with patch.object(semantic_benchmark, "evaluate_rows", evaluate):
        semantic_benchmark.run(args)
    collected = {}
    for split in args.splits:
        collected[split] = {}
        for result in sorted(output.glob(f"{split}-*.json")):
            name = result.stem.removeprefix(split + "-")
            values = torch.load(args.per_query_dir / f"{result.stem}.pt", map_location="cpu", weights_only=True)
            collected[split][name] = values
    save_tensor(collected, output / "per-query-metrics.pt")


if __name__ == "__main__":
    main()
