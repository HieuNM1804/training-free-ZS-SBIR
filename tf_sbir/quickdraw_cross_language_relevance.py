"""Evaluate the locked per-sketch relevance model on full QuickDraw."""

import argparse
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch

from .cross_language_relevance import Config, Index
from .quickdraw_eval import prepare, score_all, validate_lock
from .shared_concept_likelihood import emission


FAMILY = "cross-language-relevance"
METHOD_REVISION = "dbedfb43edb14e6d4f602685cfa2c50073ab5e03"


@torch.inference_mode()
def run(workspace, metric_workers=10):
    torch.set_float32_matmul_precision("highest")
    lock, freeze_sha = validate_lock(Path(__file__).resolve().parents[1], FAMILY, METHOD_REVISION)
    config = Config(**lock["winner"])
    inputs = prepare(workspace)
    data = SimpleNamespace(
        q=inputs["queries"][:1],
        p=inputs["photos"],
        a=inputs["gallery_concepts"],
        query_emission=emission(inputs["queries"][:1], inputs["source_words"], inputs["scale"]),
        base=None,
        source=inputs["source_words"],
        target=inputs["target_words"],
        scale=inputs["scale"],
    )
    data.base = data.query_emission @ data.a.T
    index = Index(data)
    index.prepare(config.prior)

    def score(queries):
        data.q = queries
        data.query_emission = emission(queries, data.source, data.scale)
        data.base = data.query_emission @ data.a.T
        index.initial = data.base if config.prior == "uniform" else data.query_emission @ index.language.T
        index.estimate = None
        index.last_feedback = None
        return index.score(config)

    return score_all(workspace, FAMILY, METHOD_REVISION, freeze_sha, lock["winner"], inputs,
                     score, Path(inspect.getsourcefile(Index)), metric_workers=metric_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--metric-workers", type=int, default=10)
    options = parser.parse_args()
    run(options.workspace, options.metric_workers)
