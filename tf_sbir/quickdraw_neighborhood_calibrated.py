"""Evaluate the locked neighborhood-calibrated likelihood on full QuickDraw."""

import argparse
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch

from .neighborhood_calibrated_likelihood import Config, Index
from .quickdraw_eval import prepare, score_all, validate_lock
from .shared_concept_likelihood import emission, posterior


FAMILY = "neighborhood-calibrated-likelihood"
METHOD_REVISION = "a8a73530a993133cb4f8058bfe61e7f70cb0f68f"


@torch.inference_mode()
def run(workspace, metric_workers=10):
    torch.set_float32_matmul_precision("highest")
    lock, freeze_sha = validate_lock(Path(__file__).resolve().parents[1], FAMILY, METHOD_REVISION)
    config = Config(**lock["winner"])
    inputs = prepare(workspace)
    data = SimpleNamespace(p=inputs["photos"], a=inputs["gallery_concepts"], query_emission=None)
    index = Index(data)

    # The locked score operator uses the same posterior and gallery pooling for every query.
    prior = index.priors[config.estimator]
    predictive = index.pool(config, posterior(data.a, prior))

    def score(queries):
        data.query_emission = emission(queries, inputs["source_words"], inputs["scale"])
        return data.query_emission @ predictive.T

    return score_all(workspace, FAMILY, METHOD_REVISION, freeze_sha, lock["winner"], inputs,
                     score, Path(inspect.getsourcefile(Index)), metric_workers=metric_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--metric-workers", type=int, default=10)
    options = parser.parse_args()
    run(options.workspace, options.metric_workers)
