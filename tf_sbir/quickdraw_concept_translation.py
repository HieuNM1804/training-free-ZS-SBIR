"""Evaluate the locked concept-co-occurrence translation on full QuickDraw."""

import argparse
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch

from .concept_translation_retrieval import Config, Index
from .quickdraw_eval import prepare, score_all, validate_lock
from .shared_concept_likelihood import emission


FAMILY = "concept-translation-retrieval"
METHOD_REVISION = "6abef3985d1f653ac3d616f587e785edf58a5b5b"


@torch.inference_mode()
def run(workspace, metric_workers=10):
    torch.set_float32_matmul_precision("highest")
    lock, freeze_sha = validate_lock(Path(__file__).resolve().parents[1], FAMILY, METHOD_REVISION)
    config = Config(**lock["winner"])
    inputs = prepare(workspace)
    data = SimpleNamespace(
        p=inputs["photos"],
        a=inputs["gallery_concepts"],
        source=inputs["source_words"],
        target=inputs["target_words"],
        scale=inputs["scale"],
        query_emission=None,
        base=None,
    )
    index = Index(data)

    def score(queries):
        data.query_emission = emission(queries, data.source, data.scale)
        data.base = data.query_emission @ data.a.T
        index.last_key = None
        index.intermediate = None
        index.steps = 0
        return index.score(config)

    return score_all(workspace, FAMILY, METHOD_REVISION, freeze_sha, lock["winner"], inputs,
                     score, Path(inspect.getsourcefile(Index)), metric_workers=metric_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--metric-workers", type=int, default=10)
    options = parser.parse_args()
    run(options.workspace, options.metric_workers)
