"""Evaluate S with the user-selected single prompt for each domain."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import random

import numpy as np
import open_clip
import torch

from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT, runtime
from .encoding import load_frozen_clip
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_boxable_prompt_search import BATCH, PAIRS, SKETCHY, Split, original_rows, posteriors, write_official
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt
from .semantic_disputed_word_risk import SEED, inputs


DEFAULT_PROMPTS = ("a drawing of a {}.", "a photo of the {}.")
FAMILY = "semantic-boxable-drawing-photo"


@torch.inference_mode()
def run():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    workspace = Path("/content/sbir")
    root = workspace / "results" / FAMILY
    vocabulary = json.loads((DOCS / VOCABULARY_STUDY / "freeze.json").read_bytes())
    word_sha = vocabulary.pop("sha256")
    words = vocabulary["subsets"]["boxable"]["words"]
    if len(words) != 601 or digest(vocabulary) != word_sha:
        raise ValueError("Expected the frozen 601 boxable words")
    kd, _ = load_kd(workspace)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("KD-SBIR evaluator has changed")
    pair_idx = PAIRS.index(DEFAULT_PROMPTS)
    spec = {
        "commit": revision(), "sketch_prompt": DEFAULT_PROMPTS[0], "photo_prompt": DEFAULT_PROMPTS[1],
        "vocabulary_lock_sha256": word_sha, "vocabulary_count": len(words), "pair_index": pair_idx,
        "clip_sha": CHECKPOINT, "evaluator_sha": KD_FUNCTION_SHA, "seed": SEED, "batch": BATCH,
        "device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "selection": "User-selected fixed pair from the Sketchy grid",
    }
    freeze = save_lock(root / "freeze.json", spec)
    print("Default prompts locked", DEFAULT_PROMPTS, flush=True)

    cfg = runtime(workspace, "sketchy")
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = {prompt: encode_prompt(model, tokenizer, words, prompt) for prompt in DEFAULT_PROMPTS}
    del model
    workers = min(4, len(os.sched_getaffinity(0)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for dataset, names in (("sketchy", SKETCHY), ("tuberlin", ("tuberlin",))):
            data = inputs(workspace, dataset)
            if data[-1] != vocabulary["full_vocab_sha"]:
                raise ValueError(f"{dataset} uses different vocabulary inputs")
            for name in names:
                split = Split.from_inputs(data, name)
                banks = posteriors(split, text, [DEFAULT_PROMPTS])
                scores = original_rows(split, banks, [pair_idx], kd, pool)
                write_official(root, split, scores, freeze, freeze)


if __name__ == "__main__":
    run()
