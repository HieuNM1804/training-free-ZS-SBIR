"""Evaluate replacing one gallery self-pair with a proposed cross-domain match."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import open_clip
import torch

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT, runtime
from .depiction_path_diagnostic import class_interval
from .encoding import load_frozen_clip, save_tensor
from .gallery_convex_support import SEED
from .gallery_convex_support_audit import matched_counts, newcomer_hits, prefix, summarize
from .gallery_substitution_alignment import GallerySubstitution
from .joint_gallery_witnesses import load_reference
from .locked_geometry_words import score_rows
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import BATCH, SKETCHY, Split, mean_percent, posteriors
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt
from .semantic_boxable_sparse import image_inputs
from .semantic_disputed_word_risk import word_bank


FAMILY = "gallery-substitution-alignment"
ARMS = ("substitution", "uncalibrated", "fixed_map", "fixed_substitution", "shuffled_reference", "hellinger")
MATCHED = ("fixed_substitution", "uncalibrated", "shuffled_reference")


def evaluate(root, split, text, kd, pool, lock):
    report_path = root / f"{split.name}.json"
    artifact = report_path.with_suffix(".pt")
    if report_path.is_file():
        old = json.loads(report_path.read_bytes())
        if old["protocol"] != lock or old["per_query_sha256"] != checksum(artifact):
            raise ValueError("Previous study result failed verification")
        print(split.name, "verified existing report", flush=True)
        return
    archive, source = load_reference(split)
    dot_rows = archive["dot"]
    query_bank, photos = posteriors(split, text, [DEFAULT_PROMPTS])
    q, p = query_bank[DEFAULT_PROMPTS[0]], photos[DEFAULT_PROMPTS[1]]
    torch.cuda.synchronize()
    started = time.perf_counter()
    source_photos = word_bank(split.photo, text[DEFAULT_PROMPTS[0]], split.scale).T.contiguous()
    model = GallerySubstitution(source_photos, p.T.contiguous(), seed=SEED)
    x = model.project_queries(q)
    torch.cuda.synchronize()
    indexing_seconds = time.perf_counter() - started
    numerical = model.check(x[:2])
    replay = PrefixReplay(kd, split.photo_labels, split.name)
    truth, gallery = split.sketch_labels.cuda(), split.photo_labels.cuda()
    rows = {arm: [] for arm in ARMS}
    matched = {arm: [] for arm in (*MATCHED, "dot_displaced")}
    model_seconds = 0.
    print(split.name, "scoring", len(q), "queries against", p.size(1), "photos", flush=True)
    for start in range(0, len(q), BATCH):
        sl = slice(start, min(start + BATCH, len(q)))
        dot = q[sl] @ p
        baseline = prefix(dot, *replay.cutoffs)
        in_baseline = torch.zeros_like(dot, dtype=torch.bool).scatter_(1, baseline, True)
        _, baseline_hits = newcomer_hits(baseline, in_baseline, gallery, truth[sl], dot_rows[sl, 1])
        if not torch.equal(baseline_hits[:, 0].cpu(), dot_rows[sl, 3].bool()):
            raise ValueError("Dot top-1 differs from the verified run")
        torch.cuda.synchronize()
        started = time.perf_counter()
        scores = model.scores(x[sl])
        torch.cuda.synchronize()
        model_seconds += time.perf_counter() - started
        if set(scores) != set(ARMS):
            raise ValueError("Unexpected substitution arms")
        entrants = {}
        for name, score in scores.items():
            if score.shape != dot.shape or not torch.isfinite(score).all():
                raise ValueError("Invalid score in " + name)
            observed = score_rows(score, split.sketch_labels[sl], gallery, split.name, replay, pool)
            rows[name].append(observed)
            if name == "substitution" or name in MATCHED:
                order = prefix(score, *replay.cutoffs)
                entrants[name] = newcomer_hits(order, in_baseline, gallery, truth[sl], observed[:, 1])
                if name == "substitution":
                    in_method = torch.zeros_like(in_baseline).scatter_(1, order, True)
                    displaced = ~in_method.gather(1, baseline)
                    if not torch.equal(entrants[name][0].sum(1), displaced.sum(1)):
                        raise ValueError("Different sizes for entrants and displaced Dot photos")
                    matched["dot_displaced"].append(
                        matched_counts(entrants[name], (displaced, baseline_hits))
                    )
        for control in MATCHED:
            matched[control].append(matched_counts(entrants["substitution"], entrants[control]))
        del scores, score
        if start == 0 or sl.stop == len(q) or (start // BATCH + 1) % 10 == 0:
            print(split.name, sl.stop, "/", len(q), flush=True)
    rows = {name: torch.cat(parts).double() for name, parts in rows.items()}
    rows["dot"] = dot_rows
    matched = {name: torch.cat(parts) for name, parts in matched.items()}
    save_tensor({"qi": split.query_ids, "pi": split.gallery_ids, "labels": split.sketch_labels,
                 "rows": rows, "matched": matched}, artifact)
    primary = rows["substitution"]
    report = {
        "protocol": lock, "source_dot_protocol": source["protocol"],
        "source_per_query_sha256": source["per_query_sha256"],
        "queries": len(q), "gallery": p.size(1), "classes": len(truth.unique()), "cutoffs": list(replay.cutoffs),
        "metrics": {arm: mean_percent(result[:, :2]) for arm, result in rows.items()},
        "substitution_minus_control_macro_ap": {
            arm: class_interval(100 * (primary[:, 0] - result[:, 0]), split.sketch_labels, seed=SEED)
            for arm, result in rows.items() if arm != "substitution"
        },
        "matched_promotions": {arm: summarize(result, split.sketch_labels) for arm, result in matched.items()},
        "top1_fixed_damaged_vs_dot": {
            arm: [int((result[:, 3] > dot_rows[:, 3]).sum()), int((result[:, 3] < dot_rows[:, 3]).sum())]
            for arm, result in rows.items() if arm != "dot"
        },
        "numerical_check": numerical, "index_and_projection_seconds": indexing_seconds,
        "scoring_seconds_all_arms": model_seconds, "per_query_sha256": checksum(artifact),
    }
    atomic_json(report_path, report)
    print(split.name, "verified", report["metrics"], flush=True)


@torch.inference_mode()
def run():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    workspace = Path("/content/sbir")
    root = workspace / "results" / FAMILY
    config = json.loads((DOCS / VOCABULARY_STUDY / "freeze.json").read_bytes())
    vocab_sha = config.pop("sha256")
    words = config["subsets"]["boxable"]["words"]
    if len(words) != 601 or digest(config) != vocab_sha:
        raise ValueError("Vocabulary checksum mismatch")
    kd, _ = load_kd(workspace)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("KD-SBIR evaluator checksum mismatch")
    workers = min(4, len(os.sched_getaffinity(0)))
    cfg = runtime(workspace, "sketchy", batch=BATCH, workers=workers)
    lock = save_lock(root / "freeze.json", {
        "commit": revision(), "clip_sha256": CHECKPOINT, "evaluator_sha256": KD_FUNCTION_SHA,
        "vocabulary_sha256": vocab_sha, "prompts": list(DEFAULT_PROMPTS), "seed": SEED,
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "views": ["original", "horizontal_flip"], "arms": list(ARMS), "baseline": "dot",
        "gallery_reference": "two text heads applied to each same unlabeled RGB photo",
        "selection": "fixed before scoring either Sketchy split or TU-Berlin",
        "primary_controls": list(MATCHED),
        "baseline_newcomers": "compare new photos to the equal number of Dot photos they displaced",
    })
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = {prompt: encode_prompt(model, tokenizer, words, prompt) for prompt in DEFAULT_PROMPTS}
    data = image_inputs(cfg, SKETCHY, model)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name in SKETCHY:
            evaluate(root, Split.from_inputs(data, name), text, kd, pool, lock)
        del data
        cfg = runtime(workspace, "tuberlin", batch=BATCH, workers=workers)
        split = Split.from_inputs(image_inputs(cfg, ("tuberlin",), model), "tuberlin")
        evaluate(root, split, text, kd, pool, lock)


if __name__ == "__main__":
    run()
