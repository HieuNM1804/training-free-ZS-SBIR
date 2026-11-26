"""Full unseen KD-SBIR replay for candidate-constrained semantic anchor alignment."""

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
from .forced_pair_alignment import AnchorGeometry
from .gallery_convex_support import SEED
from .gallery_convex_support_audit import matched_counts, newcomer_hits, prefix, summarize
from .joint_gallery_witnesses import load_reference
from .locked_geometry_words import score_rows
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import BATCH, SKETCHY, Split, mean_percent, posteriors
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt
from .semantic_boxable_sparse import image_inputs


FAMILY = "forced-pair-alignment"
ARMS = ("profile", "fixed_map", "bilinear", "separable", "shuffled_spectrum")
CONTROLS = ("separable", "fixed_map", "shuffled_spectrum")


def adjacent_pairs(ordered, gallery, truth, scores):
    """Pair nearby separable scores among Dot's top-K before inspecting relevance."""
    separable = scores["separable"].gather(1, ordered)
    order = separable.argsort(dim=1, descending=True, stable=True)
    candidates = ordered.gather(1, order).reshape(len(ordered), -1, 2)
    first, second = candidates.unbind(2)
    first_hit = gallery[first] == truth[:, None]
    second_hit = gallery[second] == truth[:, None]
    one_hit = first_hit ^ second_hit
    prof_margin = scores["profile"].gather(1, first) - scores["profile"].gather(1, second)
    base_margin = scores["separable"].gather(1, first) - scores["separable"].gather(1, second)
    # A strict reversal is interpretable even when implementations break exact ties differently.
    changed = one_hit & (base_margin > 0) & (prof_margin < 0)
    return torch.stack([x.sum(1) for x in (one_hit, changed, changed & second_hit, changed & first_hit)], 1).cpu()


def pair_summary(pairs, labels):
    eligible, changed, improved, damaged = pairs.T
    used = changed > 0
    return {
        "different_relevance": int(eligible.sum()), "strict_reversals": int(changed.sum()),
        "reversal_correct": int(improved.sum()), "reversal_incorrect": int(damaged.sum()),
        "macro_accuracy_delta_on_disagreements": class_interval(
            100 * (improved[used] - damaged[used]).double() / changed[used], labels[used], seed=SEED,
        ),
    }


def evaluate(root, split, text, geometry, kd, pool, lock):
    target = root / f"{split.name}.json"
    artifact = target.with_suffix(".pt")
    if target.is_file():
        report = json.loads(target.read_bytes())
        if report["protocol"] != lock or report["per_query_sha256"] != checksum(artifact):
            raise ValueError("Inconsistent results for " + split.name)
        print(split.name, "verified prior result", flush=True)
        return
    archive, _ = load_reference(split)
    reference = archive["dot"]
    queries, photos = posteriors(split, text, [DEFAULT_PROMPTS])
    q, p = queries[DEFAULT_PROMPTS[0]], photos[DEFAULT_PROMPTS[1]]
    x, y = geometry.project(q, p.T.contiguous())
    numerical = geometry.check(x[:4], y[:4])
    replay = PrefixReplay(kd, split.photo_labels, split.name)
    map_k, precision_k = replay.cutoffs
    gallery, truth = split.photo_labels.cuda(), split.sketch_labels.cuda()
    measured = {arm: [] for arm in ARMS}
    matches = {control: [] for control in CONTROLS}
    close_pairs = []
    seconds = 0.
    print(split.name, "scoring", len(x), "queries against", len(y), "photos", flush=True)
    for start in range(0, len(q), BATCH):
        sl = slice(start, min(start + BATCH, len(q)))
        dot = q[sl] @ p
        base_order = prefix(dot, map_k, precision_k)
        base_mask = torch.zeros_like(dot, dtype=torch.bool).scatter_(1, base_order, True)
        _, base_hits = newcomer_hits(base_order, base_mask, gallery, truth[sl], reference[sl, 1])
        if not torch.equal(base_hits[:, 0].cpu(), reference[sl, 3].bool()):
            raise ValueError("Dot top-1 and archive disagree")
        torch.cuda.synchronize()
        tick = time.perf_counter()
        scores = geometry.scores(x[sl], y)
        torch.cuda.synchronize()
        seconds += time.perf_counter() - tick
        if set(scores) != set(ARMS):
            raise ValueError("Unexpected solver outputs")
        ranked = {}
        for arm, score in scores.items():
            if not torch.isfinite(score).all() or score.shape != dot.shape:
                raise ValueError("Non-finite score or invalid score shape in " + arm)
            rows = score_rows(score, split.sketch_labels[sl], gallery, split.name, replay, pool)
            measured[arm].append(rows)
            if arm == "profile" or arm in CONTROLS:
                ranked[arm] = newcomer_hits(
                    prefix(score, map_k, precision_k), base_mask, gallery, truth[sl], rows[:, 1],
                )
        for control in CONTROLS:
            matches[control].append(matched_counts(ranked["profile"], ranked[control]))
        close_pairs.append(adjacent_pairs(base_order, gallery, truth[sl], scores))
        del scores, score
        if start == 0 or sl.stop == len(q) or (start // BATCH + 1) % 10 == 0:
            print(split.name, sl.stop, "/", len(q), flush=True)

    rows = {arm: torch.cat(parts).double() for arm, parts in measured.items()}
    rows["dot"] = reference
    matched = {arm: torch.cat(parts) for arm, parts in matches.items()}
    pairs = torch.cat(close_pairs)
    save_tensor({
        "qi": split.query_ids, "pi": split.gallery_ids, "labels": split.sketch_labels,
        "rows": rows, "matched": matched, "adjacent_pairs": pairs,
    }, artifact)
    report = {
        "protocol": lock, "queries": len(x), "gallery": len(y), "classes": len(split.sketch_labels.unique()),
        "cutoffs": list(replay.cutoffs), "numerical_check": numerical,
        "metrics": {arm: mean_percent(value[:, :2]) for arm, value in rows.items()},
        "profile_minus_control_macro_ap": {
            arm: class_interval(100 * (rows["profile"][:, 0] - value[:, 0]), split.sketch_labels, seed=SEED)
            for arm, value in rows.items() if arm != "profile"
        },
        "matched_promotions": {arm: summarize(counts, split.sketch_labels) for arm, counts in matched.items()},
        "adjacent_pair_reversals": pair_summary(pairs, split.sketch_labels),
        "top1_fixed_damaged_vs_dot": {
            arm: [int((value[:, 3] > reference[:, 3]).sum()), int((value[:, 3] < reference[:, 3]).sum())]
            for arm, value in rows.items() if arm != "dot"
        },
        "scoring_seconds_all_arms": seconds, "per_query_sha256": checksum(artifact),
    }
    atomic_json(target, report)
    print(split.name, "verified", report["metrics"], flush=True)


@torch.inference_mode()
def run():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    workspace = Path("/content/sbir")
    root = workspace / "results" / FAMILY
    config = json.loads((DOCS / VOCABULARY_STUDY / "freeze.json").read_bytes())
    vocab_sha = config.pop("sha256")
    words = config["subsets"]["boxable"]["words"]
    if len(words) != 601 or digest(config) != vocab_sha:
        raise ValueError("Unexpected 601-word vocabulary")
    kd, _ = load_kd(workspace)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("KD-SBIR evaluator source changed")
    workers = min(4, len(os.sched_getaffinity(0)))
    cfg = runtime(workspace, "sketchy", batch=BATCH, workers=workers)
    lock = save_lock(root / "freeze.json", {
        "commit": revision(), "clip_sha256": CHECKPOINT, "evaluator_sha256": KD_FUNCTION_SHA,
        "vocabulary_sha256": vocab_sha, "prompts": list(DEFAULT_PROMPTS), "seed": SEED,
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "views": ["original", "horizontal_flip"], "arms": list(ARMS), "baseline": "dot",
        "selection": "fixed before scoring either Sketchy split or TU-Berlin",
        "primary": "profile versus separable, fixed map and shuffled spectrum, matched Dot newcomers",
        "pairs": "within Dot precision prefix, adjacent by separable score, strict reversals only",
    })
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = {prompt: encode_prompt(model, tokenizer, words, prompt) for prompt in DEFAULT_PROMPTS}
    geometry = AnchorGeometry(*(text[prompt] for prompt in DEFAULT_PROMPTS))
    data = image_inputs(cfg, SKETCHY, model)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name in SKETCHY:
            evaluate(root, Split.from_inputs(data, name), text, geometry, kd, pool, lock)
        del data
        cfg = runtime(workspace, "tuberlin", batch=BATCH, workers=workers)
        split = Split.from_inputs(image_inputs(cfg, ("tuberlin",), model), "tuberlin")
        evaluate(root, split, text, geometry, kd, pool, lock)


if __name__ == "__main__":
    run()
