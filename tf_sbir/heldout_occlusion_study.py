"""Full independent-query KD-SBIR evaluation of held-out CLIP region interventions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import open_clip
import torch

from .benchmark import merge_inputs
from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import revision
from .crossdomain_inputs import CHECKPOINT, runtime
from .depiction_path_diagnostic import class_interval
from .encoding import load_frozen_clip, save_tensor
from .gallery_convex_support import SEED
from .gallery_convex_support_audit import matched_counts, newcomer_hits, prefix, summarize
from .heldout_occlusion_correspondence import ARMS, Correspondences, additive_predictions, enumerate_pair, fixture, scores
from .joint_gallery_witnesses import load_reference
from .locked_geometry_words import score_rows
from .part_occlusion_observations import FAMILY, directory, public_words, read_cache, seed_job
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import SKETCHY, Split, mean_percent, posteriors
from .semantic_boxable_single_prompt import encode_prompt
from .semantic_boxable_sparse import image_inputs


QUERY_BATCH = 128
CONTROLS = ARMS[1:]


def sources_for_sketches(split, sketch_paths, photo_paths):
    """Only for post-ranking evaluation of verified source photos in Sketchy."""
    keys = [(Path(photo_paths[j]).parent.name, Path(photo_paths[j]).stem) for j in split.gallery_ids.tolist()]
    lookup = {key: idx for idx, key in enumerate(keys)}
    if len(lookup) != len(keys):
        raise ValueError("Ambiguous gallery IDs")
    found = []
    for j in split.query_ids.tolist():
        path = Path(sketch_paths[j])
        found.append(lookup.get((path.parent.name, path.stem.rsplit("-", 1)[0]), -1))
    ids = torch.tensor(found)
    ok = ids >= 0
    if not torch.equal(split.sketch_labels[ok], split.photo_labels[ids[ok]]):
        raise ValueError("Reference source has the wrong class")
    return ids


def source_rows(ids, observed, gallery, truth):
    valid = ids >= 0
    source = ids[valid].cuda()[:, None]
    labels = gallery[None, :] == truth[valid.cuda(), None]
    reference = observed["uniform_correspondence"][valid.cuda()]
    distance = (reference - reference.gather(1, source)).abs()
    candidate = distance.masked_fill(~labels, torch.inf).scatter_(1, source, torch.inf).argmin(1, keepdim=True)
    indices = torch.arange(len(gallery), device=gallery.device)[None, :]
    report = {}
    for arm, raw in observed.items():
        row = raw[valid.cuda()]
        own = row.gather(1, source)
        rank = 1 + (labels & ((row > own) | ((row == own) & (indices < source)))).sum(1)
        alternative = row.gather(1, candidate)
        wins = (own > alternative).double() + .5 * (own == alternative).double()
        report[arm] = torch.stack((rank.double(), wins[:, 0]), 1).cpu()
    return report


def conditional_pairs(baseline, observed, gallery, labels):
    # The pairs and ordering are formed before reading relevance. Restrict to
    # adjacent control scores to compare among candidates of similar difficulty.
    null = observed["uniform_correspondence"].gather(1, baseline)
    pairs = baseline.gather(1, null.argsort(dim=1, descending=True, stable=True)).reshape(len(labels), -1, 2)
    first, second = pairs.unbind(2)
    score, null = observed["heldout"], observed["uniform_correspondence"]
    different = (gallery[first] == labels[:, None]) ^ (gallery[second] == labels[:, None])
    switched = different & (null.gather(1, first) > null.gather(1, second)) & (
        score.gather(1, first) < score.gather(1, second)
    )
    return torch.stack((different.sum(1), switched.sum(1),
                        (switched & (gallery[second] == labels[:, None])).sum(1)), 1).cpu()


def evaluate(root, split, text, obs_q, obs_p, kd, pool, maps, signature, native_ids):
    target = root / f"{split.name}.json"
    artifact = target.with_suffix(".pt")
    if target.is_file():
        report = json.loads(target.read_bytes())
        if report["protocol"] != signature or report["per_query_sha256"] != checksum(artifact):
            raise ValueError("Invalid saved run")
        return
    archive, _ = load_reference(split)
    reference = archive["dot"]
    queries, photos = posteriors(split, text, [DEFAULT_PROMPTS])
    q, p = queries[DEFAULT_PROMPTS[0]], photos[DEFAULT_PROMPTS[1]]
    sq, sp = obs_q[split.query_ids].cuda(), obs_p[split.gallery_ids].cuda()
    aq, ap = additive_predictions(sq, q), additive_predictions(sp, p.T)
    truth, gallery = split.sketch_labels.cuda(), split.photo_labels.cuda()
    replay = PrefixReplay(kd, split.photo_labels, split.name)
    rows, comparisons = {arm: [] for arm in ARMS}, {arm: [] for arm in CONTROLS}
    source, challenges = {arm: [] for arm in (*ARMS, "dot")}, []
    seconds, error = 0., 0.
    print(split.name, "scoring full split", len(q), p.size(1), flush=True)
    for start in range(0, len(q), QUERY_BATCH):
        sl = slice(start, min(start + QUERY_BATCH, len(q)))
        dot = q[sl] @ p
        base_top = prefix(dot, *replay.cutoffs)
        in_base = torch.zeros_like(dot, dtype=torch.bool).scatter_(1, base_top, True)
        _, base_hits = newcomer_hits(base_top, in_base, gallery, truth[sl], reference[sl, 1])
        if not torch.equal(base_hits[:, 0].cpu(), reference[sl, 3].bool()):
            raise ValueError("Dot top-1 replay differed")
        torch.cuda.synchronize()
        tick = time.perf_counter()
        measured = scores(sq[sl], sp, aq[sl], ap, maps)
        torch.cuda.synchronize()
        seconds += time.perf_counter() - tick
        if start == 0:
            for i, j in ((0, 0), (1, 7), (2, 21), (3, 63)):
                exact = enumerate_pair(sq[i], sp[j])
                error = max(error, abs(exact - measured["heldout"][i, j].item()))
            if error > 2e-5:
                raise ValueError(f"Real scores differ from exhaustive float64 reference by {error}")
        ranked = {}
        for arm, score in measured.items():
            if not torch.isfinite(score).all():
                raise ValueError("Nonfinite score")
            values = score_rows(score, split.sketch_labels[sl], gallery, split.name, replay, pool)
            rows[arm].append(values)
            ranked[arm] = newcomer_hits(prefix(score, *replay.cutoffs), in_base, gallery, truth[sl], values[:, 1])
        for arm in comparisons:
            comparisons[arm].append(matched_counts(ranked["heldout"], ranked[arm]))
        challenges.append(conditional_pairs(base_top, measured, gallery, truth[sl]))
        if native_ids is not None:
            by_source = source_rows(native_ids[sl], dict(measured, dot=dot), gallery, truth[sl])
            for arm, result in by_source.items():
                source[arm].append(result)
        if start == 0 or sl.stop == len(q) or (start // QUERY_BATCH + 1) % 20 == 0:
            print(split.name, sl.stop, "/", len(q), flush=True)
    final = {arm: torch.cat(parts).double() for arm, parts in rows.items()}
    final["dot"] = reference
    matched = {arm: torch.cat(parts) for arm, parts in comparisons.items()}
    close = torch.cat(challenges)
    active = close[:, 1] > 0
    source = {arm: torch.cat(parts) for arm, parts in source.items() if parts}
    payload = {"qi": split.query_ids, "pi": split.gallery_ids, "labels": split.sketch_labels,
               "rows": final, "matched": matched, "conditional_pairs": close, "source_rows": source}
    if native_ids is not None:
        payload["source_ids"] = native_ids
    save_tensor(payload, artifact)
    statistics = {
        "protocol": signature, "queries": len(q), "gallery": p.size(1), "classes": len(truth.unique()),
        "cutoffs": list(replay.cutoffs), "metrics": {arm: mean_percent(val[:, :2]) for arm, val in final.items()},
        "heldout_minus_macro_ap": {arm: class_interval(
            100 * (final["heldout"][:, 0] - val[:, 0]), split.sketch_labels, seed=SEED,
        ) for arm, val in final.items() if arm != "heldout"},
        "matched_promotions": {arm: summarize(value, split.sketch_labels) for arm, value in matched.items()},
        "adjacent_null_scores_in_dot_prefix": {
            "mixed_pairs": int(close[:, 0].sum()), "strict_changes": int(close[:, 1].sum()),
            "correct_changes": int(close[:, 2].sum()),
            "macro_change_on_strict_disagreement": class_interval(
                100 * (2 * close[active, 2] - close[active, 1]).double() / close[active, 1],
                split.sketch_labels[active], seed=SEED,
            ),
        },
        "source_identity_within_class": {arm: {
            "verified_queries": len(rank), "mrr_percent": float(100 * (1/rank).mean()),
            "recall1_percent": float(100 * (rank == 1).double().mean()),
            "two_choice_accuracy_nearest_null_score": float(100*wins.mean()),
        } for arm, value in source.items() for rank, wins in [value.T]},
        "scoring_seconds_all_arms": seconds, "max_error_on_double_reference": error,
        "per_query_sha256": checksum(artifact),
    }
    atomic_json(target, statistics)
    print(split.name, "verified", statistics["metrics"], flush=True)


@torch.inference_mode()
def run(dataset):
    seed_job()
    w = Path("/content/sbir")
    root = w / "results" / FAMILY
    words, vocab_sha = public_words()
    maps = Correspondences(torch.device("cuda"))
    kd, _ = load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("KD evaluator changed")
    signature = save_lock(root / "freeze.json", {
        "commit": revision(), "clip_sha256": CHECKPOINT, "evaluator_sha256": KD_FUNCTION_SHA,
        "vocabulary_sha256": vocab_sha, "seed": SEED, "prompts": list(DEFAULT_PROMPTS),
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "arms": list(ARMS),
        "posterior": "all four single deletions", "predictive_event": "six double deletions",
        "selection": "geometry, inference and controls locked before reading scores",
        "primary": "class-macro purity of matched new photos versus exact average over identity shuffles",
        "fixture": fixture(torch.device("cuda"), maps),
    })
    workers = min(8, len(os.sched_getaffinity(0)))
    cfg = runtime(w, dataset, workers=workers)
    names = SKETCHY if dataset == "sketchy" else ("tuberlin",)
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = {s: encode_prompt(model, tokenizer, words, s) for s in DEFAULT_PROMPTS}
    data = image_inputs(cfg, names, model)
    q_paths, _ = merge_inputs(cfg.root, names, "sketch")
    p_paths, _ = merge_inputs(cfg.root, names, "photo")
    banks = []
    for mode, paths in (("sketch", q_paths), ("photo", p_paths)):
        folder, spec = directory(cfg, dataset, mode, paths, vocab_sha)
        if json.loads((folder / "complete.json").read_bytes())["specification"] != spec:
            raise ValueError("Wrong set of CLIP interventions")
        banks.append(read_cache(folder))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name in names:
            split = Split.from_inputs(data, name)
            ids = sources_for_sketches(split, q_paths, p_paths) if dataset == "sketchy" else None
            evaluate(root, split, text, *banks, kd, pool, maps, signature, ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("sketchy", "tuberlin"))
    run(parser.parse_args().dataset)
