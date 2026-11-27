"""Post-hoc diagnostic on the Sketchy classes and instances shared by both splits."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json

import numpy as np
import torch
from torch.nn.functional import normalize

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .conditional_completion_study import evaluate
from .data import UNSEEN_CLASSES
from .gallery_diffusion import nearest_gallery
from .kaggle_unseen_features import DEFAULT_CACHE, load_split
from .native_block_probe import SEED, deterministic
from .replicated_simplex_study import evaluator
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import mean_percent
from .semantic_disputed_word_risk import query_posteriors
from .semantic_operator_audit import DST, K, design, word_fingerprint


SPLITS = ("sketchy_1", "sketchy_2")


def common_positions(left, right):
    if len(left.unique()) != len(left) or len(right.unique()) != len(right):
        raise ValueError("Duplicate instance ID")
    _, il, ir = np.intersect1d(left.numpy(), right.numpy(), assume_unique=True, return_indices=True)
    return torch.from_numpy(il), torch.from_numpy(ir)


def class_names(split, labels):
    return [UNSEEN_CLASSES[split][i] for i in labels.tolist()]


def full_gallery_groups(split, query_positions, base_revision):
    parts = []
    for part in (0, 1):
        path = DST / f"{split.name}-{part}.pt"
        meta = json.loads(path.with_suffix(".json").read_bytes())
        if meta["revision"] != base_revision or checksum(path) != meta["sha256"]:
            raise ValueError("Unexpected metric shard")
        parts.append(torch.load(path, weights_only=True))
    order = torch.cat([d["positions"] for d in parts]).argsort()
    all_ids = torch.cat([d["query_ids"] for d in parts])[order]
    if not torch.equal(all_ids, split.query_ids):
        raise ValueError("Saved metric rows differ from the dataset order")
    shared = torch.zeros(len(all_ids), dtype=torch.bool)
    shared[query_positions] = True
    result = {}
    for name, mask in (("shared", shared), ("other", ~shared)):
        metrics = {}
        for condition in parts[0]["metrics"]:
            rows = torch.cat([d["metrics"][condition] for d in parts])[order]
            metrics[condition] = mean_percent(rows[mask])
        result[name] = {"queries": int(mask.sum()), "gallery": len(split.gallery_ids), "metrics": metrics,
                        "classes": sorted(set(class_names(split.name, split.sketch_labels[mask])))}
    return result


def source_leverage(source, target):
    # Squared norm after whitening the affine word coordinates by the source design.
    x = design(source)
    _, r = torch.linalg.qr(x, mode="reduced")
    def measure(rows):
        coordinates = torch.linalg.solve_triangular(r.T, rows.T.contiguous(), upper=False)
        return coordinates.square().sum(0)
    leverage = measure(x)
    if abs(leverage.sum().item() - x.shape[1]) > 1e-6 or leverage.max() > 1.0000001:
        raise ValueError("Source leverage does not match its projection matrix")
    return measure(design(target))


def percentile(values):
    reference = values.sort().values
    lo = torch.searchsorted(reference, values)
    hi = torch.searchsorted(reference, values, right=True)
    return (lo + hi).double() / (2 * len(values))


def class_percentiles(leverage, labels):
    labels = labels.to(leverage.device)
    result = torch.empty_like(leverage)
    for label in labels.unique():
        take = labels == label
        result[take] = percentile(leverage[take])
    return result


def added_candidate_counts(query, banks, shared_positions, shared_labels, all_labels, leverage, per_class):
    """Count entries from the 17 other classes in rankings of the four shared classes."""
    is_added = torch.ones(len(all_labels), device=query.device, dtype=torch.bool)
    is_added[shared_positions] = False
    if torch.isin(all_labels[is_added.cpu()], shared_labels.unique()).any():
        raise ValueError("A candidate counted as an added distractor can be relevant")
    top_group = int(np.ceil(is_added.sum().item() / 100))
    hardest, wrong_heads = {}, {}
    result = {}
    for name, bank in banks.items():
        gallery = bank.float().T.contiguous()
        ranked_blocks, best_added = [], []
        for part in query.split(64):
            scores = part @ gallery
            ranked_blocks.append(scores.topk(200, dim=1).indices)
            best_added.append(scores.masked_fill(~is_added[None, :], -torch.inf).argmax(1))
        ranks = torch.cat(ranked_blocks)
        hardest[name] = torch.cat(best_added)
        new_at_rank = is_added[ranks]
        wrong_heads[name] = ranks[new_at_rank[:, 0], 0]
        counts = ranks[new_at_rank].bincount(minlength=len(all_labels))
        result[name] = {"added_photos": int(is_added.sum()), "mean_added_in_top200": new_at_rank.sum(1).double().mean().item(),
                        "fraction_of_queries_with_added_top1": new_at_rank[:, 0].double().mean().item(),
                        "most_frequent_one_percent_count": top_group,
                        "fraction_of_added_top200_slots_from_most_frequent_one_percent":
                        (counts.topk(top_group).values.double().sum() / counts.sum()).item()}
    levels = [0.50, 0.90, 0.99]
    quantiles = torch.tensor(levels, device=leverage.device, dtype=leverage.dtype)
    report = {"quantiles": levels, "shared_photo_leverage": torch.quantile(leverage[~is_added], quantiles).tolist(),
              "added_photo_leverage": torch.quantile(leverage[is_added], quantiles).tolist(),
              "best_added_photo_mean_within_its_class_percentile":
              {k: per_class[v].mean().item() for k, v in hardest.items()}}
    base = per_class[hardest["dot"]]
    fixed = per_class[hardest["fixed_ext1_query_edit"]]
    report["paired_fraction_higher_with_fixed_edit_than_dot"] = (fixed > base).double().mean().item()
    report["paired_fraction_equal"] = (fixed == base).double().mean().item()
    return result, report, wrong_heads


def positive_controls(split, text, shared_rows, banks, per_class, wrong_heads):
    other = torch.ones(len(split.query_ids), dtype=torch.bool)
    other[shared_rows] = False
    ids = torch.where(other)[0]
    query = torch.cat([query_posteriors(split.sketch[sl], split.mirror[sl], text[DEFAULT_PROMPTS[0]], split.scale)
                       for sl in ids.split(384)])
    truth = split.sketch_labels[ids].cuda()
    classes = split.photo_labels.cuda()
    results = {}
    for name, bank in banks.items():
        gallery = bank.float().T.contiguous()
        highest, best_positive = [], []
        for q, labels in zip(query.split(64), truth.split(64), strict=True):
            scores = q @ gallery
            highest.append(scores.argmax(1))
            best_positive.append(scores.masked_fill(classes[None, :] != labels[:, None], -torch.inf).argmax(1))
        top = torch.cat(highest)
        correct = top[classes[top] == truth]
        positive_percentiles = per_class[torch.cat(best_positive)]
        wrong = wrong_heads[name]
        also_correct = torch.isin(wrong.unique(), correct.unique())
        results[name] = {"queries": len(query), "correct_top1_queries": len(correct),
                         "best_positive_mean_percentile_within_its_class": positive_percentiles.mean().item(),
                         "correct_top1_mean_percentile_within_its_class": per_class[correct].mean().item(),
                         "fraction_of_correct_top1_in_own_class_top_leverage_decile":
                         (per_class[correct] >= 0.90).double().mean().item(),
                         "distinct_wrong_added_top1_photos_for_four_shared_classes": len(wrong.unique()),
                         "number_that_are_correct_top1_for_some_other_query": int(also_correct.sum()),
                         "fraction_of_shared_query_wrong_added_top1_occurrences_that_are_correct_top1_elsewhere":
                         torch.isin(wrong, correct.unique()).double().mean().item()}
    return results


@torch.inference_mode()
def run(revision):
    source, target = [load_split(DEFAULT_CACHE, name) for name in SPLITS]
    left, right = source[0], target[0]
    query_rows = common_positions(left.query_ids, right.query_ids)
    photo_rows = common_positions(left.gallery_ids, right.gallery_ids)
    shared_classes = sorted(set(UNSEEN_CLASSES[left.name]) & set(UNSEEN_CLASSES[right.name]))
    for rows, label_field in ((query_rows, "sketch_labels"), (photo_rows, "photo_labels")):
        names = [class_names(s.name, getattr(s, label_field)[r]) for s, r in zip((left, right), rows, strict=True)]
        if names[0] != names[1] or sorted(set(names[0])) != shared_classes:
            raise ValueError("Shared IDs do not agree on class name")
    indices = [torch.load(DST / f"index-{name}.pt", mmap=True, weights_only=True) for name in SPLITS]
    base_revision = indices[0]["revision"]
    queries, donors, matched = [], [], {}
    for (split, text, words), index, qi, pi in zip((source, target), indices, query_rows, photo_rows, strict=True):
        if (index["revision"] != base_revision or index["word_fingerprint"] != word_fingerprint(words)
                or not torch.equal(index["gallery_ids"], split.gallery_ids)):
            raise ValueError("Gallery index mismatch")
        queries.append(torch.cat([query_posteriors(split.sketch[chunk], split.mirror[chunk],
                                   text[DEFAULT_PROMPTS[0]], split.scale) for chunk in qi.split(384)]))
        for key in ("p", "b", "f"):
            matched[(split.name, key)] = index[key][pi].cuda()
        neighbors, _ = nearest_gallery(normalize(split.photo.float(), dim=1), 256)
        local = neighbors[pi, :K].cpu()
        for rr, nn in zip(pi.split(128), local.split(128), strict=True):
            if (index["p"][nn].mean(1) - index["b"][rr]).abs().max() > 1e-9:
                raise ValueError("Shared donor reconstructions differ from the completed experiment")
        donors.append(split.gallery_ids[local])
    q_error = (queries[0] - queries[1]).abs().max().item()
    p_error = (matched[(left.name, "p")] - matched[(right.name, "p")]).abs().max().item()
    if max(q_error, p_error) > 1e-5:
        raise ValueError(f"Matched instances differ before any gallery intervention: {q_error}, {p_error}")
    overlap = (donors[0][:, :, None] == donors[1][:, None, :]).any(2).sum(1).double()
    sample = replace(right, query_ids=right.query_ids[query_rows[1]], gallery_ids=right.gallery_ids[photo_rows[1]],
                     sketch_labels=right.sketch_labels[query_rows[1]], photo_labels=right.photo_labels[photo_rows[1]])
    kd = evaluator(DEFAULT_CACHE.parent / "real-sketchy-smoke" / "evaluator.py")
    replay = PrefixReplay(kd, sample.photo_labels, sample.name)
    metrics, heads = {}, {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for (context, key), bank in matched.items():
            if key == "p" and context == left.name:
                continue
            query = queries[1]
            bank = bank.float().contiguous()
            metrics[f"{context}/{key}"] = mean_percent(evaluate(sample, query, bank, replay, pool))
            heads[(context, key)] = (query @ bank.T).argmax(1).cpu()
    changed = {}
    for key in ("b", "f"):
        one, two = heads[(left.name, key)], heads[(right.name, key)]
        changed[key] = {"top1_photo_changed_fraction": (one != two).double().mean().item(),
                        "top1_class_changed_fraction": (sample.photo_labels[one] != sample.photo_labels[two]).double().mean().item()}
    bank = indices[1]["p"].cuda()
    fixed = design(bank) @ indices[0]["a"].cuda()
    fixed_error = (fixed[photo_rows[1]] - matched[(left.name, "f")]).abs().max().item()
    if fixed_error > 1e-7:
        raise ValueError("Source calibration changed on the common candidates")
    banks = {"dot": bank, "donor64": indices[1]["b"].cuda(),
             "query_projection": indices[1]["f"].cuda(), "fixed_ext1_query_edit": fixed}
    leverage = source_leverage(indices[0]["p"].cuda(), bank)
    per_class = class_percentiles(leverage, right.photo_labels)
    additions, extrapolation, wrong = added_candidate_counts(queries[1], banks, photo_rows[1], sample.sketch_labels,
                                                              right.photo_labels, leverage, per_class)
    positives = positive_controls(right, target[1], query_rows[1], banks, per_class, wrong)
    out = {"revision": revision, "base_revision": base_revision, "seed": SEED, "device": torch.cuda.get_device_name(),
           "type": "post-hoc descriptive diagnostic on Sketchy only", "shared_classes": shared_classes,
           "full_ext2_gallery": full_gallery_groups(right, query_rows[1], base_revision),
           "matched_candidates": {"queries": len(queries[1]), "photos": len(photo_rows[1]), "cutoffs": [200, 200],
            "query_posterior_max_abs_difference": q_error, "photo_posterior_max_abs_difference": p_error,
            "donors_per_photo": K, "shared_donor_count_mean": overlap.mean().item(),
            "shared_donor_count_median": overlap.median().item(), "context_changes": changed, "metrics": metrics},
           "added_distractors_for_shared_queries": additions, "source_leverage": extrapolation,
           "positive_control_other_17_classes": positives, "fixed_shared_index_max_abs_difference": fixed_error}
    atomic_json(DST / "sketchy-overlap.json", out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    deterministic()
    run(parser.parse_args().revision)
