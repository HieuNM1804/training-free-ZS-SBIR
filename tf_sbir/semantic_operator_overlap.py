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


def added_candidate_counts(query, banks, shared_positions, shared_labels, all_labels):
    """Count entries from the 17 other classes in rankings of the four shared classes."""
    is_added = torch.ones(len(all_labels), device=query.device, dtype=torch.bool)
    is_added[shared_positions] = False
    if torch.isin(all_labels[is_added.cpu()], shared_labels.unique()).any():
        raise ValueError("A candidate counted as an added distractor can be relevant")
    top_group = int(np.ceil(is_added.sum().item() / 100))
    result = {}
    for name, bank in banks.items():
        gallery = bank.float().T.contiguous()
        ranks = torch.cat([(part @ gallery).topk(200, dim=1).indices for part in query.split(64)])
        new_at_rank = is_added[ranks]
        counts = ranks[new_at_rank].bincount(minlength=len(all_labels))
        result[name] = {"added_photos": int(is_added.sum()), "mean_added_in_top200": new_at_rank.sum(1).double().mean().item(),
                        "fraction_of_queries_with_added_top1": new_at_rank[:, 0].double().mean().item(),
                        "most_frequent_one_percent_count": top_group,
                        "fraction_of_added_top200_slots_from_most_frequent_one_percent":
                        (counts.topk(top_group).values.double().sum() / counts.sum()).item()}
    return result


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
    additions = added_candidate_counts(queries[1], {"dot": bank, "donor64": indices[1]["b"].cuda(),
                                       "query_projection": indices[1]["f"].cuda(), "fixed_ext1_query_edit": fixed},
                                       photo_rows[1], sample.sketch_labels, right.photo_labels)
    out = {"revision": revision, "base_revision": base_revision, "seed": SEED, "device": torch.cuda.get_device_name(),
           "type": "post-hoc descriptive diagnostic on Sketchy only", "shared_classes": shared_classes,
           "full_ext2_gallery": full_gallery_groups(right, query_rows[1], base_revision),
           "matched_candidates": {"queries": len(queries[1]), "photos": len(photo_rows[1]), "cutoffs": [200, 200],
            "query_posterior_max_abs_difference": q_error, "photo_posterior_max_abs_difference": p_error,
            "donors_per_photo": K, "shared_donor_count_mean": overlap.mean().item(),
            "shared_donor_count_median": overlap.median().item(), "context_changes": changed, "metrics": metrics},
           "added_distractors_for_shared_queries": additions,"fixed_shared_index_max_abs_difference": fixed_error}
    atomic_json(DST / "sketchy-overlap.json", out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    deterministic()
    run(parser.parse_args().revision)
