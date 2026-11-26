"""An exact top-K search bound for the held-out intervention likelihood."""

import argparse
import json
import os
from pathlib import Path
from statistics import median
import time

import torch

from .benchmark import merge_inputs
from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .crossdomain_inputs import runtime
from .gallery_convex_support import SEED
from .heldout_occlusion_correspondence import Correspondences, overlap
from .part_occlusion_observations import FAMILY, directory, public_words, read_cache, seed_job
from .semantic_boxable_prompt_search import SKETCHY


def primary(query, photo, maps):
    singles = overlap(query[:, :4], photo[:, :4]).clamp_min(1e-30).log() @ maps.single_assignment
    pairs = overlap(query[:, 4:], photo[:, 4:]).clamp_min(1e-30).log() @ maps.double_assignment
    return torch.logsumexp(singles.log_softmax(2) + pairs, 2) / 6


def enclosing_vector(observations):
    # Sorting independently for each word solves the word-wise assignment relaxation.
    return observations[:, 4:].sort(dim=1).values.flatten(1)


def upper_bound(query, photo_vectors):
    size = photo_vectors.size(1)
    eps = torch.finfo(torch.float32).eps
    # Outward allowance on the positive dot and the division. The observation
    # overlap floor is also included, as it is used in production log scoring.
    gamma = (size + 2) * eps / (1 - (size + 2) * eps)
    return ((enclosing_vector(query) @ photo_vectors.T) / 6 * (1 + gamma) + 1e-30).log() + 1e-5


def canonical_top(scores, ids, k):
    # Reproduce an explicit, gallery-ID tie policy for both search implementations.
    by_id = ids.argsort(stable=True)
    by_score = scores[by_id].argsort(descending=True, stable=True)[:k]
    positions = by_id[by_score]
    return scores[positions], ids[positions]


def retrieve(query, photo, index, maps, k):
    bound = upper_bound(query, index)[0]
    order = bound.argsort(descending=True, stable=True)
    best_scores = bound.new_empty(0)
    best_ids = order.new_empty(0)
    for end in range(k, len(photo) + k, k):
        stop = min(end, len(photo))
        ids = order[end-k:stop]
        values = primary(query, photo[ids], maps)[0]
        best_scores, best_ids = canonical_top(torch.cat((best_scores, values)), torch.cat((best_ids, ids)), k)
        if stop == len(photo) or bound[order[stop]] < best_scores[-1]:
            return best_ids, stop
    raise ValueError("Failed to exhaust gallery")


def experiment(sq, sp, k, ids, maps):
    torch.cuda.synchronize()
    tick = time.perf_counter()
    index = enclosing_vector(sp)
    torch.cuda.synchronize()
    index_time = time.perf_counter() - tick
    all_ids = torch.arange(len(sp), device=sp.device)
    counts, times_scan, times_search = [], [], []
    equal_membership = equal_order = ties = 0
    greatest_overshoot = 0.
    for row in ids.tolist():
        query = sq[row:row+1]
        torch.cuda.synchronize()
        tick = time.perf_counter()
        exact = primary(query, sp, maps)[0]
        ordered, expected = canonical_top(exact, all_ids, k)
        torch.cuda.synchronize()
        times_scan.append(time.perf_counter() - tick)
        tick = time.perf_counter()
        actual, checked = retrieve(query, sp, index, maps, k)
        torch.cuda.synchronize()
        times_search.append(time.perf_counter() - tick)
        counts.append(checked)
        equal_order += int(torch.equal(actual, expected))
        equal_membership += int(torch.equal(actual.sort().values, expected.sort().values))
        ties += int((exact == ordered[-1]).sum() > 1)
        greatest_overshoot = max(greatest_overshoot, float((exact - upper_bound(query, index)[0]).max()))
    if equal_membership != len(ids) or greatest_overshoot > 0:
        raise ValueError(f"Bounded index failed exact verification: {equal_membership=} {greatest_overshoot=}")
    return {
        "gallery": len(sp), "query_rows": ids.tolist(), "k": k, "checked_counts": counts,
        "mean_fraction_scored_exactly": sum(counts)/(len(counts)*len(sp)),
        "matching_membership": equal_membership, "matching_complete_order": equal_order,
        "queries_with_ties_at_cutoff": ties, "largest_outward_bound_violation": greatest_overshoot,
        "auxiliary_index_megabytes": index.numel()*index.element_size()/1e6, "build_index_seconds": index_time,
        "median_ms_exhaustive": 1000*median(times_scan), "median_ms_bounded": 1000*median(times_search),
    }


@torch.inference_mode()
def run(dataset):
    seed_job()
    w = Path("/content/sbir")
    root = w / "results" / FAMILY
    source = json.loads((root/"freeze.json").read_bytes())
    _, vocab_sha = public_words()
    if source["config"]["vocabulary_sha256"] != vocab_sha:
        raise ValueError("Primary vocabulary differs")
    cfg = runtime(w, dataset, workers=min(8, len(os.sched_getaffinity(0))))
    names = SKETCHY if dataset == "sketchy" else ("tuberlin",)
    banks, filters = [], []
    for mode in ("sketch", "photo"):
        paths, positions = merge_inputs(cfg.root, names, mode)
        folder, spec = directory(cfg, dataset, mode, paths, vocab_sha)
        if json.loads((folder/"complete.json").read_bytes())["specification"] != spec:
            raise ValueError("Masked measurements differ")
        banks.append(read_cache(folder))
        filters.append(positions)
    maps = Correspondences(torch.device("cuda"))
    for name in names:
        query_ids, gallery_ids = (table[name][0] for table in filters)
        rows = torch.randperm(len(query_ids), generator=torch.Generator().manual_seed(SEED))[:64]
        q = banks[0][query_ids[rows]].cuda()
        p = banks[1][gallery_ids].cuda()
        k = 200 if name == "sketchy_2" else 100
        report = experiment(q, p, k, torch.arange(len(rows)), maps)
        report["query_rows"] = rows.tolist()
        report.update({"primary_protocol": source["sha256"], "audit_commit": revision(),
                       "seed": SEED, "gpu": torch.cuda.get_device_name()})
        atomic_json(root / (name+"-index.json"), report)
        print(name, report["matching_membership"], report["mean_fraction_scored_exactly"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("sketchy", "tuberlin"))
    run(parser.parse_args().dataset)
