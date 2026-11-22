"""Post-hoc decomposition and real query examples from the locked one-crop rankings."""

import argparse
from collections import Counter
from pathlib import Path

import torch

from .benchmark import merge_inputs
from .concept_benchmark import atomic_json, checksum, read
from .crossdomain_inputs import runtime
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .gallery_region_acquisition_study import FAMILY
from .regional_interventions import SIDES, windows
from .retrieval_evidence import paired_delta


def case_rows(qp, pp, qi, pi, y, labels, wrong, top_id, actual, chosen, best, oracle):
    error_indices = wrong.nonzero().flatten()
    predicted = labels[top_id]
    pairs = Counter(zip(y[error_indices].tolist(), predicted[error_indices].tolist()))
    cases = []
    for (truth, returned), count in pairs.most_common(3):
        ids = (wrong & (y == truth) & (predicted == returned)).nonzero().flatten()
        gap = oracle[ids, 0] - actual[ids, 0]
        i = int(ids[gap.argsort(stable=True)[len(ids) // 2]])
        query = Path(qp[int(qi[i])])
        photo = Path(pp[int(pi[int(top_id[i])])])
        cases.append({"true_class": query.parent.name, "top_photo_class": photo.parent.name,
                      "rank1_errors_for_pair": count, "query": query.name, "top_photo": photo.name,
                      "split_query_index": i, "actual_region": int(chosen[i]), "oracle_region": int(best[i]),
                      "actual_ap": float(actual[i, 0]), "oracle_ap": float(oracle[i, 0])})
    return cases


@torch.inference_mode()
def run(w):
    root = w / "results" / FAMILY
    windows_by_id = [{"side_patches": n, "top_row": y, "left_column": x} for n in SIDES for y, x in windows(n)]
    report = {"unit": "percentage points for paired deltas; AP and shares as fractions",
              "scope": "post-hoc diagnosis of the previously frozen system, no inference or selection feedback",
              "case_rule": "three most frequent wrong rank-one category pairs, median oracle-minus-active AP sketch in each",
              "window_ids": windows_by_id, "splits": {}}
    for ds in ("sketchy", "tuberlin"):
        cfg = runtime(w, ds, batch=384, workers=6)
        splits = SKETCHY_SPLITS if ds == "sketchy" else (ds,)
        qp, qs = merge_inputs(cfg.root, splits, "sketch")
        pp, gs = merge_inputs(cfg.root, splits, "photo")
        for s in splits:
            qi, y = qs[s]
            pi, g = gs[s]
            official = read(root / (s + ".json"))
            file = root / (s + ".pt")
            if checksum(file) != official["per_query_sha256"]:
                raise ValueError("Official per-action artifact differs")
            details = torch.load(file, weights_only=True)
            if not torch.equal(y, details["query_labels"]):
                raise ValueError("Wrong query order")
            actions = details["actions"]
            metrics = details["metrics"]
            base = metrics["uniform_random_action"]
            size = metrics["random_at_selected_scale"]
            actual = metrics[official["primary"]]
            fixed = metrics[official["matched_base"]]
            full = metrics["all13"]
            per_action = details["per_action_metrics"]
            best = per_action[:, :, 0].argmax(1)
            ids = torch.arange(len(y))
            oracle = per_action[ids, best]
            if not torch.equal(actual, per_action[ids, actions]) or not torch.equal(oracle, metrics["oracle_one_action"]):
                raise ValueError("Actual or oracle metrics differ from original action evaluations")
            delta = {"scale": paired_delta(size, base, y), "position_at_scale": paired_delta(actual, size, y),
                     "total": paired_delta(actual, base, y), "vs_fixed_at_scale": paired_delta(actual, fixed, y),
                     "cost_of_one_action": paired_delta(actual, full, y), "oracle_headroom": paired_delta(oracle, actual, y)}
            identity = (actual.double() - base.double()) - ((actual.double() - size.double()) + (size.double() - base.double()))
            miss = ~details["new_top1"]
            top = details["top25"][:, 0].long()
            if not torch.equal(miss, g[top] != y):
                raise ValueError("Rank-one correctness mismatch")
            cases = case_rows(qp, pp, qi, pi, y, g, miss, top, actual, actions, best, oracle)
            if any(e["true_class"] not in UNSEEN_CLASSES[s] or e["top_photo_class"] not in UNSEEN_CLASSES[s] for e in cases):
                raise ValueError("Wrong dataset class name")
            report["splits"][s] = {"primary": official["primary"], "per_action_sha256": official["per_query_sha256"],
                                   "queries": len(y), "rank_one_errors": int(miss.sum()), "deltas": delta,
                                   "decomposition_identity_error": float(identity.abs().max()),
                                   "fraction_of_random_improvement_from_scale": delta["scale"]["mAP"]["delta_pp"] / delta["total"]["mAP"]["delta_pp"],
                                   "selected_oracle_frequency": float((best == actions).float().mean()),
                                   "within_one_ap_point_of_oracle": float((oracle[:, 0] - actual[:, 0] <= .01).float().mean()),
                                   "oracle_gain_among_rank1_errors_pp": float(100 * (oracle[miss, 0] - actual[miss, 0]).mean()),
                                   "representative_errors": cases}
    atomic_json(root / "locked-error-diagnosis.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    run(p.parse_args().workspace)
