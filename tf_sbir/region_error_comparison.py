"""Compare stored action headroom with the stronger regional baseline and replay representative queries."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import digest
from .cross_view_region_cache import FAMILY as VIEWS
from .cross_view_regions import CandidateRegions
from .gallery_region_acquisition_study import FAMILY
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .regional_intervention_study import index as compact_index
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import load_kd, paired_delta
from .semantic import read_words
from .foundation import load_words


@torch.inference_mode()
def run(w):
    root = w / "results" / FAMILY
    cases = read(root / "locked-error-diagnosis.json")
    kd, _ = load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("Evaluating with a different reference")
    report = {"unit": "metrics are fractions; differences are percentage points", "splits": {}}
    for ds in ("sketchy", "tuberlin"):
        cfg, qp, qs, pp, galleries, qraw, praw = global_inputs(w, ds)
        words = read_words(cfg.vocabulary)
        src, tgt = (v.cuda() for v in load_words(cfg, words))
        _, scale = bare_words(cfg, words)
        cache = torch.load(w / "cache" / VIEWS / ds / "views.pt", weights_only=True, mmap=True)
        for s, (qi, y) in qs.items():
            pi, g = galleries[s]
            target = praw[pi].cuda()
            strong_dir = w / "results" / VIEWS
            anchor = read(strong_dir / (s + ".json"))
            primary = read(root / (s + ".json"))
            for folder, rec in ((strong_dir, anchor), (root, primary)):
                if checksum(folder / (s + ".pt")) != rec["per_query_sha256"]:
                    raise ValueError("Unexpected per-query source")
            recent = torch.load(root / (s + ".pt"), weights_only=True)["metrics"]
            older = torch.load(strong_dir / (s + ".pt"), weights_only=True)
            if not torch.equal(older["query_labels"], y):
                raise ValueError("Query orders differ")
            strong = older["metrics"][anchor["best_contrast"]]
            oracle = recent["oracle_one_action"]
            difference = (oracle[:, 0] - strong[:, 0]) * 100
            difficult = strong[:, 0] <= strong[:, 0].quantile(.25)
            examples = cases["splits"][s]["representative_errors"]
            at = torch.tensor([e["split_query_index"] for e in examples])
            raw = qraw[qi[at]].cuda()
            history = torch.load(w / "cache/pragmatic-gallery-competition" / (s + "-named-rgb16.pt"), weights_only=True, mmap=True)[at].cuda()
            ident = digest({"signature": anchor["signature"], "gallery": digest([str(pp[j]) for j in pi.tolist()]),
                            "vocabulary": checksum(cfg.vocabulary)})[:16]
            heads, comp = compact_index(w, s, target, tgt, scale, ident)
            regions = CandidateRegions(raw, cache["global"][qi[at]].cuda(), cache["original_regions"][qi[at]].cuda(),
                                       cache["regions"][qi[at]].cuda(), src, heads, comp, src.new_empty((0, len(src))), history)
            regions.prepare(scale)
            scores = {"native_cosine": raw @ target.T, "strong_regional": .65 * regions.first + .35 * regions.read("self", 1., "gallery")}
            extra = [{} for _ in examples]
            reference = Replay(kd, g, s)
            with ThreadPoolExecutor(3) as pool:
                for method, x in scores.items():
                    fast, order, relevance, _ = map_values(x, y[at].cuda(), g.cuda(), s)
                    actual = reference.score(order, y[at], pool)
                    if float((actual - fast.cpu()).abs().max()) > 2e-6:
                        raise ValueError("Reference mismatch on representative query")
                    for j, rec in enumerate(extra):
                        photo = Path(pp[int(pi[int(order[j, 0])])])
                        rec[method] = {"top_class": photo.parent.name, "top_photo": photo.name,
                                       "rank_of_first_correct": int(relevance[j].int().argmax()) + 1,
                                       "AP": float(actual[j, 0]), "precision": float(actual[j, 1])}
            out = dict(zip(("queries","gallery"), (len(qi),len(pi))))
            out.update({"strong": anchor["best_contrast"], "strong_sha256": anchor["per_query_sha256"],
                        "one_action_sha256": primary["per_query_sha256"], "oracle_vs_strong": paired_delta(oracle, strong, y),
                        "share_oracle_at_least_5pp_better": float((difference >= 5).float().mean()),
                        "share_strong_at_least_5pp_better": float((difference <= -5).float().mean()),
                        "lower_quartile_strong_AP_cutoff": float(strong[difficult,0].max()),
                        "lower_quartile_mean_oracle_gain_pp": float(difference[difficult].mean()),
                        "case_stored_vs_replayed_strong_max_abs_error": max(abs(float(strong[int(at[j]), 0]) - case["strong_regional"]["AP"]) for j,case in enumerate(extra)),
                        "cases": [{"query": row["query"], "truth": row["true_class"], "single_zoom_wrong": row["top_photo_class"],
                                   "single_zoom_ap": row["actual_ap"], "oracle_ap": row["oracle_ap"]} | replay for row,replay in zip(examples,extra)]})
            report["splits"][s] = out
    atomic_json(root / "strong-error-comparison.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    run(p.parse_args().workspace)
