"""How often does the public-word retriever overrule a genuinely correct frozen native ranking?"""

import argparse
from collections import Counter
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
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .gallery_region_acquisition_study import FAMILY
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import load_kd, paired_delta
from .regional_intervention_study import index as compact_index
from .semantic import read_words


@torch.inference_mode()
def run(w):
    root = w / "results" / FAMILY
    reference = w / "results" / VIEWS
    kd, _ = load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("Unexpected evaluator")
    summaries, tensors = {}, {}
    for ds in ("sketchy", "tuberlin"):
        cfg, _, qs, pp, galleries, qraw, praw = global_inputs(w, ds)
        words = read_words(cfg.vocabulary)
        src, tgt = (v.cuda() for v in load_words(cfg, words))
        _, scale = bare_words(cfg, words)
        views = torch.load(w / "cache" / VIEWS / ds / "views.pt", weights_only=True, mmap=True)
        for s, (qi, y) in qs.items():
            pi, g = galleries[s]
            photos = praw[pi].cuda()
            gallery_labels = g.cuda()
            old = read(reference / (s + ".json"))
            tensor = reference / (s + ".pt")
            if checksum(tensor) != old["per_query_sha256"]:
                raise ValueError("Reference ranks changed")
            saved = torch.load(tensor, weights_only=True)
            strong = saved["metrics"][old["best_contrast"]]
            if not torch.equal(y, saved["query_labels"]):
                raise ValueError("Different query order")
            ident = digest({"signature": old["signature"], "gallery": digest([str(pp[j]) for j in pi.tolist()]),
                            "vocabulary": checksum(cfg.vocabulary)})[:16]
            heads, compact = compact_index(w, s, photos, tgt, scale, ident)
            global_words = torch.load(w / "cache/pragmatic-gallery-competition" / (s + "-named-rgb16.pt"), weights_only=True, mmap=True)
            replay = Replay(kd, g, s)
            raw_rows, primary_idx, raw_idx = [], [], []
            raw_err = strong_max = strong_sum = 0.
            with ThreadPoolExecutor(10) as pool:
                for start in range(0, len(y), 32):
                    loc = slice(start, start + 32)
                    ids = qi[loc]
                    q = qraw[ids].cuda()
                    gold = y[loc]
                    regional = CandidateRegions(q, views["global"][ids].cuda(), views["original_regions"][ids].cuda(),
                                                views["regions"][ids].cuda(), src, heads, compact, src.new_empty((0, len(src))), global_words[loc].cuda())
                    regional.prepare(scale)
                    ranks = .35 * regional.read("self", 1., "gallery") + .65 * regional.first
                    approx, strong_order, _, _ = map_values(ranks, gold.cuda(), gallery_labels, s)
                    diff = (approx.cpu() - strong[loc]).abs()
                    strong_sum += float(diff.sum())
                    strong_max = max(strong_max, float(diff.max()))
                    # The original KD function maps native cosine into [0,1] before sorting.
                    native_score = ((torch.nn.functional.normalize(q) @ torch.nn.functional.normalize(photos).T) + 1.) / 2.
                    fast, order, _, _ = map_values(native_score, gold.cuda(), gallery_labels, s)
                    metrics = replay.score(order, gold, pool)
                    raw_err = max(raw_err, float((metrics - fast.cpu()).abs().max()))
                    if raw_err > 2e-6:
                        raise ValueError("Native per-query KD rank replay mismatch")
                    primary_idx.append(strong_order[:, 0].cpu())
                    raw_idx.append(order[:, 0].cpu())
                    raw_rows.append(metrics)
            native = torch.cat(raw_rows)
            strong_top, raw_top = torch.cat(primary_idx), torch.cat(raw_idx)
            good = g[strong_top] == y
            raw_good = g[raw_top] == y
            recovered = ~good & raw_good
            if strong_sum / strong.numel() > 1e-5:
                raise ValueError("Reconstructed regional rankings differ substantially from the recorded official reference")
            best = torch.where((native[:, 0] > strong[:, 0])[:, None], native, strong)
            counts = Counter(zip(y[recovered].tolist(), g[strong_top][recovered].tolist()))
            escaped = [{"true": UNSEEN_CLASSES[s][a], "regional_top": UNSEEN_CLASSES[s][b], "count": n} for (a,b),n in counts.most_common(6)]
            summaries[s] = {"queries": len(y), "gallery": len(g), "native_map_precision": native.mean(0).tolist(),
                            "regional_map_precision": strong.mean(0).tolist(), "both_correct": int((good & raw_good).sum()),
                            "both_wrong": int((~good & ~raw_good).sum()), "only_regional_correct": int((good & ~raw_good).sum()),
                            "only_native_correct": int(recovered.sum()), "regional_misses": int((~good).sum()),
                            "native_recovers_share_of_regional_misses": float(raw_good[~good].float().mean()),
                            "native_wins_by_at_least_5pp_AP_share": float((native[:, 0] - strong[:, 0] >= .05).float().mean()),
                            "oracle_native_or_regional": best.mean(0).tolist(), "oracle_gain": paired_delta(best, strong, y),
                            "top_confusions_when_native_was_correct": escaped, "native_kd_max_difference": raw_err,
                            "regional_original_mean_abs_difference": strong_sum / strong.numel(), "regional_original_max_abs_difference": strong_max,
                            "regional_reference_sha256": old["per_query_sha256"]}
            tensors[s] = {"query_labels": y, "raw_top_gallery_index": raw_top, "regional_top_gallery_index": strong_top,
                          "native_metrics": native, "strong_metrics": strong}
            print(s, summaries[s]["only_native_correct"], summaries[s]["regional_misses"], summaries[s]["oracle_gain"]["mAP"], flush=True)
    file = root / "native-region-disagreement.pt"
    save_tensor(tensors, file)
    atomic_json(root / "native-region-disagreement.json", {"kd_sha256": KD_FUNCTION_SHA, "per_query_sha256": checksum(file),
                "scope": "post-hoc oracle bound on selecting between the two existing rank lists, labels only used after scoring", "splits": summaries})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    run(p.parse_args().workspace)
