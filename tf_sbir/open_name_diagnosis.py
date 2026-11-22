"""Post-hoc localization of missing public-name evidence and a source-proposal oracle on Sketchy."""

import argparse
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import digest
from .cross_view_region_cache import FAMILY as REGIONAL
from .data import UNSEEN_CLASSES
from .foundation import load_words
from .regional_intervention_study import index as compact_index
from .semantic import read_words
from .retrieval_diagnosis import global_inputs
from .verification_inputs import get_inputs

FAMILY = "lexical-omission-recovery"


def per_term_ap(scores, labels, classes, split):
    order = scores.argsort(-1, descending=True, stable=True)
    k = 200 if split == "sketchy_2" else 100
    ranked = labels[order[:, :200] if split == "sketchy_2" else order]
    position = torch.arange(1, ranked.size(1) + 1, device=scores.device)
    result = scores.new_empty((len(scores), classes, 2))
    for c in range(classes):
        good = ranked == c
        ap = ((good.cumsum(1) / position) * good).sum(1) / good.sum(1).clamp_min(1)
        result[:, c, 0] = ap
        result[:, c, 1] = good[:, :k].float().mean(1)
    return result


def mean(x, mask):
    return float(x[mask].double().mean()) if mask.any() else None


@torch.inference_mode()
def run(w):
    cfg, _, qs, pp, gs, _, _ = global_inputs(w, "sketchy")
    words = read_words(cfg.vocabulary)
    text_src, text_tgt = (x.cuda() for x in load_words(cfg, words))
    _, scale = bare_words(cfg, words)
    record = {"scope": "diagnosis on Sketchy only. Public proposals are made before looking up any category. Oracle uses labels only after rankings exist.",
              "vocabulary": len(words), "vocabulary_sha256": checksum(cfg.vocabulary), "splits": {}}
    for s, (data, ycpu, gcpu) in get_inputs(w, "sketchy").items():
        y, g = ycpu.cuda(), gcpu.cuda()
        q = data["query_native"].cuda()
        photo = data["gallery_native"].cuda()
        base = data["regional_score"].cuda()
        regional_top = g[base.argmax(1)]
        fail = regional_top != y
        raw_top = g[(q @ photo.T).argmax(1)]
        strong_dir = w / "results" / REGIONAL
        locked = read(strong_dir / (s + ".json"))
        per_query = torch.load(strong_dir / (s + ".pt"), weights_only=True)
        base_metric = per_query["metrics"][locked["best_contrast"]].cuda()
        pi = gs[s][0]
        ident = digest({"signature": locked["signature"], "gallery": digest([str(pp[i]) for i in pi.tolist()]),
                        "vocabulary": checksum(cfg.vocabulary)})[:16]
        heads, _ = compact_index(w, s, photo, text_tgt, scale, ident)
        qids = qs[s][0]
        names = UNSEEN_CLASSES[s]
        exact = torch.tensor([words.index(n.replace("_", " ").casefold()) if n.replace("_"," ").casefold() in words else -1 for n in names],device=q.device)
        genuine = exact[y]
        rival = exact[regional_top]
        both_known = (genuine >= 0) & (rival >= 0)
        vocab_set = torch.zeros(len(words), dtype=torch.bool, device=q.device)
        vocab_set[heads] = True
        present = (genuine >= 0) & vocab_set[genuine.clamp_min(0)]
        views = torch.load(w / "cache" / REGIONAL / "sketchy/views.pt", weights_only=True, mmap=True)["original_regions"]
        top_candidates, observed = {}, {}
        for kind in ("whole", "regions"):
            proposed, query_stats = [], []
            for start in range(0, len(y), 96):
                loc = slice(start, start + 96)
                if kind == "whole":
                    response = (q[loc] @ text_src.T) * scale
                else:
                    z = (views[qids[loc]].cuda().float() @ text_src.T) * scale
                    response = torch.logaddexp(z[:, :9].logsumexp(1) - 2.1972245773362196, z[:, 9:].logsumexp(1) - 1.3862943611198906) - .6931471805599453
                proposed.append(response.topk(64).indices)
                ids = torch.arange(len(response), device=q.device)
                true_val = response[ids, genuine[loc].clamp_min(0)]
                wrong_val = response[ids, rival[loc].clamp_min(0)]
                rank = (response > true_val[:, None]).sum(1) + 1
                query_stats.append(torch.stack((rank.float(), true_val - wrong_val), 1))
            top_candidates[kind] = torch.cat(proposed)
            observed[kind] = torch.cat(query_stats)
        relevant = torch.cat(list(top_candidates.values())).unique()
        table = photo.new_empty((len(relevant), len(names), 2))
        for start in range(0, len(relevant), 72):
            loc = slice(start, start + 72)
            table[loc] = per_term_ap(text_tgt[relevant[loc]] @ photo.T, g, len(names), s)
        metrics = {}
        for kind, ids in top_candidates.items():
            positions = torch.searchsorted(relevant, ids)
            scores = table[positions, y[:, None]]
            per_name = {}
            for k in (1, 4, 16, 64):
                choice = scores[:, :k, 0].argmax(1)
                ix = torch.arange(len(y),device=q.device)
                best = scores[ix, choice]
                oracle_word = ids[ix, choice]
                per_name[str(k)] = {"oracle_single_word_map_precision": best.mean(0).tolist(),
                                    "improvement_vs_strong_on_its_rank1_misses_pp": (100 * (best - base_metric)[fail].mean(0)).tolist(),
                                    "share_oracle_query_word_outside_64_heads": float((~vocab_set[oracle_word]).float().mean()),
                                    "share_with_5pp_AP_advantage_over_strong": float((best[:,0] > base_metric[:,0] + .05).float().mean())}
            metrics[kind] = per_name
        slices = {"all": torch.ones_like(fail), "regional_hit": ~fail, "regional_miss": fail,
                  "regional_miss_native_correct": fail & (raw_top == y), "both_rank1_wrong": fail & (raw_top != y)}
        diagnosis = {}
        for name, mask in slices.items():
            valid = mask & both_known
            diagnosis[name] = {"n": int(mask.sum()), "both_class_names_in_public_vocabulary": int(valid.sum()),
                               "own_name_in_heads_when_in_public_dictionary": mean(present.float(), mask & (genuine>=0))}
            for kind, stats in observed.items():
                diagnosis[name][kind] = {"true_name_over_rival": mean((stats[:, 1] > 0).float(), valid & fail),
                                          "true_in_top16": mean((stats[:,0] <= 16).float(), mask & (genuine >= 0)),
                                          "true_in_top64": mean((stats[:,0] <= 64).float(), mask & (genuine >= 0))}
        record["splits"][s] = {"queries": len(y), "gallery": len(g), "category_exact_coverage": int((exact >= 0).sum()), "classes": len(names),
                               "category_exact_in_heads": int(((exact >= 0) & vocab_set[exact.clamp_min(0)]).sum()),
                               "distinct_proposed_names": len(relevant), "category_diagnosis": diagnosis, "single_name_oracle": metrics}
        print(s, record["splits"][s]["category_exact_in_heads"], diagnosis["regional_miss"], metrics["whole"]["16"], flush=True)
    atomic_json(w / "results" / FAMILY / "open-name-diagnosis.json", record)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
