"""Replay every frozen ranking through the original KD-SBIR function on Colab."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json, read
from .data import UNSEEN_CLASSES
from .foundation import load_words
from .metrics import cosine_batches, evaluate_batch
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import distribution, load_kd, tensor, verify_locks, within_class_percentile
from .reverse_prompt import PhotoIndex, Setting, nearest_words
from .semantic import read_words


def text_coverage(words, names, ids, labels):
    # Match only literal class names, treating dataset filename separators as spaces.
    # This diagnostic is performed after scoring and never changes selected words.
    def plain(word):
        return " ".join(word.casefold().replace("_", " ").replace("-", " ").split())

    lookup = {plain(name): i for i, name in enumerate(names)}
    dictionary = torch.tensor([lookup.get(plain(word), -1) for word in words])
    covered = sorted(set(dictionary.tolist()) - {-1})
    mask = torch.isin(labels, torch.tensor(covered, dtype=labels.dtype))
    correct = dictionary[ids.cpu()].eq(labels[:, None]).any(1)
    return {"classes_with_exact_public_name": [names[i] for i in covered],
            "covered_items": int(mask.sum()), "items": len(labels),
            "literal_hit_covered": float(correct[mask].float().mean()) if mask.any() else None,
            "literal_hit_all": float(correct.float().mean())}


def replay(kd, pool, rows, ql, pl, reference, split):
    # Every gallery rank has a unique, monotonically decreasing cosine value.
    # Only the order of gallery labels changes between queries.
    fixed = torch.linspace(.75, -.75, len(pl))
    gallery = torch.stack((fixed, (1. - fixed.square()).sqrt()), 1)
    query = torch.tensor([[1., 0.]])
    rank_values = torch.linspace(.9, -.9, len(pl))
    map_k, p_k = (200, 200) if split == "sketchy_2" else (0, 100)
    def evaluate(item):
        label, permutation = item
        ap, precision, _, _ = kd(query, gallery, label.view(1), pl[permutation], split)
        score = torch.empty_like(rank_values).scatter_(0, permutation, rank_values)
        evaluator = evaluate_batch([(score, label)], pl, map_k, p_k)[0]
        return torch.stack((ap, precision)), evaluator

    start = 0
    verified, top = [], []
    for block in rows:
        order = block.argsort(dim=1, descending=True, stable=True).cpu()
        labels = ql[start:start + len(block)]
        top.append(order[:, :10].clone())
        verified.extend(pool.map(evaluate, zip(labels, order, strict=True)))
        start += len(block)
    kd_rows, evaluator_rows = zip(*verified, strict=True)
    observed = torch.stack(kd_rows)
    same_ranking_difference = (torch.stack(evaluator_rows) - observed).abs()
    if observed.shape != reference.shape:
        raise ValueError("Replay omitted sketches")
    abs_difference = (observed - reference).abs()
    return {"queries": len(ql), "mAP": float(observed[:, 0].mean()), "precision": float(observed[:, 1].mean()),
            "kd_minus_pipeline_max_abs_same_order_AP_P": same_ranking_difference.max(dim=0).values.tolist(),
            "max_per_query_abs_AP_P": abs_difference.max(dim=0).values.tolist(),
            "mean_per_query_abs_AP_P": abs_difference.mean(dim=0).tolist(),
            "queries_within_1e-6_both": int((abs_difference.max(dim=1).values <= 1e-6).sum())}, torch.cat(top)


def native_top_ids(queries, photos):
    top = []
    for block in cosine_batches(queries, photos, 4):
        score = ((block + 1.) * .5).clamp(min=torch.finfo(block.dtype).eps, max=1.)
        top.append(score.argsort(dim=1, descending=True, stable=True)[:, :10].cpu().clone())
    return torch.cat(top)


@torch.inference_mode()
def run(workspace):
    results = workspace / "results"
    joint = verify_locks(results)
    family = joint["primary"]
    if family != "reverse-prompt-diffusion":
        raise ValueError("This script accepts reverse-prompt index scores")
    root = results / family
    selected = Setting(**read(root / "freeze.json")["winner"])
    kd, sha = load_kd(workspace)
    audit = {"kd_source_sha256": sha, "primary": selected.name, "splits": {}}
    torch.set_num_threads(1)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for dataset in ("sketchy", "tuberlin"):
            args, _, queries, _, galleries, rgb_q, rgb_g = global_inputs(workspace, dataset)
            words = read_words(args.vocabulary)
            source, target = (w.cuda() for w in load_words(args, words))
            for split, (qi, ql) in queries.items():
                pi, pl = galleries[split]
                q = rgb_q[qi].cuda()
                index = PhotoIndex(rgb_g[pi].cuda(), source, target)
                sub = "heldout" if split == "tuberlin" else "sketchy"
                reference = tensor(root / sub / "per-query" / f"{split}-{selected.name}.pt")
                metrics, top = replay(kd, pool, index.scores(q, selected), ql, pl, reference, split)
                native_top = native_top_ids(q, index.rgb)
                qwords, _ = nearest_words(q, source, selected.terms)
                names = UNSEEN_CLASSES[split]
                axis = F.normalize((source - target).mean(0), dim=0)
                percentile = within_class_percentile((index.rgb @ axis).cpu(), pl)
                old, new = pl[native_top[:, 0]] == ql, pl[top[:, 0]] == ql
                audit["splits"][split] = {"kd_full_replay": metrics,
                    "stable_gallery_order": {
                        "native": distribution(native_top, ql, pl, names, percentile),
                        "method": distribution(top, ql, pl, names, percentile),
                        "top1_rescued": int((new & ~old).sum()), "top1_regressed": int((old & ~new).sum())},
                    "literal_class_name_in_first_eight_terms": {
                        "photo": text_coverage(words, names, index.ids[:, :selected.terms], pl),
                        "sketch": text_coverage(words, names, qwords, ql)}}
                print(split, metrics, flush=True)
                del index
    atomic_json(results / "inquiry-report/verified-rankings.json", audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
