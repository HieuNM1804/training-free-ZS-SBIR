"""Test whether words with identical decisions can substitute across depictions."""

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
from .interventional_concept_states import (
    collision_matrix, collision_reference, decision_questions, decisions, fixture,
    gallery_control, partitions, pooled_distribution,
)
from .joint_gallery_witnesses import load_reference
from .locked_geometry_words import score_rows
from .part_occlusion_observations import directory, public_words, read_cache, seed_job
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import BATCH, SKETCHY, Split, mean_percent, posteriors
from .semantic_boxable_single_prompt import encode_prompt
from .semantic_boxable_sparse import image_inputs


FAMILY = "interventional-concept-states"
ARMS = ("decisions", "size_only", "donor_partition", "additional_contexts", "query_partition", "photo_partition")


def check_prediction(whole, obs, state, text, labels):
    # Select word pairs before reading any double deletion. Match non-alias
    # control words in the same image to the same head word and rank range,
    # choosing the closest cosine to the positive pair in the public text bank.
    ids = torch.randperm(len(obs), generator=torch.Generator().manual_seed(SEED))[:512]
    positions = torch.arange(32, device=whole.device)
    bins = (positions.float()+1).log2().floor()
    same_band = bins[:, None] == bins[None, :]
    records = []
    _, context = decision_questions()
    for rows in ids.split(32):
        names = whole[rows].topk(32).indices
        trials = obs[rows].gather(2, names[:, None, :].expand(-1, 10, -1))
        held = decisions(trials, context)
        codes = state[rows].gather(1, names)
        vectors = text[names]
        similarity = vectors @ vectors.transpose(1, 2)
        totals = torch.zeros((len(rows), 5), device=whole.device, dtype=torch.float64)
        for i in range(31):
            eligible = positions > i
            same = codes[:, i:i+1] == codes
            target = same & eligible
            available = (~same & eligible)[:, None, :] & same_band[None, :, :]
            differences = (similarity[:, i, :, None] - similarity[:, i, None, :]).abs()
            gaps, controls = differences.masked_fill(~available, torch.inf).min(2)
            valid = target & gaps.isfinite()
            held_accuracy = (held == held[:, :, i:i+1]).double().mean(1)
            paired = held_accuracy.gather(1, controls)
            totals += torch.stack((valid.sum(1), (held_accuracy*valid).sum(1), (paired*valid).sum(1),
                                   gaps.masked_fill(~valid, 0.).sum(1), target.sum(1)), dim=1)
        records.append(totals.cpu())
    n, correct, null, gap, eligible = torch.cat(records).T
    active = n > 0
    return {
        "sample_images": len(ids), "images_with_comparator": int(active.sum()),
        "alias_pairs": int(eligible.sum()), "alias_pairs_with_control": int(n.sum()),
        "mean_text_cosine_gap": float(gap.sum()/n.sum()) if active.any() else None,
        "held_context_agreement_percent": [float(100*success.sum()/n.sum()) for success in (correct, null)]
            if active.any() else None,
        "difference_by_class": class_interval(100*(correct[active]-null[active])/n[active], labels[ids][active], seed=SEED),
    }


def official(scores, split, replay, pool, mask_ids):
    gallery, truth = split.photo_labels.cuda(), split.sketch_labels.cuda()
    blocks, flags, hits, prefix_ids = [], [], [], []
    for start in range(0, len(scores), BATCH):
        sl = slice(start, min(start+BATCH, len(scores)))
        batch = scores[sl]
        if not torch.isfinite(batch).all():
            raise ValueError("Non-finite score")
        result = score_rows(batch, split.sketch_labels[sl], gallery, split.name, replay, pool)
        ids = prefix(batch, *replay.cutoffs)
        blocks.append(result)
        prefix_ids.append(ids.cpu())
        if mask_ids is not None:
            baseline = torch.zeros_like(batch, dtype=torch.bool).scatter_(1, mask_ids[sl].cuda(), True)
            new, labels = newcomer_hits(ids, baseline, gallery, truth[sl], result[:, 1])
            flags.append(new.cpu())
            hits.append(labels.cpu())
    status = (torch.cat(flags), torch.cat(hits)) if flags else None
    return torch.cat(blocks).double(), status, torch.cat(prefix_ids)


def describe_alias(q, p, sq, sp, words):
    q, p, sq, sp = (v.cpu() for v in (q, p, sq, sp))
    cell = (sq[:, None] == sq) & (sp[:, None] == sp)
    off = (q[:, None]*p[None, :])*cell.fill_diagonal_(False)
    best = int(off.argmax())
    i, j = divmod(best, len(q))
    return {"literal": float(q @ p), "extra": float(off.sum()),
            "largest_distinct_name_pair": [words[i], words[j]], "mass": float(off[i, j])}


def evaluate(w, split, observations, text, words, signature, kd, pool, paths):
    root = w/"results"/FAMILY
    target = root/(split.name+".json")
    artifact = target.with_suffix(".pt")
    if target.exists():
        previous = json.loads(target.read_bytes())
        if previous["protocol"] != signature or checksum(artifact) != previous["per_query_sha256"]:
            raise ValueError("Unverified saved result")
        return
    archive, _ = load_reference(split)
    queries, gallery = posteriors(split, text, [DEFAULT_PROMPTS])
    q, p = queries[DEFAULT_PROMPTS[0]], gallery[DEFAULT_PROMPTS[1]].T.contiguous()
    qo = observations[0][split.query_ids].cuda()
    po = observations[1][split.gallery_ids].cuda()
    qs, qfine = partitions(qo)
    ps, pfine = partitions(po)
    validation = {
        "sketch": check_prediction(q, qo, qs, text[DEFAULT_PROMPTS[0]], split.sketch_labels),
        "photo": check_prediction(p, po, ps, text[DEFAULT_PROMPTS[1]], split.photo_labels),
    }
    del qo, po
    replay = PrefixReplay(kd, split.photo_labels, split.name)
    rows, matched, times = {"dot": archive["dot"]}, {}, {}
    literal_bank = gallery[DEFAULT_PROMPTS[1]]
    base_ids = torch.cat([prefix(q[s:s+BATCH] @ literal_bank, *replay.cutoffs).cpu() for s in range(0, len(q), BATCH)])
    base_hit = split.photo_labels[base_ids] == split.sketch_labels[:, None]
    if not torch.allclose(base_hit.float().mean(1), rows["dot"][:, 1].float(), atol=2e-6, rtol=0):
        raise ValueError("Did not reproduce Dot precision")
    if not torch.equal(base_hit[:, 0], rows["dot"][:, 3].bool()):
        raise ValueError("Did not reproduce Dot top-1")
    donors = gallery_control(p)
    primary_status, primary_ids, opportunity = None, None, None
    numerical = 0.
    for name in ARMS:
        tick = time.perf_counter()
        if name == "decisions":
            score, opportunity = collision_matrix(q, p, qs, ps, count_opportunities=True)
            for i, j in ((0, 0), (1, 17), (2, 127), (3, 1024), (5, 2048), (13, 3900)):
                exact, chance = collision_reference(q[i].double(), p[j].double(), qs[i], ps[j])
                numerical = max(numerical, abs(float(score[i,j])-float(exact)), abs(float(opportunity[i,j])-float(chance)))
            if numerical > 2e-5:
                raise ValueError("Lifted result differs from directly pooling the joint classes")
        elif name == "size_only":
            score, opportunity = opportunity, None
        elif name == "donor_partition":
            score, _ = collision_matrix(q, p, qs, ps[donors])
        elif name == "additional_contexts":
            score, _ = collision_matrix(q, p, qfine, pfine)
        elif name == "query_partition":
            score = pooled_distribution(q, qs) @ p.T
        elif name == "photo_partition":
            score = q @ pooled_distribution(p, ps).T
        torch.cuda.synchronize()
        times[name] = time.perf_counter()-tick
        result, status, top = official(score, split, replay, pool, base_ids)
        rows[name] = result
        if name == "decisions":
            primary_status, primary_ids = status, top
        else:
            matched[name] = matched_counts(primary_status, status)
        del score
        print(split.name, name, mean_percent(result[:, :2]), "score seconds", round(times[name], 2), flush=True)

    # Print one concrete failure and repair using archive ordering. The method never reads these IDs/labels.
    base_correct = rows["dot"][:, 3].bool()
    primary_correct = rows["decisions"][:, 3].bool()
    examples = {}
    for event, keep in (("repair", ~base_correct & primary_correct), ("damage", base_correct & ~primary_correct)):
        query_rows = keep.nonzero().flatten()
        if len(query_rows):
            i = int(query_rows[0])
            file = Path(paths[0][split.query_ids[i]])
            example = {"query": str(Path(file.parent.name)/file.name)}
            for arm, rank in (("dot", base_ids), ("decisions", primary_ids)):
                j = int(rank[i, 0])
                photo = Path(paths[1][split.gallery_ids[j]])
                example[arm] = {"photo": str(Path(photo.parent.name)/photo.name),
                                "correct_class": bool(split.photo_labels[j] == split.sketch_labels[i]),
                                **describe_alias(q[i], p[j], qs[i], ps[j], words)}
            examples[event] = example
    save_tensor({"qi": split.query_ids, "pi": split.gallery_ids, "labels": split.sketch_labels,
                 "rows": rows, "matched": matched}, artifact)
    table = {
        "protocol": signature, "queries": len(q), "gallery": len(p), "classes": len(split.sketch_labels.unique()),
        "cutoffs": list(replay.cutoffs), "metrics": {k: mean_percent(v[:, :2]) for k,v in rows.items()},
        "primary_minus_macro_ap": {arm: class_interval(
            100*(rows["decisions"][:, 0]-value[:, 0]), split.sketch_labels, seed=SEED,
        ) for arm,value in rows.items() if arm != "decisions"},
        "matched_promotions": {arm: summarize(value, split.sketch_labels) for arm,value in matched.items()},
        "top1_repairs_and_damage": [int((~base_correct & primary_correct).sum()), int((base_correct & ~primary_correct).sum())],
        "prediction_on_withheld_contexts": validation, "examples": examples,
        "gallery_control_movable": int((donors != torch.arange(len(p),device=p.device)).sum()),
        "scoring_seconds": times, "max_error_vs_explicit_states": numerical, "per_query_sha256": checksum(artifact),
    }
    atomic_json(target, table)
    print(split.name, "verified", table["metrics"], flush=True)


@torch.inference_mode()
def run(dataset):
    seed_job()
    w = Path("/content/sbir")
    root = w/"results"/FAMILY
    words, vocab_sha = public_words()
    kd, _ = load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("KD evaluator changed")
    lock = save_lock(root/"freeze.json", {
        "commit": revision(), "evaluator_sha256": KD_FUNCTION_SHA, "clip_sha256": CHECKPOINT,
        "vocabulary_sha256": vocab_sha, "prompts": list(DEFAULT_PROMPTS), "seed": SEED,
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "arms": list(ARMS),
        "grouping": "tie-preserving Bayes decisions for all six binary single-deletion experiments",
        "test": "unseen shared-deletion contexts, cardinality-matched chance, partitions from a similar native photo",
        "selection": "primary comparison fixed before scoring Sketchy or TU; no tuning",
        "fixture": fixture(torch.device("cuda")),
    })
    names = SKETCHY if dataset == "sketchy" else ("tuberlin",)
    workers = min(4, len(os.sched_getaffinity(0)))
    cfg = runtime(w, dataset, workers=workers)
    clip = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    tok = open_clip.get_tokenizer("ViT-B-32")
    text = {s: encode_prompt(clip, tok, words, s) for s in DEFAULT_PROMPTS}
    data = image_inputs(cfg, names, clip)
    banks, paths = [], []
    for mode in ("sketch", "photo"):
        items, _ = merge_inputs(cfg.root, names, mode)
        folder, spec = directory(cfg, dataset, mode, items, vocab_sha)
        if json.loads((folder/"complete.json").read_bytes())["specification"] != spec:
            raise ValueError("Masked features differ")
        banks.append(read_cache(folder))
        paths.append(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name in names:
            split = Split.from_inputs(data, name)
            evaluate(w, split, banks, text, words, lock, kd, pool, paths)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=("sketchy", "tuberlin"))
    run(p.parse_args().dataset)
