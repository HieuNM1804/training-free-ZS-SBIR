"""Check whether the fixed word-map mistakes were already present in frozen visual CLIP."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json

import torch
from torch.nn.functional import normalize

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json
from .conditional_completion_study import evaluate
from .kaggle_unseen_features import DEFAULT_CACHE, load_split
from .native_block_probe import SEED, deterministic
from .replicated_simplex_study import evaluator
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import mean_percent
from .semantic_disputed_word_risk import query_posteriors
from .semantic_operator_audit import DST, design, word_fingerprint
from .semantic_operator_overlap import class_percentiles, common_positions, source_leverage


@torch.inference_mode()
def run(revision):
    source, _, _ = load_split(DEFAULT_CACHE, "sketchy_1")
    target, text, words = load_split(DEFAULT_CACHE, "sketchy_2")
    _, qi = common_positions(source.query_ids, target.query_ids)
    _, pi = common_positions(source.gallery_ids, target.gallery_ids)
    indices = [torch.load(DST / f"index-{name}.pt", mmap=True, weights_only=True) for name in (source.name, target.name)]
    for idx, split in zip(indices, (source, target), strict=True):
        if idx["word_fingerprint"] != word_fingerprint(words) or not torch.equal(idx["gallery_ids"], split.gallery_ids):
            raise ValueError("Precomputed gallery index does not match its frozen inputs")
    q0, q1 = target.sketch[qi].float(), target.mirror[qi].float()
    visual_q = normalize(normalize(q0, dim=1) + normalize(q1, dim=1), dim=1)
    visual_g = normalize(target.photo.float(), dim=1)
    word_q = torch.cat([query_posteriors(a, b, text[DEFAULT_PROMPTS[0]], target.scale)
                        for a,b in zip(q0.split(384), q1.split(384), strict=True)])
    p = indices[1]["p"].cuda()
    fixed = design(p) @ indices[0]["a"].cuda()
    classes = target.photo_labels.cuda()
    truth = target.sketch_labels[qi].cuda()
    added = torch.ones(len(p), device=p.device, dtype=torch.bool)
    added[pi] = False
    if torch.isin(classes[added], truth.unique()).any():
        raise ValueError("The added pool contains a positive photo")
    within = class_percentiles(source_leverage(indices[0]["p"].cuda(), p), classes)
    conditions = {"native_visual": (visual_q, visual_g), "dot": (word_q, p.float()),
                  "fixed_ext1_query_edit": (word_q, fixed.float())}
    heads, stats = {}, {}
    for key, (query, gallery) in conditions.items():
        first, best_added = [], []
        bank = gallery.T.contiguous()
        for part in query.split(64):
            scores = part @ bank
            first.append(scores.argmax(1))
            best_added.append(scores.masked_fill(~added[None, :], -torch.inf).argmax(1))
        heads[key] = torch.cat(first)
        stats[key] = {"top1_correct_fraction": (classes[heads[key]] == truth).double().mean().item(),
                      "top1_added_fraction": added[heads[key]].double().mean().item(),
                      "best_added_mean_within_class_leverage_percentile": within[torch.cat(best_added)].mean().item()}
    fixed_head = heads["fixed_ext1_query_edit"]
    event = added[fixed_head]
    for name, head in heads.items():
        stats[name].update({"on_fixed_added_events_top1_correct_fraction": (classes[head[event]] == truth[event]).double().mean().item(),
                            "on_fixed_added_events_exact_same_photo_fraction": (head[event] == fixed_head[event]).double().mean().item()})
    historical = json.loads((DST / "sketchy-overlap.json").read_bytes())
    for key in ("dot", "fixed_ext1_query_edit"):
        gold = historical["added_distractors_for_shared_queries"][key]["fraction_of_queries_with_added_top1"]
        if abs(gold - stats[key]["top1_added_fraction"]) > 1e-9:
            raise ValueError("Failed to replay the previously archived ranking on identical queries")
    kd = evaluator(DEFAULT_CACHE.parent / "real-sketchy-smoke" / "evaluator.py")
    native_metrics = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, positions in (("shared_gallery", pi), ("full_ext2_gallery", torch.arange(len(p)))):
            sample = replace(target, sketch_labels=truth.cpu(), photo_labels=target.photo_labels[positions])
            replay = PrefixReplay(kd, sample.photo_labels, sample.name)
            rows = evaluate(sample, visual_q, visual_g[positions], replay, pool)
            native_metrics[name] = mean_percent(rows)
    out = {"revision": revision, "seed": SEED, "device": torch.cuda.get_device_name(), "queries": len(qi),
           "shared_photos": len(pi), "ext2_photos": len(p), "number_of_fixed_added_top1_events": int(event.sum()),
           "all_conditions": stats, "native_mAP200_P200": native_metrics, "overlap_revision": historical["revision"]}
    atomic_json(DST / "sketchy-native-control.json", out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    deterministic()
    run(parser.parse_args().revision)
