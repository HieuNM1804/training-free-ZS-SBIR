"""Choose a zoom before observing it, and compare against the exact random-action reference."""

import argparse
from collections import defaultdict
import math
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import SIZES, bare_words
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT
from .cross_view_region_cache import FAMILY as VIEW_FAMILY
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .gallery_region_actions import photo_channels, utilities, group_prior, within_scale_mismatch, concordance
from .map_diagnosis import map_values
from .partial_object_parts import vocabulary
from .regional_intervention_study import index as compact_index
from .retrieval_diagnosis import global_inputs, stratified_sample
from .semantic import read_words

FAMILY = "gallery-directed-region-acquisition"
TEMPERATURES = (.5, 1., 2.)
RIVALS = (4, 12)
WEIGHTS = (.2, .4, .7)
CHOICES = ("heldout", "within", "information", "fixed", "misplaced", "stop")


def single_region_scores(native, source, heads, compact, terms, channels, membership, scale):
    words = native @ source[heads].T * scale
    shift = words.amax((1, 2), keepdim=True)
    likelihood = ((words - shift).exp() @ compact.T).clamp_min(1e-30).log() + shift
    soft = (native @ source[terms].T * scale).softmax(-1)
    by_group = torch.einsum("brw,grw->brg", soft, channels.mean(0)).clamp_min(1e-30).log()
    return {"photo": likelihood, "group": by_group[:, :, membership]}


def counterfactual_metrics(readouts, full, y, g, split, weights):
    out = {}
    for reader, local in readouts.items():
        for mix in weights:
            scores = (1. - mix) * full[:, None] + mix * local
            results = [map_values(scores[:, r], y, g, split)[0] for r in range(scores.size(1))]
            out[reader, mix] = torch.stack(results, 1)
    return out


def append(target, name, values):
    target[name].append(values.detach().cpu())


def cell(policy, temperature, rivals, reader, mix):
    return f"{policy}/{temperature:g}/{rivals}/{reader}/{mix:g}"


@torch.inference_mode()
def run(w):
    folder = w / "results" / FAMILY
    rev = revision()
    spec = {"revision": rev, "checkpoint": CHECKPOINT, "seed": 63241, "per_class": 36, "groups": 64,
            "public_words": 20931, "channel_words": 128, "rivals": RIVALS, "temperatures": TEMPERATURES,
            "weights": WEIGHTS, "primary": "heldout", "policies": CHOICES, "phase": "balanced-pilot-only"}
    sig = digest(spec)
    if (folder / "protocol.json").exists() and read(folder / "protocol.json")["sha256"] != sig:
        raise ValueError("Incompatible pilot record")
    atomic_json(folder / "protocol.json", spec | {"sha256": sig})
    cfg, _, qs, paths, galleries, _, raw = global_inputs(w, "sketchy")
    words = read_words(cfg.vocabulary)
    if len(words) != spec["public_words"]:
        raise ValueError("Unexpected public vocabulary")
    source, target = (x.cuda() for x in load_words(cfg, words))
    _, scale = bare_words(cfg, words)
    qviews = torch.load(w / "cache" / VIEW_FAMILY / "sketchy/views.pt", mmap=True, weights_only=True)["original_regions"]
    pviews = torch.load(w / "cache/nested-scale-word-evidence/sketchy/photo.pt", mmap=True, weights_only=True)
    means = {}
    for split, (qi, y) in qs.items():
        pi, g = galleries[split]
        if (len(qi), len(pi), len(g.unique())) != SIZES[split] or digest(UNSEEN_CLASSES[split]) != SPLIT_HASHES[split]:
            raise ValueError("Unexpected split")
        output = folder / ("pilot-" + split + ".json")
        if output.exists():
            prev = read(output)
            if prev["signature"] != sig or checksum(output.with_suffix(".pt")) != prev["per_query_sha256"]:
                raise ValueError("Inconsistent result checkpoint")
            means[split] = prev["values"]
            continue
        photos = raw[pi].cuda()
        zooms = pviews[pi]
        ident = digest({"gallery": digest([str(paths[j]) for j in pi.tolist()]), "revision": rev})[:16]
        heads, compact = compact_index(w, split, photos, target, scale, ident)
        terms = vocabulary(zooms, target, scale, spec["channel_words"])
        assignment, channels = photo_channels(photos, zooms, target[terms], scale)
        n_groups = channels.size(1)
        mean_err = float((channels.sum(-1) - 1).abs().max())
        some = torch.arange(4, device=photos.device)[None]
        flat = channels[:, :1].expand(-1, 4, -1, -1)
        null = utilities(photos.new_ones((1, 4)), some, flat)
        null_error = max(float(x.abs().max()) for x in null.values())
        if max(null_error, mean_err) > 3e-5:
            raise ValueError("Observation channel normalization or equal-group identity failed")
        uniform = photos.new_ones((1, n_groups))
        fixed = utilities(uniform, torch.arange(n_groups, device=photos.device)[None], channels)["heldout"].argmax(1)
        history = torch.load(w / "cache/pragmatic-gallery-competition" / (split + "-named-rgb16.pt"), mmap=True, weights_only=True)
        if history.shape != (len(qi), len(pi)):
            raise ValueError("Global baseline row mismatch")
        ids = stratified_sample(y, spec["per_class"], spec["seed"])
        gallery_labels = g.cuda()
        scores, candidate_actions, diagnostics = defaultdict(list), defaultdict(list), defaultdict(list)
        region_prior = photos.new_tensor([1 / 18] * 9 + [1 / 8] * 4)
        max_over_optimistic = 0.
        for part in ids.split(42):
            full = history[part].cuda().clamp_min(1e-30).log()
            labels = y[part].cuda()
            base, _, relevance, _ = map_values(full, labels, gallery_labels, split)
            append(scores, "global", base)
            append(diagnostics, "base_top1_correct", relevance[:, 0])
            batch = torch.arange(len(part), device=full.device)
            selected = {}
            # No sketch crop is accessed until all actions are decided.
            for temp in TEMPERATURES:
                for rivals in RIVALS:
                    prior, groups = group_prior(full, assignment, temp, rivals)
                    util = utilities(prior, groups, channels)
                    max_over_optimistic = max(max_over_optimistic, float((util["heldout"] - util["within"]).amax()))
                    chosen = {k: v.argmax(1) for k, v in util.items()}
                    chosen["fixed"] = fixed.expand(len(part))
                    chosen["misplaced"] = within_scale_mismatch(chosen["heldout"])
                    chosen["stop"] = chosen["heldout"]
                    positive = util["heldout"].amax(1) > 1e-7
                    selected[temp, rivals] = (chosen, util["heldout"], positive)
                    for k, ix in chosen.items():
                        append(candidate_actions, f"{k}/{temp:g}/{rivals}", ix)
                    append(diagnostics, f"positive/{temp:g}/{rivals}", positive)
            native = qviews[qi[part]].cuda().float()
            local = single_region_scores(native, source, heads, compact, terms, channels, assignment, scale)
            all_metrics = counterfactual_metrics(local, full, labels, gallery_labels, split, WEIGHTS)
            pooled = torch.logsumexp(local["photo"] + region_prior.log()[None, :, None], 1)
            for mix in WEIGHTS:
                append(scores, f"all13/photo/{mix:g}", map_values((1. - mix) * full + mix * pooled, labels, gallery_labels, split)[0])
            for (reader, mix), observed in all_metrics.items():
                append(scores, f"random/{reader}/{mix:g}", (observed * region_prior[None, :, None]).sum(1))
                oracle = observed[:, :, 0].argmax(1)
                append(scores, f"oracle/{reader}/{mix:g}", observed[batch, oracle])
                for (temp, rivals), (actions, values, positive) in selected.items():
                    for policy, action in actions.items():
                        measured = observed[batch, action]
                        if policy == "stop":
                            measured = torch.where(positive[:, None], measured, base)
                        append(scores, cell(policy, temp, rivals, reader, mix), measured)
                    if reader == "photo" and mix == .4:
                        rate, n = concordance(values, observed[:, :, 0])
                        append(diagnostics, f"pair_agreement/{temp:g}/{rivals}", torch.stack((rate, n.float()), 1))
        if max_over_optimistic > 2e-5:
            raise ValueError("Cross-panel expected gain exceeded within-panel optimism")
        result = {k: torch.cat(v) for k, v in scores.items()}
        actions = {k: torch.cat(v) for k, v in candidate_actions.items()}
        info = {k: torch.cat(v) for k, v in diagnostics.items()}
        vals = {k: value.mean(0).tolist() for k, value in result.items()}
        save_tensor({"metrics": result, "query_ids": ids, "query_labels": y[ids], "actions": actions, "diagnostics": info}, output.with_suffix(".pt"))
        summary = {"signature": sig, "values": vals, "queries": len(ids), "gallery": len(pi), "classes": len(g.unique()),
                   "gallery_fixed_window": int(fixed), "null_error": null_error, "max_channel_error": mean_err,
                   "max_cross_above_within": max_over_optimistic, "per_query_sha256": checksum(output.with_suffix(".pt")),
                   "action_counts": {k: v.bincount(minlength=13).tolist() for k, v in actions.items()},
                   "pair_agreement": {k: float(v[v[:, 1] > 0, 0].mean()) for k, v in info.items() if k.startswith("pair_agreement/")}}
        atomic_json(output, summary)
        means[split] = vals
        primary = max((k for k in vals if k.startswith("heldout/")), key=vals.get)
        print(split, primary, vals[primary], "global", vals["global"], flush=True)
    by_setting = {k: [sum(means[s][k][i] for s in qs) / len(qs) for i in (0, 1)] for k in means["sketchy_1"]}
    atomic_json(folder / "pilot.json", {"splits": means, "means": by_setting})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
