"""Use conditional log score changes of a sketch zoom to discriminate photo groups."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import SIZES, bare_words
from .conditional_zoom_model import build_bank, predict_observation, signature
from .crossdomain_benchmark import digest, revision, SPLIT_HASHES
from .crossdomain_inputs import CHECKPOINT
from .cross_view_region_cache import FAMILY as ZOOM
from .cross_view_regions import CandidateRegions
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .partial_object_parts import vocabulary
from .region_preview_cache import FAMILY as PREVIEW
from .regional_intervention_study import index as compact_index
from .retrieval_diagnosis import global_inputs, stratified_sample
from .semantic import read_words

FAMILY = "conditional-zoom-innovation"
TEMPERATURES = (8., 20.)
STRENGTHS = (.25, 1., 4., 12.)


def evidence(high, low, log_score, region_weights):
    return torch.einsum("brw,fbgrw,r->bg", high - low, log_score, region_weights) / log_score.size(0)


def base_score(raw, flipped, high, reverse, src, heads, compact, full, scale):
    regions = CandidateRegions(raw, flipped, high, reverse, src, heads, compact, src.new_empty((0, len(src))), full)
    regions.prepare(scale)
    return .65 * regions.first + .35 * regions.read("self", 1., "gallery")


def innovations(qpreview, qfull, zoom, terms, scale, groups, bank):
    query = signature(qpreview, qfull, terms)
    prediction = predict_observation(query, groups, bank, TEMPERATURES)
    unconditional = prediction["unconditional", 0.]
    unconditional = unconditional / unconditional.sum(-1, keepdim=True).clamp_min(1e-20)
    high = (zoom @ terms.T * scale).softmax(-1)
    low = (qpreview @ terms.T * scale).softmax(-1)
    region_weights = terms.new_tensor([1 / 18] * 9 + [1 / 8] * 4)
    background_log = unconditional.clamp_min(1e-20).log()
    scores = {"unconditional_change": evidence(high, low, background_log, region_weights)}
    null_error = 0.
    zero = torch.zeros_like(low)
    for t in TEMPERATURES:
        n = bank[-1].sum(-1)[:, groups]
        for kind in ("aligned", "broken_pair", "wrong_preview"):
            ess_fraction = prediction["ess_wrong" if kind == "wrong_preview" else "ess", t]
            ess = (ess_fraction * n[..., None])[..., None]
            cond = (ess * prediction[kind, t] + unconditional) / (ess + 1)
            cond /= cond.sum(-1, keepdim=True).clamp_min(1e-20)
            cond_log = cond.clamp_min(1e-20).log()
            ratio = cond_log - background_log
            scores[f"{kind}/{t:g}"] = evidence(high, low, ratio, region_weights)
            if kind == "aligned":
                scores[f"new_words_only/{t:g}"] = evidence(high, zero, ratio, region_weights)
                scores[f"absolute_score/{t:g}"] = evidence(high, zero, cond_log, region_weights)
                scores[f"reverse/{t:g}"] = -scores[f"aligned/{t:g}"]
                null_error = max(null_error, float(evidence(low, low, ratio, region_weights).abs().max()))
    return scores, null_error


def rival_margin(group_score, membership, gallery_labels, truth, wrong_class):
    by_class = group_score.new_zeros((len(truth), int(gallery_labels.max()) + 1))
    by_class.index_add_(1, gallery_labels, group_score[:, membership])
    by_class /= gallery_labels.bincount()
    batch = torch.arange(len(truth), device=truth.device)
    return by_class[batch, truth] - by_class[batch, wrong_class]


@torch.inference_mode()
def run(w):
    root = w / "results" / FAMILY
    cfg, _, qs, paths, galleries, qraw, praw = global_inputs(w, "sketchy")
    words = read_words(cfg.vocabulary)
    if len(words) != 20931:
        raise ValueError("Public vocabulary mismatch")
    src, tgt = (x.cuda() for x in load_words(cfg, words))
    _, scale = bare_words(cfg, words)
    pd = w / "cache" / PREVIEW / "sketchy"
    previews = {d: torch.load(pd / (d + ".pt"), weights_only=True, mmap=True) for d in ("photo", "sketch")}
    qviews = torch.load(w / "cache" / ZOOM / "sketchy/views.pt", weights_only=True, mmap=True)
    pviews = torch.load(w / "cache/nested-scale-word-evidence/sketchy/photo.pt", weights_only=True, mmap=True)
    spec = {"revision": revision(), "base": "cross-view-region-verification/locked-self", "seed": 63241,
            "samples_per_class": 36, "temperatures": TEMPERATURES, "strengths": STRENGTHS, "primary": "aligned",
            "text_names": 20931, "retained_words": 128, "panel_cap_per_group": 96, "checkpoint": CHECKPOINT}
    sig = digest(spec)
    if (root / "protocol.json").exists() and read(root / "protocol.json")["sha256"] != sig:
        raise ValueError("Study signature changed")
    atomic_json(root / "protocol.json", spec | {"sha256": sig})
    table = {}
    for s, (qi, y) in qs.items():
        pi, g = galleries[s]
        if (len(qi), len(pi), len(g.unique())) != SIZES[s] or digest(UNSEEN_CLASSES[s]) != SPLIT_HASHES[s]:
            raise ValueError("Evaluation split mismatch")
        dest = root / ("pilot-" + s + ".json")
        if dest.exists():
            saved = read(dest)
            if saved["signature"] != sig or saved["per_query_sha256"] != checksum(dest.with_suffix(".pt")):
                raise ValueError("Stale pilot")
            table[s] = saved["values"]
            continue
        ident = digest({"gallery": digest([str(paths[i]) for i in pi.tolist()]), "revision": "174579185794aca151e95a9309cf2b49f32c0fcb"})[:16]
        photo = praw[pi].cuda()
        heads, compact = compact_index(w, s, photo, tgt, scale, ident)
        crops = pviews[pi]
        terms = vocabulary(crops, tgt, scale, 128)
        membership, *bank = build_bank(photo, previews["photo"][pi], crops, tgt[terms], scale)
        ids = stratified_sample(y, 36, 63241)
        g = g.cuda()
        hist = torch.load(w / "cache/pragmatic-gallery-competition" / (s + "-named-rgb16.pt"), weights_only=True, mmap=True)
        if hist.shape != (len(qi), len(pi)):
            raise ValueError("Different baseline order")
        rows, info = defaultdict(list), defaultdict(list)
        neutral = 0.
        for part in ids.split(8):
            at = qi[part]
            h = qviews["original_regions"][at].cuda().float()
            rev = qviews["regions"][at].cuda().float()
            raw = qraw[at].cuda()
            base = base_score(raw, qviews["global"][at].cuda(), h, rev, src, heads, compact, hist[part].cuda(), scale)
            all_groups = torch.arange(bank[0].size(1), device=raw.device).expand(len(part), -1)
            scores, null = innovations(previews["sketch"][at].cuda().float(), raw, h, src[terms], scale, all_groups, bank)
            neutral = max(neutral, null)
            if neutral != 0.:
                raise ValueError("Identical coarse and high word readings should give zero")
            labels = y[part].cuda()
            baseline, order, rel, _ = map_values(base, labels, g, s)
            rows["base"].append(baseline.cpu())
            info["base_top1_correct"].append(rel[:, 0].cpu())
            first_wrong = g[order[torch.arange(len(order), device=g.device), (~rel).int().argmax(1)]]
            for mode, values in scores.items():
                if mode.startswith("aligned/") or mode.startswith("broken_pair/"):
                    info["true_minus_top_rival/" + mode].append(rival_margin(values, membership, g, labels, first_wrong).cpu())
                for a in STRENGTHS:
                    measured = map_values(base + a * values[:, membership], labels, g, s)[0]
                    rows[f"{mode}/{a:g}"].append(measured.cpu())
        rows = {k: torch.cat(v) for k, v in rows.items()}
        info = {k: torch.cat(v) for k, v in info.items()}
        vals = {k: v.mean(0).tolist() for k, v in rows.items()}
        table[s] = vals
        file = dest.with_suffix(".pt")
        save_tensor({"metrics": rows, "query_ids": ids, "query_labels": y[ids], "diagnostics": info}, file)
        fail = ~info["base_top1_correct"]
        diagnostics = {k: {"miss_margin_mean": float(v[fail].mean()), "miss_positive": float((v[fail] > 0).float().mean()),
                            "hit_margin_mean": float(v[~fail].mean())} for k, v in info.items() if k != "base_top1_correct"}
        atomic_json(dest, {"signature": sig, "queries": len(ids), "gallery": len(g), "values": vals,
                           "misses": int(fail.sum()), "diagnosis": diagnostics, "max_null_error": neutral,
                           "per_query_sha256": checksum(file)})
        top = max((k for k in vals if k.startswith("aligned/")), key=vals.get)
        print(s, top, vals[top], "base", vals["base"], flush=True)
    mean = {k: [sum(table[s][k][i] for s in qs) / len(qs) for i in (0, 1)] for k in table["sketchy_1"]}
    atomic_json(root / "pilot.json", {"splits": table, "means": mean})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    run(p.parse_args().workspace)
