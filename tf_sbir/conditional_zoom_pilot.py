"""Does a native regional preview predict the value of the next full-resolution observation?"""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import SIZES, bare_words
from .conditional_zoom_model import build_bank, predict_observation, signature
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT
from .cross_view_region_cache import FAMILY as ZOOM
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .gallery_region_actions import conditional_utilities, group_prior, concordance
from .map_diagnosis import map_values
from .partial_object_parts import vocabulary
from .region_preview_cache import FAMILY
from .regional_intervention_study import index as compact_index
from .retrieval_diagnosis import global_inputs, stratified_sample
from .semantic import read_words

TEMPERATURES = (2., 8., 20.)


def photo_readout(native, text, compact, scale):
    z = native @ text.T * scale
    offset = z.amax((1, 2), keepdim=True)
    return ((z - offset).exp() @ compact.T).clamp_min(1e-30).log() + offset


def append(d, key, x):
    d[key].append(x.cpu())


def plan(preview_codes, prior, rivals, gallery):
    prediction = predict_observation(preview_codes, rivals, gallery, TEMPERATURES)
    prior = prior / prior.sum(1, keepdim=True)
    policy, expected, ess = {}, {}, {}
    for (kind, temp), channel in prediction.items():
        tag = f"{kind}/{temp:g}"
        if kind == "preview_response":
            group_like = channel.logsumexp(0) - .6931471805599453
            after = (prior.clamp_min(1e-30).log()[:, :, None] + group_like).softmax(1)
            policy[tag] = (after * (after.clamp_min(1e-30).log() - prior.clamp_min(1e-30).log()[:, :, None])).sum(1)
        elif kind == "ess":
            ess[tag] = (channel.mean(0) * prior[:, :, None]).sum(1).mean(1)
        else:
            norm = channel / channel.sum(-1, keepdim=True).clamp_min(1e-30)
            policy[tag] = conditional_utilities(prior, norm[0], norm[1])["heldout"]
            expected[tag] = (prior[:, :, None, None] * norm.mean(0)).sum(1)
    return policy, expected, ess


@torch.inference_mode()
def run(w):
    root = w / "results" / FAMILY
    cfg, _, qs, paths, galleries, qraw, praw = global_inputs(w, "sketchy")
    words = read_words(cfg.vocabulary)
    if len(words) != 20931:
        raise ValueError("Public vocabulary mismatch")
    src, tgt = (x.cuda() for x in load_words(cfg, words))
    _, scale = bare_words(cfg, words)
    cache = w / "cache" / FAMILY / "sketchy"
    ppre, qpre = (torch.load(cache / (n + ".pt"), weights_only=True, mmap=True) for n in ("photo", "sketch"))
    pzoom = torch.load(w / "cache/nested-scale-word-evidence/sketchy/photo.pt", weights_only=True, mmap=True)
    qzoom = torch.load(w / "cache" / ZOOM / "sketchy/views.pt", weights_only=True, mmap=True)["original_regions"]
    spec = {"revision": revision(), "checkpoint": CHECKPOINT, "temps": TEMPERATURES, "phase": "balanced-pilot",
            "seed": 63241, "sample_per_class": 36, "rivals": 12, "global_temperature": 2., "mix": .4,
            "support_per_panel_group": 96, "preview": {n: read(cache / (n + ".json")) for n in ("photo", "sketch")}}
    sig = digest(spec)
    if (root / "protocol.json").exists() and read(root / "protocol.json")["sha256"] != sig:
        raise ValueError("Pilot protocol changed")
    atomic_json(root / "protocol.json", spec | {"sha256": sig})
    overall = {}
    for split, (qi, y) in qs.items():
        pi, g = galleries[split]
        if (len(qi), len(pi), len(g.unique())) != SIZES[split] or digest(UNSEEN_CLASSES[split]) != SPLIT_HASHES[split]:
            raise ValueError("Wrong split")
        output = root / ("pilot-" + split + ".json")
        if output.exists():
            previous = read(output)
            if previous["signature"] != sig or previous["per_query_sha256"] != checksum(output.with_suffix(".pt")):
                raise ValueError("Stored pilot is inconsistent")
            overall[split] = previous["values"]
            continue
        photo = praw[pi].cuda()
        ident = digest({"gallery": digest([str(paths[i]) for i in pi.tolist()]), "revision": "174579185794aca151e95a9309cf2b49f32c0fcb"})[:16]
        heads, compact = compact_index(w, split, photo, tgt, scale, ident)
        crops = pzoom[pi]
        terms = vocabulary(crops, tgt, scale, 128)
        membership, *gallery = build_bank(photo, ppre[pi], crops, tgt[terms], scale, maximum=96)
        history = torch.load(w / "cache/pragmatic-gallery-competition" / (split + "-named-rgb16.pt"), weights_only=True, mmap=True)
        if history.shape != (len(qi), len(pi)):
            raise ValueError("Global reference mismatch")
        ids = stratified_sample(y, 36, 63241)
        labels = g.cuda()
        weights = photo.new_tensor([1 / 18] * 9 + [1 / 8] * 4)
        rows, diagnostics = defaultdict(list), defaultdict(list)
        for part in ids.split(18):
            full = history[part].cuda().clamp_min(1e-30).log()
            at = qi[part]
            query = qraw[at].cuda()
            coarse = qpre[at].cuda().float()
            p, rivals = group_prior(full, membership, 2., 12)
            plans, future, evidence = plan(signature(coarse, query, src[terms]), p, rivals, gallery)
            # From this point on the alternatives can be revealed for evaluation.
            native = qzoom[at].cuda().float()
            observed = (native @ src[terms].T * scale).softmax(-1)
            zoom = photo_readout(native, src[heads], compact, scale)
            low = photo_readout(coarse, src[heads], compact, scale)
            target = y[part].cuda()
            batch = torch.arange(len(part), device=photo.device)
            metrics = torch.stack([map_values(.6 * full + .4 * zoom[:, r], target, labels, split)[0] for r in range(13)], 1)
            append(rows, "global", map_values(full, target, labels, split)[0])
            append(rows, "all13", map_values(.6 * full + .4 * (zoom + weights.log()[None, :, None]).logsumexp(1), target, labels, split)[0])
            append(rows, "all_native_previews", map_values(.6 * full + .4 * (low + weights.log()[None, :, None]).logsumexp(1), target, labels, split)[0])
            for k, utility in plans.items():
                action = utility.argmax(1)
                append(rows, k, metrics[batch, action])
                if k.startswith("aligned/"):
                    append(rows, "preview_only/" + k, map_values(.6 * full + .4 * low[batch, action], target, labels, split)[0])
                concord, count = concordance(utility, metrics[:, :, 0])
                append(diagnostics, "actions/" + k, action)
                append(diagnostics, "concordance/" + k, torch.stack((concord, count.float()), 1))
            for name, pred in future.items():
                loss = -(observed * pred.clamp_min(1e-30).log()).sum(-1).mean(1)
                append(diagnostics, "zoom_word_cross_entropy/" + name, loss)
            for name, v in evidence.items():
                append(diagnostics, name, v)
        rows = {k: torch.cat(v) for k, v in rows.items()}
        diagnostics = {k: torch.cat(v) for k, v in diagnostics.items()}
        vals = {k: v.mean(0).tolist() for k, v in rows.items()}
        overall[split] = vals
        summary = {k: (float(val[val[:, 1] > 0, 0].mean()) if k.startswith("concordance")
                       else val.bincount(minlength=13).tolist() if k.startswith("actions/") else float(val.mean()))
                   for k, val in diagnostics.items()}
        dest = output.with_suffix(".pt")
        save_tensor({"metrics": rows, "query_ids": ids, "query_labels": y[ids], "diagnostics": diagnostics}, dest)
        atomic_json(output, {"signature": sig, "queries": len(ids), "gallery": len(g), "values": vals,
                             "diagnostics": summary, "per_query_sha256": checksum(dest)})
        k = max((k for k in vals if k.startswith("aligned/")), key=vals.get)
        print(split, k, vals[k], "unconditional", vals["unconditional/0"], flush=True)
    means = {k: [sum(overall[s][k][i] for s in qs) / len(qs) for i in (0, 1)] for k in overall["sketchy_1"]}
    atomic_json(root / "pilot.json", {"splits": overall, "means": means})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    run(p.parse_args().workspace)
