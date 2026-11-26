"""Test the shared region assignment on matched photos and explain which rankings it changes."""

import argparse
import json
import math
import os
from pathlib import Path

import open_clip
import torch

from .benchmark import merge_inputs
from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import revision
from .crossdomain_inputs import runtime
from .depiction_path_diagnostic import class_interval
from .encoding import load_frozen_clip
from .gallery_convex_support import SEED
from .gallery_convex_support_audit import prefix
from .heldout_occlusion_correspondence import Correspondences, overlap
from .part_occlusion_observations import FAMILY, directory, public_words, read_cache, seed_job
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import SKETCHY, Split, posteriors
from .semantic_boxable_single_prompt import encode_prompt
from .semantic_boxable_sparse import image_inputs


def measurements(single_matrix, pair_matrix, maps):
    single = single_matrix.clamp_min(1e-30).log() @ maps.single_assignment
    double = pair_matrix.clamp_min(1e-30).log() @ maps.double_assignment
    one, two = single.log_softmax(-1), double.log_softmax(-1)
    shared = torch.logsumexp(one + two, -1) + maps.log_count
    uniform = (torch.logsumexp(double, -1) - maps.log_count) / 6
    heldout = torch.logsumexp(one + double, -1) / 6
    if (6 * (heldout - uniform) - shared).abs().max() > 2e-5:
        raise ValueError("Conditional/unconditional score ratio disagrees with shared-assignment Bayes factor")
    # Remove the fixed identity mapping and renormalize each correspondence distribution.
    nonidentity = (torch.logsumexp(single[..., 1:].log_softmax(-1) + double[..., 1:].log_softmax(-1), -1)
                   + math.log(23))
    return {
        "uniform_log_agreement": uniform, "heldout_log_agreement": heldout,
        "log_bayes_factor": shared, "log_factor_excluding_identity": nonidentity,
        "effective_single_assignments": (-(one.exp()*one).sum(-1)).exp(),
        "effective_double_assignments": (-(two.exp()*two).sum(-1)).exp(),
        "same_modal_assignment": (one.argmax(-1) == two.argmax(-1)).float(),
    }


def selected_overlap(query, candidates):
    b, masks, words = query.shape
    k = candidates.size(1)
    result = torch.bmm(query, candidates.reshape(b, k*masks, words).transpose(1, 2))
    return result.reshape(b, masks, k, masks).transpose(1, 2).reshape(b, k, masks*masks)


def candidate_table(query, selected, maps):
    return measurements(selected_overlap(query[:, :4], selected[:, :, :4]),
                        selected_overlap(query[:, 4:], selected[:, :, 4:]), maps)


def pair_totals(values, flags, order):
    ordered = values.gather(1, order)
    first = ordered[:, 0::2]
    second = ordered[:, 1::2]
    mixed, correct_first = flags
    advantage = torch.where(correct_first, first - second, second - first)
    return torch.stack((mixed.sum(1), (mixed & (advantage > 0)).sum(1),
                        (mixed & (advantage == 0)).sum(1)), 1).cpu()


def pair_summary(records, labels):
    n, wins, ties = torch.cat(records).double().T
    keep = n > 0
    return {
        "comparisons": int(n.sum()), "wins": int(wins.sum()), "ties": int(ties.sum()),
        "correct_percent": float(100 * (wins.sum() + .5*ties.sum()) / n.sum()),
        "macro_accuracy_minus_chance": class_interval(
            100 * (wins[keep] + .5*ties[keep]) / n[keep] - 50, labels[keep], seed=SEED,
        ),
    }


def trace(i, split, sq, sp, q, p, maps, paths):
    whole = q[i:i+1] @ p
    primary = []
    for photos in sp.split(768):
        obs = measurements(overlap(sq[i:i+1, :4], photos[:, :4]),
                           overlap(sq[i:i+1, 4:], photos[:, 4:]), maps)
        primary.append(obs["heldout_log_agreement"][0])
    complete = torch.cat(primary)
    selected = torch.stack((whole[0].argmax(), complete.argmax()))
    details = candidate_table(sq[i:i+1], sp[selected][None], maps)
    qp, pp = paths
    result = {"query": str(Path(qp[split.query_ids[i]]).relative_to(Path(qp[split.query_ids[i]]).parent.parent))}
    for pos, (method, idx) in enumerate(zip(("dot", "heldout"), selected.tolist(), strict=True)):
        path = Path(pp[split.gallery_ids[idx]])
        result[method] = {"photo": str(Path(path.parent.name) / path.name),
                          "same_class": bool(split.photo_labels[idx] == split.sketch_labels[i]),
                          "dot_score": whole[0, idx].item(),
                          **{name: v[0, pos].item() for name, v in details.items()}}
    return result


def audit(root, split, banks, text, maps, paths, signature):
    report = json.loads((root / (split.name + ".json")).read_bytes())
    blob = root / (split.name + ".pt")
    if report["protocol"] != signature or checksum(blob) != report["per_query_sha256"]:
        raise ValueError("Primary experiment changed")
    saved = torch.load(blob, weights_only=True)
    if any(not torch.equal(saved[key], ref) for key, ref in (
        ("qi", split.query_ids), ("pi", split.gallery_ids), ("labels", split.sketch_labels),
    )):
        raise ValueError("Different sample order")
    native = saved["rows"]
    qbank, pbank = posteriors(split, text, [DEFAULT_PROMPTS])
    q, p = qbank[DEFAULT_PROMPTS[0]], pbank[DEFAULT_PROMPTS[1]]
    sq, sp = banks[0][split.query_ids].cuda(), banks[1][split.gallery_ids].cuda()
    truth, gallery = split.sketch_labels.cuda(), split.photo_labels.cuda()
    counts = {name: [] for name in ("log_bayes_factor", "log_factor_excluding_identity")}
    averages = {}
    for begin in range(0, len(q), 24):
        sl = slice(begin, min(begin + 24, len(q)))
        ids = prefix(q[sl] @ p, *report["cutoffs"])
        hits = gallery[ids] == truth[sl, None]
        if not torch.allclose(hits.float().mean(1).cpu(), native["dot"][sl, 1].float(), atol=2e-6, rtol=0):
            raise ValueError("Dot precision mismatch")
        values = candidate_table(sq[sl], sp[ids], maps)
        order = values["uniform_log_agreement"].argsort(dim=1, descending=True, stable=True)
        ordered_hits = hits.gather(1, order)
        flags = ordered_hits[:, 0::2] ^ ordered_hits[:, 1::2], ordered_hits[:, 0::2]
        for name, rows in counts.items():
            rows.append(pair_totals(values[name], flags, order))
        for name, data in values.items():
            averages.setdefault(name, []).append(data.double().mean(1).cpu())
    before, after = native["dot"][:, 3].bool(), native["heldout"][:, 3].bool()
    examples = {}
    for name, mask in (("first_repaired_top1", ~before & after), ("first_damaged_top1", before & ~after)):
        ids = mask.nonzero().flatten()
        if len(ids):
            examples[name] = trace(int(ids[0]), split, sq, sp, q, p, maps, paths)
            expected = (False, True) if name == "first_repaired_top1" else (True, False)
            if tuple(examples[name][arm]["same_class"] for arm in ("dot", "heldout")) != expected:
                raise ValueError("Trace failed to reproduce official class decisions")
    return {
        "primary_protocol": signature, "audit_commit": revision(), "seed": SEED,
        "queries": len(q), "gallery": p.size(1), "dot_prefix": report["cutoffs"][1],
        "means_in_dot_prefix": {name: float(torch.cat(rows).mean()) for name, rows in averages.items()},
        "paired_different_class_relevance": {name: pair_summary(rows, split.sketch_labels) for name, rows in counts.items()},
        "retrieval_examples_in_archive_order": examples,
    }


@torch.inference_mode()
def run(dataset):
    seed_job()
    w = Path("/content/sbir")
    root = w / "results" / FAMILY
    lock = json.loads((root / "freeze.json").read_bytes())
    words, vocab_sha = public_words()
    if vocab_sha != lock["config"]["vocabulary_sha256"]:
        raise ValueError("Different vocabulary")
    names = SKETCHY if dataset == "sketchy" else ("tuberlin",)
    cfg = runtime(w, dataset, workers=min(8, len(os.sched_getaffinity(0))))
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = {s: encode_prompt(model, tokenizer, words, s) for s in DEFAULT_PROMPTS}
    data = image_inputs(cfg, names, model)
    maps = Correspondences(torch.device("cuda"))
    banks, paths = [], []
    for mode in ("sketch", "photo"):
        items, _ = merge_inputs(cfg.root, names, mode)
        folder, spec = directory(cfg, dataset, mode, items, vocab_sha)
        if json.loads((folder / "complete.json").read_bytes())["specification"] != spec:
            raise ValueError("Masked features from another run")
        banks.append(read_cache(folder))
        paths.append(items)
    for name in names:
        split = Split.from_inputs(data, name)
        result = audit(root, split, banks, text, maps, paths, lock["sha256"])
        atomic_json(root / (name + "-binding.json"), result)
        print(name, result["paired_different_class_relevance"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("sketchy", "tuberlin"))
    run(parser.parse_args().dataset)
