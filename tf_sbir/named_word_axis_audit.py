"""Isolate the axes, the pointwise readout, and the text prompts in the existing semantic baseline."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import numpy as np
import torch
from torch.nn.functional import normalize

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .conditional_completion_study import evaluate
from .crossdomain_inputs import CHECKPOINT
from .encoding import save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE, load_split
from .native_block_probe import SEED, deterministic
from .replicated_simplex_study import evaluator
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_donor_specificity import boot
from .semantic_operator_audit import SPLITS


OUT = DEFAULT_CACHE.parent / "results" / "named-word-axis-audit"
PANELS = ("core", "rotation0", "rotation1", "rotation2")


def center(x):
    return x - x.mean(-1, keepdim=True)


def logits(split, photos, source, target):
    views = torch.stack((split.sketch, split.mirror.float()), dim=1)
    return split.scale * (views @ source.T), split.scale * (photos @ target.T)


def text_readouts(sketch, photo):
    return (center(sketch).mean(1), center(photo),
            sketch.softmax(-1).mean(1), photo.softmax(-1))


def fixed_rotation(size, index):
    gen = np.random.default_rng(SEED)
    for _ in range(index + 1):
        matrix = gen.standard_normal((size-1, size-1))
    q, r = np.linalg.qr(matrix)
    q *= np.where(np.diag(r) >= 0, 1., -1.)
    block = np.eye(size)
    block[:-1, :-1] = q
    v = np.full(size, -1. / np.sqrt(size))
    v[-1] += 1
    v /= np.linalg.norm(v)
    householder = np.eye(size) - 2 * np.outer(v, v)
    u = np.ascontiguousarray(householder @ block @ householder)
    errors = {"orthogonality": float(np.abs(u.T @ u - np.eye(size)).max()),
              "constant": float(np.abs(u.sum(1) - 1).max())}
    if max(errors.values()) > 1e-10:
        raise ValueError(f"Invalid contrast-space rotation {errors}")
    return torch.from_numpy(u).cuda(), hashlib.sha256(u.tobytes()).hexdigest(), errors


def match_order_statistics(native, rotated):
    ids = rotated.argsort(dim=1, stable=True)
    sorted_values = native.sort(dim=1).values
    result = torch.empty_like(native).scatter_(1, ids, sorted_values)
    if not torch.equal(sorted_values, result.sort(1).values):
        raise ValueError("Per-image probability spectrum changed")
    return result


def conditional_auc(scores, positive):
    p = scores[positive]
    n = np.sort(scores[~positive])
    lower = n.searchsorted(p, side="left")
    upper = n.searchsorted(p, side="right")
    return (lower + upper).mean() / (2 * len(n))


def audit_winners(q, p, visual_q, visual_p, split):
    names = ("winner", "remaining_words", "all_words", "visual")
    q_top, p_top = q.argmax(1).cpu(), p.argmax(1).cpu()
    values = torch.full((len(q), len(names)), torch.nan, dtype=torch.float64)
    pairs = torch.zeros(len(q), 2, dtype=torch.long)
    for word in q_top.unique().tolist():
        g = torch.where(p_top == word)[0]
        if g.numel() < 2:
            continue
        labels = split.photo_labels[g].numpy()
        if np.unique(labels).size < 2:
            continue
        gallery = p[g].double()
        gallery_tail = gallery.clone()
        gallery_tail[:, word] = 0
        gallery_visual = visual_p[g].double().T.contiguous()
        for ids in torch.where(q_top == word)[0].split(128):
            query = q[ids].double()
            head = query[:, word:word+1] * gallery[:, word]
            tail = query @ gallery_tail.T
            visual = visual_q[ids].double() @ gallery_visual
            readings = [a.cpu().numpy() for a in (head, tail, head + tail, visual)]
            for i, row in enumerate(ids.tolist()):
                positive = labels == split.sketch_labels[row].item()
                n = positive.sum()
                if 0 < n < len(g):
                    pairs[row] = torch.tensor((n, len(g) - n))
                    values[row] = torch.tensor([conditional_auc(s[i], positive) for s in readings],
                                               dtype=torch.float64)
    keep = pairs.min(1).values > 0
    if not torch.equal(keep, values.isfinite().all(1)):
        raise ValueError("Inconsistent matched-pair coverage")
    classes = split.sketch_labels[keep].unique()
    rows = values[keep]
    cls = torch.stack([values[keep & (split.sketch_labels == c)].mean(0) for c in classes])
    if ((rows < 0) | (rows > 1)).any():
        raise ValueError("Invalid tie-adjusted AUC")
    stats = {"columns": names, "eligible_queries": int(keep.sum()), "all_queries": len(q),
             "classes": len(classes), "eligible_pairs": int(pairs.prod(1).sum()),
             "mean_query_AUC_percent": (100 * rows.mean(0)).tolist(),
             "class_balanced_AUC_percent": boot((100*cls).numpy()),
             "class_balanced_remaining_minus_winner": boot((100*(cls[:, 1]-cls[:, 0])).numpy()),
             "class_balanced_all_minus_winner": boot((100*(cls[:, 2]-cls[:, 0])).numpy())}
    return {"auc": values, "positive_negative_counts": pairs}, stats


@torch.inference_mode()
def measure(split_name, panel, revision):
    split, text, words = load_split(DEFAULT_CACHE, split_name)
    if len(words) != 601 or len(set(words)) != len(words):
        raise ValueError("Unexpected vocabulary")
    photos = normalize(split.photo.float(), dim=1)
    visual_query = normalize(normalize(split.sketch.float(), dim=1) + normalize(split.mirror.float(), dim=1), dim=1)
    source, target = (text[prompt] for prompt in DEFAULT_PROMPTS)
    zs, zp = logits(split, photos, source, target)
    linear_q, linear_p, q, p = text_readouts(zs, zp)
    checks = {}
    extra = {}
    if panel == "core":
        source_photo = logits(split, photos, target, target)[0]
        target_drawing = split.scale * (photos @ source.T)
        banks = {"LL": (linear_q, linear_p), "NL": (q, linear_p), "LN": (linear_q, p),
                 "NN": (q, p), "both_photo": (source_photo.softmax(-1).mean(1), p),
                 "both_drawing": (q, target_drawing.softmax(-1)), "visual": (visual_query, photos)}
        permutation = torch.randperm(len(words), generator=torch.Generator().manual_seed(SEED))
        a, b = zs[:32].double(), zp[:32].double()
        native = a.softmax(-1).mean(1) @ b.softmax(-1).T
        permuted = a[..., permutation].softmax(-1).mean(1) @ b[:, permutation].softmax(-1).T
        checks["common_permutation_max_score_error"] = (native-permuted).abs().max().item()
        if checks["common_permutation_max_score_error"] > 1e-12:
            raise ValueError("Shared permutation changed the native score")
        extra, checks["same_winner"] = audit_winners(q, p, visual_query, photos, split)
    else:
        u, signature, errors = fixed_rotation(len(words), int(panel[-1]))
        a = center(zs[:64].double()).mean(1)
        b = center(zp[:64].double())
        errors["linear_scores"] = ((a @ u) @ (b @ u).T - a @ b.T).abs().max().item()
        if errors["linear_scores"] > 1e-8:
            raise ValueError("Common rotation changed the centered linear kernel")
        checks = {"rotation_sha256": signature, "errors": errors}
        rq = (zs.double() @ u).softmax(-1).mean(1).float()
        rp = (zp.double() @ u).softmax(-1).float()
        banks = {"rotated": (rq, rp), "matched": (match_order_statistics(q, rq), match_order_statistics(p, rp))}
    kd_file = DEFAULT_CACHE.parent / "real-sketchy-smoke/evaluator.py"
    replay = PrefixReplay(evaluator(kd_file), split.photo_labels, split.name)
    rows = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, (query, bank) in banks.items():
            rows[name] = evaluate(split, query, bank.contiguous(), replay, pool)[:, :2].double()
            if tuple(rows[name].shape) != (len(q), 2) or not rows[name].isfinite().all():
                raise ValueError(f"Incomplete official evaluation for {name}")
            print(split_name, panel, name, flush=True)
    dest = OUT / f"{split_name}-{panel}.pt"
    save_tensor({"query_ids": split.query_ids, "gallery_ids": split.gallery_ids,
                 "query_labels": split.sketch_labels, "rows": rows, **extra}, dest)
    atomic_json(dest.with_suffix(".json"), {"split": split_name, "panel": panel, "revision": revision,
                "seed": SEED, "checkpoint": CHECKPOINT, "prompts": list(DEFAULT_PROMPTS),
                "gallery": len(p), "queries": len(q), "scale": split.scale, "device": torch.cuda.get_device_name(),
                "metrics": {k: (100*v.mean(0)).tolist() for k, v in rows.items()}, "checks": checks,
                "tensor_sha256": checksum(dest), "text_sha256": checksum(DEFAULT_CACHE/"text.pt"),
                "evaluator_sha256": checksum(kd_file)})


def summarize(split_name, revision):
    rows, hashes, fixed = {}, {}, None
    for panel in PANELS:
        path = OUT / f"{split_name}-{panel}.pt"
        meta = json.loads(path.with_suffix(".json").read_bytes())
        blob = torch.load(path, weights_only=True, mmap=True)
        keys = (meta["evaluator_sha256"], meta["text_sha256"], meta["seed"], meta["revision"])
        if checksum(path) != meta["tensor_sha256"] or meta["revision"] != revision:
            raise ValueError("Inconsistent artifact signature")
        if panel == "core":
            ids, gallery_ids, labels = blob["query_ids"], blob["gallery_ids"], blob["query_labels"]
            fixed = keys
            rows.update(blob["rows"])
        else:
            if keys != fixed or not all(torch.equal(x, blob[k]) for x,k in
                ((ids,"query_ids"),(gallery_ids,"gallery_ids"),(labels,"query_labels"))):
                raise ValueError("Unequal evaluation populations")
            rows.update({f"{k}_{panel[-1]}": t for k,t in blob["rows"].items()})
            hashes[panel] = meta["checks"]["rotation_sha256"]
    if len(set(hashes.values())) != 3:
        raise ValueError("Duplicate basis draw")
    effects = {"NN_minus_LL": rows["NN"]-rows["LL"],
               "NL_minus_LL": rows["NL"]-rows["LL"], "LN_minus_LL": rows["LN"]-rows["LL"],
               "NN_minus_both_photo": rows["NN"]-rows["both_photo"],
               "NN_minus_both_drawing": rows["NN"]-rows["both_drawing"]}
    for key in ("rotated","matched"):
        mean = torch.stack([rows[f"{key}_{i}"] for i in range(3)]).mean(0)
        effects[f"NN_minus_{key}_mean"] = rows["NN"] - mean
    details = {}
    for name, delta in effects.items():
        class_delta = torch.stack([delta[labels == c].mean(0) for c in labels.unique()])
        details[name] = {"query_points": (100*delta.mean(0)).tolist(),
                         "class_balanced": boot((100*class_delta).numpy())}
    atomic_json(OUT / f"summary-{split_name}.json", {"revision": revision, "seed": SEED,
                "queries": len(ids), "gallery": len(gallery_ids), "classes": len(labels.unique()),
                "rotation_sha256": hashes, "metrics": {k: (100*v.mean(0)).tolist() for k,v in rows.items()},
                "effects": details})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--panel", required=True, choices=(*PANELS,"summarize"))
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    deterministic()
    if args.panel == "summarize":
        summarize(args.split, args.revision)
    else:
        measure(args.split, args.panel, args.revision)
