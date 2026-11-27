"""Attribute same-noun rankings to the sketch or to a shared gallery order."""

import argparse
from collections import defaultdict
import hashlib
from itertools import combinations
import json

import numpy as np
import torch
from torch.nn.functional import normalize

from .concept_benchmark import atomic_json, checksum
from .data import SKETCHY_SPLITS
from .kaggle_unseen_features import DEFAULT_CACHE, load_split
from .named_word_axis_audit import OUT, center, logits, text_readouts
from .native_block_probe import SEED, deterministic
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_operator_audit import SPLITS


CAP = 64
NAMES = ("winner", "remaining_words", "all_words", "visual")


def held_photos(split):
    dataset = "sketchy" if split.name in SKETCHY_SPLITS else "tuberlin"
    return [hashlib.sha256(f"{SEED}/{dataset}/{x}".encode()).digest() for x in split.gallery_ids.tolist()]


def prefer_a(scores, n_a):
    diff = scores[:, :n_a, None] - scores[:, None, n_a:]
    return ((diff > 0).double() + .5*(diff == 0)).mean(0)


def rate(units, n_a, q_a, q_b):
    u = prefer_a(units[q_a], n_a)
    v = prefer_a(units[q_b], n_a)
    right = (u*(1-v)).mean()
    exchanged = ((1-u)*v).mean()
    gap = (u-v).mean()
    if (right-exchanged-gap).abs() > 1e-12:
        raise ValueError("Crossed-rate decomposition failed")
    return torch.stack((right, exchanged, gap))


def collision(split, q, p, vq, vp):
    qt, pt = q.argmax(1).cpu(), p.argmax(1).cpu()
    priority = held_photos(split)
    used_queries, used_nouns, cells = set(), set(), []
    for w in qt.unique().tolist():
        qs = torch.where(qt == w)[0]
        ps = torch.where(pt == w)[0]
        classes = sorted(set(split.sketch_labels[qs].tolist()) & set(split.photo_labels[ps].tolist()))
        if len(classes) < 2:
            continue
        used_nouns.add(w)
        chosen, belongs = {}, {}
        for c in classes:
            pics = ps[split.photo_labels[ps] == c].tolist()
            chosen[c] = sorted(pics, key=priority.__getitem__)[:CAP]
            belongs[c] = torch.where(split.sketch_labels[qs] == c)[0]
            used_queries.update(qs[belongs[c]].tolist())
        # All scores created once per word and evaluated on identical photo-pair panels.
        gallery = [i for c in classes for i in chosen[c]]
        boundaries, start = {}, 0
        for c in classes:
            boundaries[c] = torch.arange(start, start+len(chosen[c]), device=q.device)
            start += len(chosen[c])
        pq, pp = q[qs].double(), p[gallery].double()
        head = pq[:, w:w+1] * pp[:, w]
        tail_only = pp.clone()
        tail_only[:, w] = 0
        tail = pq @ tail_only.T
        visual = vq[qs].double() @ vp[gallery].double().T
        scores = (head, tail, head + tail, visual)
        for a,b in combinations(classes, 2):
            take = torch.cat((boundaries[a],boundaries[b]))
            ia, ib = belongs[a], belongs[b]
            out = torch.stack([rate(s[:, take], len(chosen[a]), ia, ib) for s in scores])
            if abs(out[0,2].item()) > 1e-12:
                raise ValueError("A single leading word distinguished two queries with an identical noun")
            g_pairs = len(chosen[a])*len(chosen[b])
            cells.append({"word_id": w, "classes": [a,b], "sketches": [len(ia),len(ib)],
                          "photos": [len(chosen[a]),len(chosen[b])], "assignments": g_pairs*len(ia)*len(ib),
                          "photo_pairs": g_pairs, "rates": out.cpu().tolist()})
    panels = defaultdict(list)
    for row in cells:
        panels[tuple(row["classes"])].append(row["rates"])
    values = np.stack([np.mean(v,axis=0) for v in panels.values()])
    return {"rows": NAMES, "columns": ("both_correct","both_exchanged","difference"), "seed": SEED,
            "mean_class_pair_percent": (100*values.mean(0)).tolist(), "class_pairs": len(panels),
            "eligible_classes": len({c for group in panels for c in group}), "class_pair_word_panels": len(cells),
            "sketches": len(used_queries), "words": len(used_nouns), "photo_pairs": sum(c["photo_pairs"] for c in cells),
            "assignments":sum(c["assignments"] for c in cells), "cap_per_word_class": CAP}, cells


def inversion(bank, images, raw, scale):
    design = bank.double() - bank.double().mean(0)
    u,s,vh = torch.linalg.svd(design, full_matrices=False)
    rank = int((s > s[0]*max(design.shape)*torch.finfo(s.dtype).eps).sum())
    result = {"rank":rank, "dimension":bank.shape[1], "smallest_singular_value": s[-1].item(),
              "condition_number":(s[0]/s[-1]).item()}
    if rank != bank.shape[1]:
        return result
    inv = (u/s) @ vh
    for name, x in images.items():
        errors = defaultdict(list)
        for xb, zb in zip(x.split(512), raw[name].split(512), strict=True):
            xb = xb.double()
            native = zb.softmax(-1).double()
            if native.min() <= 0:
                raise ValueError("A CLIP probability underflowed")
            lp64 = (scale * (xb @ bank.double().T)).log_softmax(-1)
            for label, lp in (("stored_precision",native.log()),("reference64",lp64)):
                rec = (center(lp)/scale) @ inv
                errors[label].append((rec-xb).norm(dim=1) / xb.norm(dim=1))
        result[name] = {k: {"median_relative_error":torch.cat(e).median().item(),
                            "max_relative_error":torch.cat(e).max().item()} for k,e in errors.items()}
    return result


@torch.inference_mode()
def run(split_name, revision, audit_revision):
    prior_file = OUT / f"{split_name}-core.json"
    prior = json.loads(prior_file.read_bytes())
    if prior["revision"] != audit_revision or prior["text_sha256"] != checksum(DEFAULT_CACHE/"text.pt"):
        raise ValueError("Mismatch with frozen baseline diagnosis")
    split, text, _ = load_split(DEFAULT_CACHE, split_name)
    photos = normalize(split.photo.float(), dim=1)
    vq = normalize(normalize(split.sketch.float(), dim=1) + normalize(split.mirror.float(), dim=1), dim=1)
    source,target = (text[prompt] for prompt in DEFAULT_PROMPTS)
    zs,zp = logits(split, photos, source, target)
    _,_,q,p = text_readouts(zs,zp)
    summary,cells = collision(split, q,p,vq,photos)
    sketch_inv = inversion(source,{"original":split.sketch,"mirror":split.mirror.float()},
                           {"original":zs[:,0],"mirror":zs[:,1]}, split.scale)
    photo_inv = inversion(target,{"rgb":photos},{"rgb":zp},split.scale)
    target_dir = OUT/"attribution"
    cell_file = target_dir/f"{split_name}-panels.json"
    atomic_json(cell_file,cells)
    atomic_json(target_dir/f"{split_name}.json", {"revision":revision,"audit_revision":audit_revision,
                "baseline_json_sha256":checksum(prior_file),"classes":len(split.sketch_labels.unique()),
                "queries":len(q),"gallery":len(p),"collision":summary,"inversion":{"drawing":sketch_inv,"photo":photo_inv},
                "panels_sha256":checksum(cell_file),"device":torch.cuda.get_device_name()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",required=True,choices=SPLITS)
    parser.add_argument("--revision",required=True)
    parser.add_argument("--audit-revision",required=True)
    args = parser.parse_args()
    deterministic()
    run(args.split,args.revision,args.audit_revision)
