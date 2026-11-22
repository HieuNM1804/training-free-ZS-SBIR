"""Prepare gallery and independent query views of the existing frozen regional retriever."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import SIZES, bare_words
from .crossdomain_benchmark import SPLIT_HASHES, digest
from .crossdomain_inputs import CHECKPOINT
from .cross_view_region_cache import FAMILY as VIEWS
from .cross_view_regions import CandidateRegions
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .regional_intervention_study import index as compact_index
from .retrieval_diagnosis import global_inputs
from .semantic import read_words

FAMILY = "word-conditional-visual-verification"


def detail_view(x):
    x = x.float()
    return F.normalize((x[:, :9].mean(1) + x[:, 9:].mean(1)) * .5, dim=-1)


def get_inputs(w, ds):
    cfg, qp, qs, pp, galleries, query_raw, photo_raw = global_inputs(w, ds)
    root = w / "cache" / FAMILY
    result = {}
    for split, (qi, y) in qs.items():
        pi, g = galleries[split]
        if (len(qi), len(pi), len(g.unique())) != SIZES[split] or digest(UNSEEN_CLASSES[split]) != SPLIT_HASHES[split]:
            raise ValueError("Unexpected split")
        signature = digest({"checkpoint": CHECKPOINT, "strong": "cbc4f57/self/1/gallery/first/0.35",
                            "query": digest([str(qp[i]) for i in qi.tolist()]), "gallery": digest([str(pp[i]) for i in pi.tolist()]),
                            "vocabulary": checksum(cfg.vocabulary)})
        name = root / (split + ".pt")
        info = name.with_suffix(".json")
        if name.is_file() and info.is_file() and read(info)["signature"] == signature:
            result[split] = (torch.load(name, weights_only=True, mmap=True), y, g)
            continue
        words = read_words(cfg.vocabulary)
        if len(words) != 20931:
            raise ValueError("Different external vocabulary")
        source, target = (v.cuda() for v in load_words(cfg, words))
        _, scale = bare_words(cfg, words)
        strong_record = read(w / "results" / VIEWS / (split + ".json"))
        if strong_record["best_contrast"] != '["self",1.0,"gallery","first",0.35]':
            raise ValueError("Unexpected regional reference")
        ident = digest({"signature": strong_record["signature"], "gallery": digest([str(pp[i]) for i in pi.tolist()]),
                        "vocabulary": checksum(cfg.vocabulary)})[:16]
        a = photo_raw[pi].cuda()
        heads, compact = compact_index(w, split, a, target, scale, ident)
        qviews = torch.load(w / "cache" / VIEWS / ds / "views.pt", weights_only=True, mmap=True)
        pviews = torch.load(w / "cache/nested-scale-word-evidence" / ds / "photo.pt", weights_only=True, mmap=True)
        b = torch.cat([detail_view(pviews[ids].cuda()).cpu() for ids in pi.split(256)])
        hist = torch.load(w / "cache/pragmatic-gallery-competition" / (split + "-named-rgb16.pt"), weights_only=True, mmap=True)
        if hist.shape != (len(qi), len(pi)):
            raise ValueError("Gallery score dimensions differ")
        base = torch.empty((len(qi), len(pi)), dtype=torch.float32)
        qb = torch.empty((len(qi), a.size(1)))
        for start in range(0, len(qi), 32):
            loc = slice(start, start + 32)
            ids = qi[loc]
            z = qviews["original_regions"][ids].cuda()
            reflected = qviews["regions"][ids].cuda()
            reader = CandidateRegions(query_raw[ids].cuda(), qviews["global"][ids].cuda(), z, reflected, source,
                                      heads, compact, source.new_empty((0, len(source))), hist[loc].cuda())
            reader.prepare(scale)
            base[loc] = (.35 * reader.read("self", 1., "gallery") + .65 * reader.first).cpu()
            qb[loc] = detail_view(z).cpu()
        data = {"gallery_native": a.cpu(), "gallery_detail": b, "query_native": query_raw[qi], "query_detail": qb,
                "public_words": compact.cpu(), "regional_score": base}
        save_tensor(data, name)
        atomic_json(info, {"signature": signature, "queries": len(qi), "gallery": len(pi), "native_dimensions": a.size(1)})
        result[split] = (torch.load(name, weights_only=True, mmap=True), y, g)
        print("cached", split, len(qi), len(pi), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--dataset", choices=("sketchy", "tuberlin"), required=True)
    args = parser.parse_args()
    with torch.inference_mode():
        get_inputs(args.workspace, args.dataset)
