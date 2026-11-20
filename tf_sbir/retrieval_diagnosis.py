"""Probe retrieval errors on the two Sketchy splits of one frozen CLIP model."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .benchmark import merge_inputs
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT, projection, runtime
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .encoding import load_global_inputs
from .foundation import load_words
from .semantic import read_words


def global_inputs(workspace, dataset):
    splits = SKETCHY_SPLITS if dataset == "sketchy" else ("tuberlin",)
    cfg = runtime(workspace, dataset, batch=384, workers=6)
    qp, queries = merge_inputs(cfg.root, splits, "sketch")
    pp, photos = merge_inputs(cfg.root, splits, "photo")
    q, p, _ = load_global_inputs(cfg, qp, pp, False)
    return cfg, qp, queries, pp, photos, q, p


def stratified_sample(labels, count, seed):
    rng = torch.Generator().manual_seed(seed)
    selected = []
    for category in labels.unique(sorted=True):
        ids = (labels == category).nonzero().flatten()
        selected.append(ids[torch.randperm(len(ids), generator=rng)[:count]])
    return torch.cat(selected)


def ranking_statistics(scores, qlabels, plabels, k):
    order = scores.argsort(dim=1, descending=True, stable=True)
    correct = plabels[order] == qlabels[:, None]
    top = order[:, :10]
    errors = top[~correct[:, :10]].bincount(minlength=len(plabels))
    hot = max(1, math.ceil(len(plabels) * .01))
    ranks = correct.int().argmax(1) + 1
    summary = {
        "top1": correct[:, 0].float().mean().item(),
        "precision": correct[:, :k].float().mean().item(),
        "mean_first_positive_rank": ranks.float().mean().item(),
        "median_first_positive_rank": ranks.float().median().item(),
        "wrong_top10_count": int(errors.sum()),
        "wrong_top10_share_top1pct_gallery": (errors.topk(hot).values.sum() / errors.sum().clamp_min(1)).item(),
    }
    return summary, order, ranks


def relative_codes(inputs, anchors, temperature=.03):
    logits, ids = (inputs @ anchors.T).topk(16, dim=1)
    return ids, (logits / temperature).softmax(1)


def diagnosis(split, qpaths, ppaths, queries, photos, qlabels, plabels, texts, transform):
    names = UNSEEN_CLASSES[split]
    if digest(names) != SPLIT_HASHES[split]:
        raise ValueError("Class order changed")
    qids = stratified_sample(qlabels, 24, 1703)
    pids = stratified_sample(plabels, 24, 1203)
    p, q = photos.cuda(), queries[qids].cuda()
    ql, pl = qlabels[qids].cuda(), plabels.cuda()
    k = 100 if split == "sketchy_1" else 200
    base = q @ p.T
    summary, order, first = ranking_statistics(base, ql, pl, k)
    correct = pl[order[:, 0]] == ql
    nearest_photo = p[pids] @ p.T
    nearest_photo[torch.arange(len(pids)), pids] = -torch.inf
    reference = ranking_statistics(nearest_photo, pl[pids], pl, k)[0]
    adapted = {}
    def compare(name, scores):
        item, ranking, _ = ranking_statistics(scores, ql, pl, k)
        matched = pl[ranking[:, 0]] == ql
        item.update(top1_rescued=int((matched & ~correct).sum()), top1_lost=int((~matched & correct).sum()))
        adapted[name] = item
        return ranking[:, 0].tolist()

    q_iso, p_iso = (F.normalize(x @ transform, dim=-1) for x in (q, p))
    compare("gallery_centered", F.normalize(q - p.mean(0)) @ F.normalize(p - p.mean(0)).T)
    compare("isoclip_only", q_iso @ p_iso.T)
    sketch_words, photo_words = texts
    # Each reference bank is independent of the test sketches. Exclude the gallery diagonal.
    density, prompt_density = [], []
    for start in range(0, len(p), 256):
        block = p[start:start + 256]
        similarities = block @ p.T
        locals_ = torch.arange(len(block), device=p.device)
        similarities[locals_, start + locals_] = -torch.inf
        density.append(similarities.topk(64).values.mean(1))
        prompt_density.append((block @ sketch_words.T).topk(64).values.mean(1))
    compare("nnn_gallery_reference", base - .75 * torch.cat(density))
    compare("nnn_public_prompt_reference", base - .75 * torch.cat(prompt_density))
    term_ids, weights = relative_codes(q, sketch_words)
    projected = F.normalize((photo_words[term_ids] * weights[..., None]).sum(1))
    rescued = compare("paired_terms_to_photo", projected @ p.T)
    same = F.normalize((sketch_words[term_ids] * weights[..., None]).sum(1))
    compare("same_domain_terms_to_photo", same @ p.T)
    perm = torch.randperm(len(photo_words), generator=torch.Generator().manual_seed(1704)).cuda()
    shuffled = F.normalize((photo_words[perm[term_ids]] * weights[..., None]).sum(1))
    compare("permuted_term_identity", shuffled @ p.T)
    p_ids, p_weights = relative_codes(p, photo_words)
    photo_codes = F.normalize((photo_words[p_ids] * p_weights[..., None]).sum(1))
    compare("paired_terms_to_gallery_words", projected @ photo_codes.T)

    errors = Counter((names[a], names[b]) for a, b in zip(ql.tolist(), pl[order[:, 0]].tolist()) if a != b)
    examples, used = [], set()
    for j in first.argsort(descending=True).tolist():
        category = int(ql[j])
        if category in used or bool(correct[j]):
            continue
        used.add(category)
        def relative(path):
            return "/".join(Path(path).parts[-3:])
        examples.append({
            "query": relative(qpaths[qids[j]]), "wrong_top1": relative(ppaths[order[j, 0]]),
            "first_correct": relative(ppaths[order[j, first[j] - 1]]), "first_correct_rank": int(first[j]),
            "paired_term_top1": relative(ppaths[rescued[j]]),
        })
        if len(examples) == 8:
            break
    return {
        "queries_sampled": len(q), "gallery": len(p), "cutoff": k,
        "class_balanced_queries": True, "seed": 1703, "classes": len(names),
        "native": summary, "photo_to_photo_excluding_self": reference, "interventions": adapted,
        "top1_confusions": [[*pair, count] for pair, count in errors.most_common(8)], "examples": examples,
    }


@torch.inference_mode()
def run(workspace):
    args, qp, queries, pp, galleries, raw, photos = global_inputs(workspace, "sketchy")
    terms = read_words(args.vocabulary)
    texts = tuple(x.cuda() for x in load_words(args, terms))
    result = {"checkpoint": CHECKPOINT, "terms": len(terms), "commit": revision(),
              "gpu": torch.cuda.get_device_name(), "splits": {}}
    transform = projection(args)
    for split in SKETCHY_SPLITS:
        qi, ql = queries[split]
        pi, pl = galleries[split]
        result["splits"][split] = diagnosis(split, [qp[i] for i in qi.tolist()], [pp[i] for i in pi.tolist()],
                                            raw[qi], photos[pi], ql, pl, texts, transform)
    out = workspace / "results/clip-retrieval-diagnosis.json"
    temp = out.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(out)
    print(out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
