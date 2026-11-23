"""Test per-word regional evidence when the two whole-image posteriors disagree."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .branch_view_study import PrefixReplay
from .gallery_panel_resume import region_views, semantic_inputs
from .gallery_region_panels import CONFIG, Index
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd


def witnesses(first, flipped, regions, words, scale, decompose=False):
    mean = (first + flipped) / 2
    # Each of the 13 crops, in both poses, votes once for the public vocabulary.
    flat = regions.flatten(0, 2)
    local = (scale * (flat @ words.T)).softmax(1)
    local = local.view(len(first), -1, len(words)).mean(1)
    low = torch.minimum(first, flipped)
    high = torch.maximum(first, flipped)
    # The median of the two global observations and the independent crop readout.
    adjudicated = local.clamp(min=low, max=high)
    # Negative control: rotate crop votes among terms of similar global probability.
    order = mean.argsort(dim=1, descending=True)
    source = torch.arange(len(words), device=words.device)
    for block in source.split(16):
        block.copy_(block.roll(8))
    shuffled = torch.empty_like(local).scatter_(1, order, local.gather(1, order[:, source]))
    sham = shuffled.clamp(min=low, max=high)
    normalize = lambda x: x / x.sum(1, keepdim=True)
    output = {"mean": mean, "median": normalize(adjudicated), "sham": normalize(sham),
              "third_view": (2 * mean + local) / 3}
    if decompose:
        # local - mean = (adjudicated - mean) + positive_excess - negative_excess.
        # Uniform rescaling of one sketch posterior preserves its retrieval order.
        positive_excess = (local-high).clamp_min(0)
        negative_excess = (low-local).clamp_min(0)
        output.update({
            "inside_only": normalize(mean + (adjudicated-mean)/3),
            "outside_only": normalize(mean + (positive_excess-negative_excess)/3),
            "positive_only": normalize(mean + positive_excess/3),
            "negative_only": normalize(mean - negative_excess/3),
            "third_view_sham": (2 * mean + shuffled) / 3,
        })
    return output


def score_rows(queries, bank, regional):
    score = {name: post @ bank.T for name, post in queries.items()}
    logmean = score["mean"].clamp_min(1e-30).log()
    score["regional"] = regional
    for method in ("median", "sham", "third_view"):
        delta = score[method].clamp_min(1e-30).log() - logmean
        score["regional_" + method] = regional + .05 * regional.std(1, keepdim=True) * (
            delta / delta.std(1, keepdim=True).clamp_min(1e-8))
    return score


@torch.inference_mode()
def run(workspace, dataset, official, decompose=False, internal=False):
    cfg, paths, qs, _, gs, q, p, src, tgt, scale = semantic_inputs(workspace, dataset)
    kd, _ = load_kd(workspace)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() == KD_FUNCTION_SHA
    result = {}
    for split, (qi, labels) in qs.items():
        pi, classes = gs[split]
        bank = torch.cat([(scale * (z @ tgt.T)).softmax(1) for z in p[pi].cuda().split(256)])
        views, _ = region_views(workspace, split, cfg, paths, qi, q)
        regional = torch.load(workspace/"cache/gallery-stable-region-evidence"/(split+"-regional-reference.pt"),
                              mmap=True, weights_only=True)
        if internal:
            graph = Index(SimpleNamespace(p=p[pi].cuda()), rgb_only=True)
            history = torch.load(workspace/"cache/pragmatic-gallery-competition"/(split+"-named-rgb16.pt"),
                                 mmap=True, weights_only=True)
        replay = PrefixReplay(kd, classes, split)
        gclass = classes.cuda()
        output = {}
        maximum_error = 0.
        for start in range(0, len(qi), 48):
            sl = slice(start, start + 48)
            first = (scale * (q[qi[sl]].cuda() @ src.T)).softmax(1)
            flipped = (scale * (views["flipped"][sl].float().cuda() @ src.T)).softmax(1)
            post = witnesses(first, flipped, views["regions"][sl].cuda().float(), src, scale, decompose)
            base = regional[sl].float().cuda()
            if internal:
                old = graph.score(CONFIG, first @ bank.T).clamp_min(1e-30).log()
                historical = history[sl].cuda().clamp_min(1e-30).log()
                gap = historical - old
                # Softmax vs max-normalized emissions differ by a query-wide multiplier.
                if (gap-gap.median(dim=1,keepdim=True).values).abs().mean() > .01:
                    raise ValueError("Could not reproduce the frozen pre-region branch")
                scores_by_name = {"regional":base}
                for name in ("mean","median","third_view","outside_only","third_view_sham"):
                    alt = graph.score(CONFIG, post[name] @ bank.T).clamp_min(1e-30).log()
                    scores_by_name[name] = base + .65 * (alt-old)
            else:
                scores_by_name = score_rows(post, bank, base)
            for name, scores in scores_by_name.items():
                count = 200 if split == "sketchy_2" else len(pi)
                rank = scores.topk(count).indices
                hit = gclass[rank] == labels[sl,None].cuda()
                ap = (hit * hit.cumsum(1) / torch.arange(1,count+1,device=rank.device)).sum(1) / hit.sum(1).clamp_min(1)
                precision = hit[:,:replay.cutoffs[1]].float().mean(1)
                values = torch.stack([ap,precision],1).cpu()
                length = len(rank) if official else 1
                exact = replay.chunk((rank[:length].cpu(), labels[sl][:length]))
                error = (exact-values[:length]).abs().max().item()
                maximum_error = max(maximum_error,error)
                assert error < 2e-6
                output.setdefault(name,[]).append(exact if official else values)
        all_rows = {k:torch.cat(v).double() for k,v in output.items()}
        result[split] = {"queries":len(labels),"photos":len(pi),"official_all":official,
                         "maximum_checked_error":maximum_error,
                         "metrics":{k:(v.mean(0)*100).tolist() for k,v in all_rows.items()}}
        comparisons = ([("third_view","regional"),("third_view","mean"),("outside_only","third_view")]
                       if internal else [("median","mean"),("median","sham"),("regional_median","regional")])
        for left,right in comparisons:
            delta = (all_rows[left][:,0]-all_rows[right][:,0])*100
            class_delta = torch.stack([delta[labels==c].mean() for c in labels.unique()])
            result[split][left+" minus "+right] = {"classes_improved":int((class_delta>0).sum()),
                                                   "macro_delta":class_delta.mean().item()}
        print(json.dumps({split:result[split]}),flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sketchy","tuberlin"],default="sketchy")
    parser.add_argument("--official",action="store_true")
    parser.add_argument("--decompose",action="store_true")
    parser.add_argument("--internal",action="store_true")
    args = parser.parse_args()
    result = run(Path("/content/sbir"),args.dataset,args.official,args.decompose or args.internal,args.internal)
    out = Path("/content/sbir/results/view-disputed-words")
    out.mkdir(parents=True,exist_ok=True)
    suffix = ("-official" if args.official else "") + ("-decomposed" if args.decompose else "")
    suffix += "-internal" if args.internal else ""
    (out/(args.dataset+suffix+".json")).write_text(json.dumps(result,indent=2))
