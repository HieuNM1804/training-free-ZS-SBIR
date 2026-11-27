"""Compute the exact KD metric ceilings for changes confined to the query's winning noun."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json

import torch
from torch.nn.functional import normalize

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .encoding import save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE, load_split
from .named_word_axis_audit import OUT, logits, text_readouts
from .native_block_probe import SEED, deterministic
from .replicated_simplex_study import evaluator
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import BATCH
from .semantic_operator_audit import SPLITS


def complete_order(scores, cut):
    if not cut:
        return scores.topk(scores.shape[1]).indices
    prefix = scores.topk(cut).indices
    unused = torch.ones_like(scores, dtype=torch.bool).scatter_(1, prefix, False)
    ids = torch.arange(scores.shape[1], device=scores.device).expand_as(scores)
    return torch.cat((prefix, ids[unused].reshape(len(scores), -1)), dim=1)


def prefix_counts(allowed, relevant, cut):
    positives = (allowed & relevant).sum(1)
    if not cut:
        return positives, positives
    pos = torch.arange(1, cut+1, device=allowed.device, dtype=torch.float64)
    choices = torch.arange(cut+1, device=allowed.device)
    best_counts, max_counts = [], []
    for g, r in zip(allowed.split(32), relevant.split(32), strict=True):
        seen = g[:, :cut]
        number = seen.sum(1)
        lo = (number - (g & ~r).sum(1)).clamp_min(0)
        hi = torch.minimum(number, (g & r).sum(1))
        eligible = (choices[None, :] >= lo[:, None]) & (choices[None, :] <= hi[:, None])
        hits = ((r[:, :cut] & ~seen)[:, None, :] |
                (seen[:, None, :] & (seen.cumsum(1)[:, None, :] <= choices[None, :, None])))
        ap = (hits * hits.cumsum(2)/pos).sum(2) / hits.sum(2).clamp_min(1)
        best_counts.append(ap.masked_fill(~eligible, -1).argmax(1))
        max_counts.append(hi)
    return torch.cat(best_counts), torch.cat(max_counts)


def realize(order, allowed, relevant, count, cut):
    number = allowed.cumsum(1)
    wanted = allowed & (number <= count[:, None])
    if cut:
        remaining = (allowed & relevant).sum(1)-count
        wanted[:, cut:] = allowed[:, cut:] & (number[:, cut:] - allowed[:, :cut].sum(1,keepdim=True) <= remaining[:, None])
    if not torch.equal(wanted.sum(1), (allowed & relevant).sum(1)):
        raise ValueError("Unattainable relevance pattern")
    # Sort sources and destination slots into matching positive, negative and fixed groups.
    fixed = (~allowed).long()*2
    actual_order = (fixed+(allowed & ~relevant).long()).argsort(dim=1,stable=True)
    wanted_order = (fixed+(allowed & ~wanted).long()).argsort(dim=1,stable=True)
    result = torch.empty_like(order).scatter_(1,wanted_order,order.gather(1,actual_order))
    if (result[~allowed] != order[~allowed]).any():
        raise ValueError("A photo moved outside the allowed slots")
    return result, wanted


@torch.inference_mode()
def run(name, revision, audit_revision):
    split,text,_ = load_split(DEFAULT_CACHE, name)
    zs,zp = logits(split,normalize(split.photo.float(),dim=1),*(text[p] for p in DEFAULT_PROMPTS))
    _,_,q,p = text_readouts(zs,zp)
    photo_word, query_word = p.argmax(1),q.argmax(1)
    photo_class = split.photo_labels.cuda()
    original_path = OUT/f"{name}-core.pt"
    meta = json.loads(original_path.with_suffix(".json").read_bytes())
    if meta["revision"] != audit_revision or checksum(original_path) != meta["tensor_sha256"]:
        raise ValueError("Invalid reference baseline")
    original = torch.load(original_path, mmap=True, weights_only=True)["rows"]["NN"]
    kd = evaluator(DEFAULT_CACHE.parent/"real-sketchy-smoke/evaluator.py")
    replay = PrefixReplay(kd,split.photo_labels,name)
    cut = replay.cutoffs[0]
    rows = {"native":[],"AP_optimal":[],"P_optimal":[]}
    photo_bank = p.T.contiguous()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0,len(q),BATCH):
            sl = slice(start,start+BATCH)
            labels = split.sketch_labels[sl]
            order = complete_order(q[sl] @ photo_bank, cut)
            allowed = photo_word[order] == query_word[sl,None]
            relevant = photo_class[order] == labels.cuda()[:,None]
            rows["native"].append(replay.score(order,labels,pool).double())
            ap_count,p_count = prefix_counts(allowed,relevant,cut)
            for key,count in (("AP_optimal",ap_count),("P_optimal",p_count)):
                if key == "P_optimal" and torch.equal(ap_count,p_count):
                    rows[key].append(rows["AP_optimal"][-1])
                    continue
                changed,wanted = realize(order,allowed,relevant,count,cut)
                if not torch.equal(photo_class[changed] == labels.cuda()[:,None], (relevant & ~allowed) | wanted):
                    raise ValueError("Counterfactual relevance did not replay")
                rows[key].append(replay.score(changed,labels,pool).double())
    rows = {k:torch.cat(v) for k,v in rows.items()}
    if (rows["native"]-original).abs().max() > 2e-6:
        raise ValueError("Original per-query ranks did not reproduce")
    ceiling = torch.stack((rows["AP_optimal"][:,0],rows["P_optimal"][:,1]),dim=1)
    if not (ceiling >= rows["native"]-2e-6).all():
        raise ValueError("Purported oracle bound below observed scores")
    for key in ("AP_optimal","P_optimal"):
        if not (ceiling >= rows[key]-2e-6).all():
            raise ValueError("Metric-specific attainable bounds violated")
    dest = OUT/"attribution"/f"ceiling-{name}.pt"
    save_tensor({"query_ids":split.query_ids,"gallery_ids":split.gallery_ids,"rows":rows},dest)
    atomic_json(dest.with_suffix(".json"), {"split":name,"revision":revision,"seed":SEED,"audit_revision":audit_revision,
                "rows":{k:(100*v.mean(0)).tolist() for k,v in rows.items()},"queries":len(q),"gallery":len(p),
                "gain_AP_P_points":(100*(ceiling-rows["native"]).mean(0)).tolist(),
                "query_fraction_at_least_four_AP":((ceiling[:,0]-rows["native"][:,0])>=.04).double().mean().item(),
                "tensor_sha256":checksum(dest),"device":torch.cuda.get_device_name()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",required=True,choices=SPLITS)
    parser.add_argument("--revision",required=True)
    parser.add_argument("--audit-revision",required=True)
    args = parser.parse_args()
    deterministic()
    run(args.split,args.revision,args.audit_revision)
