"""Project local word probabilities onto the ordering of a single whole query.

For the sorted global posterior, antitonic least squares pools adjacent crop
coordinates that contradict the global ordering. Controls preserve the exact
crop spectrum or match entropy by a single temperature.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import isotonic_regression
import torch

from .view_disputed_words import run


def match_entropy(reference, target):
    log_ref = reference.clamp_min(1e-30).log()
    entropy = -(target * target.clamp_min(1e-30).log()).sum(1,keepdim=True)
    low = torch.zeros_like(entropy)
    high = torch.full_like(low,32)
    for _ in range(24):
        beta = (low+high)/2
        lp = (log_ref*beta).log_softmax(1)
        above = -(lp.exp()*lp).sum(1,keepdim=True) > entropy
        low = torch.where(above,beta,low)
        high = torch.where(above,high,beta)
    output = (log_ref*((low+high)/2)).softmax(1)
    error = (-(output*output.clamp_min(1e-30).log()).sum(1,keepdim=True)-entropy).abs()
    if error.max() > 1e-4:
        raise RuntimeError("Entropy control did not match")
    return output


def ranked_views(first, flipped, regions, words, scale, decompose):
    mean = (first+flipped)/2
    probabilities = (scale * (regions.flatten(0,2) @ words.T)).softmax(1)
    local = probabilities.view(len(first),-1,len(words)).mean(1)
    order = mean.argsort(dim=1,descending=True,stable=True)
    values = local.gather(1,order).double().cpu().numpy()
    fitted = np.stack([isotonic_regression(v,increasing=False).x for v in values])
    projected = torch.empty_like(local).scatter_(1,order,torch.from_numpy(fitted).to(local))
    assert (projected.sum(1)-1).abs().max()<1e-5
    rearranged = torch.empty_like(local).scatter_(1,order,local.sort(dim=1,descending=True).values)
    corrected = (2*mean+projected)/3
    return {"mean":mean, "raw":(2*mean+local)/3, "isotonic":corrected,
            "rearranged":(2*mean+rearranged)/3, "temperature":match_entropy(mean,corrected)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=["sketchy","tuberlin"],default="sketchy")
    parser.add_argument("--official",action="store_true")
    args = parser.parse_args()
    w = Path("/content/sbir")
    result = run(w,args.dataset,args.official,internal=True,strong=True,
                 witness_fn=ranked_views,internal_methods=("mean","raw","isotonic","rearranged","temperature"),
                 pairs=[("isotonic",r) for r in ("mean","raw","rearranged","temperature")])
    root = w/"results/rank-preserving-crop-calibration"
    root.mkdir(exist_ok=True,parents=True)
    suffix="-official" if args.official else ""
    (root/(args.dataset+suffix+".json")).write_text(json.dumps(result,indent=2))
