"""Hold one wrong answer fixed while testing the class specificity of the other on disjoint subjects."""

import argparse
import json

import numpy as np
import torch

from .description_gallery_transfer import OUT
from .native_block_probe import SEED, deterministic
from .native_wrong_alternatives import CHANNELS, local_choices
from .photo_name_interaction import record, verified
from .wordnet_renaming_pilot import SPLITS


def weighted_joint(row, col, classes, ids, n):
    """One vote per photo source, divided among its drawings before taking any conditional means."""
    _, group, counts = ids.unique(return_inverse=True, return_counts=True)
    weights = counts[group].double().reciprocal().cuda()
    row, col = row.cuda(), col.cuda()
    distributions = {}
    for c in classes.unique().tolist():
        take = classes == c
        bins = (n * row[take] + col[take]).sort(stable=True)
        weights_sorted = weights[take][bins.indices]
        values, repeats = bins.values.unique_consecutive(return_counts=True)
        cumulative = weights_sorted.cumsum(0)[repeats.cumsum(0) - 1]
        mass = torch.diff(cumulative, prepend=torch.zeros_like(cumulative[:1]))
        table = torch.zeros(n*n, dtype=torch.float64, device="cuda")
        table[values] = mass / weights[take].sum()
        distributions[c] = table.reshape(n,n)
    return distributions


def compare(tables):
    # Each matrix has conditioning answer on rows and measured answer on columns.
    mass = torch.stack([t.sum(1) for t in tables])
    overlap = mass.amin(0)
    common = overlap > 0
    ac, ad, bc, bd = (t[common] / m[common,None] for t,m in zip(tables,mass))
    same = (ac * bc + ad * bd).sum(1) / 2
    different = (ac * bd + ad * bc).sum(1) / 2
    identity = ((ac - ad) * (bc - bd)).sum(1) / 2
    if not torch.allclose(same - different, identity, atol=1e-12, rtol=0):
        raise ValueError("Conditional class contrast failed its decomposition")
    gap = same - different
    matched_same = (mass[0].dot(mass[2]) + mass[1].dot(mass[3]))/2
    matched_foil = (mass[0].dot(mass[3]) + mass[1].dot(mass[2]))/2
    row = torch.stack((overlap.sum(), overlap[common].dot(same), overlap[common].dot(different),
                       overlap[common].dot(gap), matched_same, matched_foil))
    strata = torch.zeros((len(overlap),4),device="cuda",dtype=torch.float64)
    strata[:,0] = overlap
    strata[common,1:] = torch.stack((same,different,gap),1)
    return row.cpu(), strata.cpu()


def top(scores, mask):
    return scores.masked_fill(~mask, -torch.inf).argmax(1)


def pair_data(store, pool, c, d):
    scores = store["scores"]
    labels = store["class_public_name_ids"]
    visual = local_choices(pool, c, d, torch.arange(len(labels)))
    words = local_choices(pool, c, d, labels)
    joint = {}
    for prefix in ("sketch", "other"):
        classes, subjects = store["roles"][prefix]
        take = (classes == c) | (classes == d)
        photos = scores[prefix+"_RGB"][take].cuda()
        names = scores[prefix+"_names"][take].cuda()
        # Use original class indices for both observations. Duplicate text rows are globally removed by words.
        row, col = top(names, words), top(photos, visual)
        joint[prefix] = weighted_joint(row,col,classes[take],subjects[take],len(labels))
    raw = [joint[role][cat] for role,cat in (("sketch",c),("sketch",d),("other",c),("other",d))]
    return {"RGB_at_fixed_name":compare(raw),"name_at_fixed_RGB":compare([v.T for v in raw])}


def summary(rows):
    overlap = rows[:,0]
    present = overlap > 0
    if not present.any():
        return {"covered_focal_classes":0, "mean_overlap_percent":0.0}
    standardized = rows[present,1:4] / overlap[present,None]
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(standardized),size=(10000,len(standardized)))
    ci = np.quantile(standardized.numpy()[draws].mean(1),(.025,.975),axis=0)
    return {
        "covered_focal_classes":int(present.sum()), "positive_gap_classes":int((standardized[:,2]>0).sum()),
        "mean_overlap_percent":100*float(overlap.mean()),"min_max_overlap_percent":(100*torch.stack((overlap.min(),overlap.max()))).tolist(),
        "equal_focal_percent":(100*standardized.mean(0)).tolist(), "equal_focal_interval95":(100*ci).tolist(),
        "overlap_weighted_percent":(100*rows[:,1:4].sum(0)/overlap.sum()).tolist(),
        "matching_conditioner_prob_percent_same_different":(100*rows[:,4:].mean(0)).tolist(),
    }


def run(split, revision, rival_revision):
    stores = [verified(OUT/f"wrong-alternatives-{split}-{i}.pt",rival_revision) for i in (0,1)]
    meta = json.loads((OUT/f"wrong-alternatives-{split}-0.json").read_bytes())
    modes = ("RGB_at_fixed_name","name_at_fixed_RGB")
    result, material = {}, {}
    maximum_marginal_error = 0.0
    for panel,pool in meta["pools"].items():
        values = {mode:[] for mode in modes}
        for held,store in enumerate(stores):
            rows = {mode:[] for mode in modes}
            for c in pool:
                d = store["hard_foils"][panel][c]
                for mode,(r,strata) in pair_data(store,pool,c,d).items():
                    conditioner = "sketch_names_to_other_names" if mode == "RGB_at_fixed_name" else "sketch_RGB_to_other_RGB"
                    old_i = list(CHANNELS).index(conditioner)
                    expected = store["pair_tables"][panel][c,d,old_i,0,:2]
                    residual = float((r[4:] - expected).abs().max())
                    maximum_marginal_error = max(maximum_marginal_error, residual)
                    if residual > 1e-11:
                        raise ValueError("Joint observations did not recover the recorded original marginals")
                    rows[mode].append(r)
                    material[(panel,held,c,d,mode)] = strata
            for mode in modes:
                values[mode].append(torch.stack(rows[mode]))
        result[panel] = {}
        for mode,v in values.items():
            folds = torch.stack(v)
            material[(panel,mode)] = folds
            result[panel][mode] = summary(folds.mean(0))
    record(OUT/f"rival-conditional-{split}.pt", {"data":material,"query_ids":[s["query_ids"] for s in stores],
           "gallery_ids":stores[0]["gallery_ids"],"hard_foils":[s["hard_foils"] for s in stores]},
           revision,{"split":split,"classes":meta["classes"],"pools":meta["pools"],"rival_revision":rival_revision,
                     "query_count":sum(len(z["query_ids"]) for z in stores),"photo_count":len(stores[0]["gallery_ids"]),
                     "max_absolute_marginal_error":maximum_marginal_error,
                     "summary":result,"probability_columns":["same","different","same_minus_different"]})


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",choices=SPLITS,required=True)
    parser.add_argument("--revision",required=True)
    parser.add_argument("--rival-revision",required=True)
    args=parser.parse_args()
    deterministic()
    run(args.split,args.revision,args.rival_revision)
