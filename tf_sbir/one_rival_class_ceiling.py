"""Class-shared single-foil upper bounds and exact certificates on observed neighbor gains."""

import argparse
from concurrent.futures import ThreadPoolExecutor

import torch

from .branch_view_study import PrefixReplay
from .description_block_results import Population
from .description_gallery_transfer import OUT
from .gallery_name_rank_decomposition import semantic_inputs
from .native_block_probe import deterministic
from .one_rival_ceiling import best_rival, witness
from .photo_name_evaluation import KD
from .photo_name_interaction import record, verified
from .replicated_simplex_study import evaluator
from .wordnet_renaming_pilot import SPLITS


def class_map(values, classes, reference):
    means = torch.stack([values[reference & (classes == c)].mean(0) for c in classes.unique()])
    if not torch.isfinite(means).all():
        raise ValueError("All eligible classes need reference queries")
    return best_rival(means,classes.unique())


@torch.inference_mode()
def run(args):
    split = args.split
    qi,labels,gi,classes,src,_,q,p,_ = semantic_inputs(split,args.name_revision,args.native_revision)
    summaries = verified(OUT/f"one-rival-summary-{split}.pt",args.ceiling_revision)
    original = summaries["measured"]
    if not torch.equal(qi,summaries["query_ids"]) or not torch.equal(gi,summaries["gallery_ids"]):
        raise ValueError("The complete retrieval populations differ")
    parts = [verified(OUT/f"one-rival-{split}-{i}.pt",args.ceiling_revision) for i in (0,1)]
    positions = torch.cat([v["positions"] for v in parts])
    restore = positions.argsort()
    values = torch.cat([v["optimum"] for v in parts])[restore]
    counts = torch.cat([v["positive_count"] for v in parts])[restore]
    if not torch.equal(positions[restore],torch.arange(len(qi))):
        raise ValueError("Incomplete per-rival score cube")
    folds = torch.full_like(qi,-1)
    for held in (0,1):
        f = verified(OUT/f"wrong-alternatives-{split}-{held}.pt",args.rival_revision)
        mask = torch.isin(qi,f["query_ids"])
        if int(mask.sum()) != len(f["query_ids"]) or (folds[mask] != -1).any():
            raise ValueError("Duplicate or missing fold identities")
        folds[mask] = held
    if (folds < 0).any() or torch.isin(src[folds == 0].unique(),src[folds == 1].unique()).any():
        raise ValueError("Reference and held physical subjects overlap")
    pooled = class_map(values,labels,torch.ones_like(labels,dtype=torch.bool))
    cross = torch.stack([class_map(values,labels,folds != i) for i in (0,1)])
    foils = {"pooled_class":pooled[labels],"other_half_class":cross[folds,labels]}
    results = {key:torch.empty((len(qi),2),dtype=torch.float64) for key in foils}
    replay = PrefixReplay(evaluator(KD),classes,split)
    gallery = classes.cuda()
    bank = p.T.contiguous()
    max_error = 0.
    with ThreadPoolExecutor(max_workers=6) as pool:
        # Reuse the exact parity layout and matrix shape that fixed the original scores and ties.
        for part in (0,1):
            for ids in torch.arange(part,len(qi),2).split(192):
                order = (q[ids]@bank).topk(len(gi),dim=1).indices
                ranked = gallery[order]
                for key,choice in foils.items():
                    f = choice[ids]
                    moved = witness(order,ranked,labels[ids].cuda(),f.cuda(),counts[ids,f].cuda())
                    if split == "sketchy_2":
                        moved = moved[:,:200]
                    actual = replay.score(moved,labels[ids],pool).double()
                    max_error = max(max_error,float((actual-values[ids,f]).abs().max()))
                    results[key][ids] = actual
    if max_error > 2e-6:
        raise ValueError("Actual KD results do not attain class-shared bounds")
    tolerance = 2e-6
    ceiling = original["best_one_rival"][:,0]
    neighbors = original["neighbors_64"][:,0]
    raw = original["semantic"][:,0]
    if any((z[:,0]>ceiling+tolerance).any() for z in results.values()):
        raise ValueError("A class-shared rule exceeded the query-adaptive optimum")
    beyond = neighbors > ceiling + tolerance
    large_gain = neighbors >= raw + .04
    pop = Population(labels,src)
    reference = {"pooled_classes":pooled.tolist(),"other_fold_classes":cross.tolist()}
    for k,rows in results.items():
        reference[k] = {"mean_percent":(100*rows.mean(0)).tolist(),"delta_vs_semantic":pop.effect(rows-original["semantic"]),
                        "difference_vs_query_optimum":pop.effect(rows-original["best_one_rival"])}
    reference["neighbor_certificate"] = {
        "count":int(beyond.sum()),"all_queries":len(qi),"fraction":pop.effect(beyond),
        "mean_positive_excess_AP_points":100*float((neighbors-ceiling).clamp_min(0).mean()),
        "queries_with_neighbor_gain_at_least_four":int(large_gain.sum()),
        "certified_among_those":int((beyond & large_gain).sum()),
        "tolerance":tolerance,
    }
    record(OUT/f"shared-rival-{split}.pt", {"query_ids":qi,"sources":src,"labels":labels,"gallery_ids":gi,
           "folds":folds,"foils":foils,"original_KD":results,"above_best_one_rival":beyond},
           args.revision,{"split":split,"queries":len(qi),"gallery":len(gi),"ceiling_revision":args.ceiling_revision,
           "rival_revision":args.rival_revision,"witness_max_error":max_error,"summary":reference})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",required=True,choices=SPLITS)
    for key in ("revision","rival-revision","ceiling-revision","name-revision","native-revision"):
        parser.add_argument("--"+key,required=True)
    args = parser.parse_args()
    deterministic()
    run(args)
