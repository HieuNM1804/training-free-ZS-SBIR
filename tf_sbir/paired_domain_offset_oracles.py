"""Post-hoc, source-disjoint offset diagnostics on full original Sketchy galleries."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .conditional_completion_study import evaluate
from .data import SKETCHY_SPLITS
from .encoding import save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE, load_visual_split
from .paired_patch_interaction_probe import DST, SEED
from .profiled_study import true_sources
from .replicated_simplex_study import evaluator
from .semantic_boxable_prompt_search import mean_percent


def two_fold_references(q, p, source, gallery_labels):
    ids, group = source.unique(return_inverse=True)
    classes = gallery_labels[ids]
    count = group.bincount(minlength=len(ids))
    summed = q.new_zeros(len(ids),q.shape[1])
    summed.index_add_(0,group,q)
    # All redraws of one physical source contribute one mean with equal identity weight.
    sq = summed/count[:,None]
    sp = p[ids]
    folds = torch.empty(len(ids),dtype=torch.long,device=q.device)
    rng = torch.Generator().manual_seed(SEED)
    for c in classes.unique():
        ix = torch.where(classes==c)[0]
        order = ix[torch.randperm(len(ix),generator=rng)]
        folds[order[::2]],folds[order[1::2]] = 0,1
    sc,pc = [],[]
    for side in (0,1):
        qm,pm = [],[]
        for c in classes.unique():
            mask = (classes==c)&(folds==side)
            qm.append(sq[mask].mean(0))
            pm.append(sp[mask].mean(0))
        sc.append(torch.stack(qm))
        pc.append(torch.stack(pm))
    return torch.stack(sc),torch.stack(pc),1-folds[group],ids,folds


@torch.inference_mode()
def run(args):
    split=load_visual_split(args.cache,args.split)
    q=F.normalize(F.normalize(split.sketch.float(),dim=1)+F.normalize(split.mirror.float(),dim=1),dim=1)
    p=F.normalize(split.photo.float(),dim=1)
    source=true_sources(split,args.cache)
    if (source<0).any():
        raise ValueError("Requires original photo for this retrospective measurement")
    labels,glabels=split.sketch_labels.cuda(),split.photo_labels.cuda()
    if not torch.equal(glabels[source],labels):
        raise ValueError("Source class mismatch")
    sc,pc,other,ids,folds=two_fold_references(q,p,source,glabels)
    c=sc.shape[1]
    source_mean = (sc.sum(1)[other]-sc[other,labels])/(c-1)
    target_mean = (pc.sum(1)[other]-pc[other,labels])/(c-1)
    class_source,class_target = sc[other,labels],pc[other,labels]
    gallery_mean=p.mean(0)
    conditions={
        "native":q,
        "unlabeled_gallery_mean_subtracted":q-gallery_mean,
        "unrelated_source_sketch_mean_subtracted":q-source_mean,
        "unrelated_class_offset_added":q+target_mean-source_mean,
        "true_class_offset_added":q+class_target-class_source,
        "true_class_photo_mean_alone":class_target,
        "original_source_photo_alone":p[source],
    }
    kd=evaluator(args.evaluator)
    replay=PrefixReplay(kd,split.photo_labels,split.name)
    rows={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name,vec in conditions.items():
            rows[name]=evaluate(split,F.normalize(vec,dim=1),p,replay,pool)
            print(name,mean_percent(rows[name]),flush=True)
    artifact=args.output/f"offset-oracles-{args.split}.pt"
    save_tensor({"qi":split.query_ids,"pi":split.gallery_ids,"labels":split.sketch_labels,"source":source.cpu(),
                 "source_ids":ids.cpu(),"source_folds":folds.cpu(),"rows":rows},artifact)
    atomic_json(artifact.with_suffix(".json"),{"revision":args.revision,"seed":SEED,"queries":len(q),"gallery":len(p),
        "classes":c,"source_ids":len(ids),"cutoffs":replay.cutoffs,"metrics":{k:mean_percent(v) for k,v in rows.items()},
        "per_query_sha256":checksum(artifact),
        "design":"Original gallery and official KD evaluator. Report oracle rows only as diagnostics. "
                 "Reference class means use disjoint original photo identities. The unrelated source mean additionally excludes "
                 "the query category. True-class rows use the target label. The source photo is also present in the gallery, "
                 "as it is for every row. The input-source-photo row measures an optimistic readout with privileged data."})


if __name__=="__main__":
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--split",choices=SKETCHY_SPLITS,required=True)
    arg.add_argument("--cache",type=Path,default=DEFAULT_CACHE)
    arg.add_argument("--output",type=Path,default=DST)
    arg.add_argument("--evaluator",type=Path,default=DEFAULT_CACHE.parent/"real-sketchy-smoke"/"evaluator.py")
    arg.add_argument("--revision",required=True)
    args=arg.parse_args()
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    run(args)
