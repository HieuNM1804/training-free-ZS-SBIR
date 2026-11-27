"""Retrospective diagnosis of class-correct versus class-incorrect leaders in the full gallery."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .concept_benchmark import atomic_json, checksum
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .encoding import save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE, index_file, load_split
from .paired_patch_interaction_probe import DST, SEED, read
from .profiled_study import true_sources
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_disputed_word_risk import query_posteriors, word_bank


def linear_readers(matrix, groups, count):
    sums = matrix.new_zeros(count,matrix.shape[1])
    sums.index_add_(0,groups,matrix)
    sizes = groups.bincount(minlength=count)
    return sums / sizes[:,None].clamp_min(1), sizes


def choose_endpoints(query, photo, labels, gallery, source):
    selected, own_scores = [], []
    index = photo.T.contiguous()
    for start in range(0,len(query),128):
        sl = slice(start,start+128)
        scores = query[sl] @ index
        scores.scatter_(1,source[sl,None],-torch.inf)
        same = labels[sl,None] == gallery[None]
        positive, good = scores.masked_fill(~same,-torch.inf).max(1)
        negative, bad = scores.masked_fill(same,-torch.inf).max(1)
        selected.append(torch.stack([good,bad],1))
        own_scores.append(positive-negative)
    return torch.cat(selected),torch.cat(own_scores)


def evidence(query, candidate, endpoints, source, labels, gallery):
    correct,rival = endpoints.T
    contrast = candidate[correct] - candidate[rival]
    identity,groups = source.unique(return_inverse=True)
    per_subject,n = linear_readers(query,groups,len(identity))
    by_class,n_subject = linear_readers(per_subject,gallery[identity],int(labels.max())+1)
    photo_by_class,_ = linear_readers(candidate,gallery,len(by_class))
    other_source = (by_class[labels]*n_subject[labels,None]-per_subject[groups])/(n_subject[labels,None]-1)
    sibling = (per_subject[groups]*n[groups,None]-query)/(n[groups,None]-1).clamp_min(1)
    sibling[n[groups]<=1] = float("nan")
    def unrelated(means):
        return (means.sum(0)-means[labels]-means[gallery[rival]])/(len(means)-2)
    def margin(matrix):
        return (matrix*contrast).sum(1)
    own = margin(query)
    return {
        "own":own, "other_drawing_same_source_mean":margin(sibling),
        "other_source_drawings_same_class_mean":margin(other_source), "original_source_photo":margin(candidate[source]),
        "unrelated_sketch_preference":margin(unrelated(by_class)),
        "unrelated_photo_preference":margin(unrelated(photo_by_class)),
        "own_minus_unrelated_sketch_preference":own-margin(unrelated(by_class)),
        "own_minus_unrelated_photo_preference":own-margin(unrelated(photo_by_class)),
    }


def actual_class_summary(event, valid, labels):
    mask = valid & torch.isfinite(event)
    frequency = (event[mask]>0).double()
    num = int(mask.sum())
    if not num:
        return {"queries":0}
    per_class = torch.stack([(event[mask & (labels==i)]>0).double().mean() for i in labels[mask].unique()])
    g = torch.Generator().manual_seed(SEED)
    indices = torch.randint(len(per_class),(10000,len(per_class)),generator=g)
    per_class = per_class.cpu()
    return {"queries":num, "percent_preferring_positive":float(100*frequency.mean()),
            "class_equal_percent":float(100*per_class.mean()),
            "class_bootstrap95":(per_class[indices].mean(1).quantile(torch.tensor([.025,.975],dtype=torch.float64))*100).tolist()}


def examples(split, root, source, endpoints, own, votes):
    selected = {}
    wanted = {"sibling_recovers_native_error":(own<0)&(votes["other_drawing_same_source_mean"]>0),
              "sibling_repeats_native_error":(own<0)&(votes["other_drawing_same_source_mean"]<0)}
    for kind,keep in wanted.items():
        ids = torch.where(keep)[0]
        if not len(ids):
            continue
        idx = int(ids[(-own[ids]).argsort()[len(ids)//2]])
        output = {}
        for label,pos in (("source",source[idx]),("correct",endpoints[idx,0]),("rival",endpoints[idx,1])):
            output[label] = root["photo"]["paths"][int(split.gallery_ids[int(pos)])]
        output["query"] = root["sketch"]["paths"][int(split.query_ids[idx])]
        brothers = torch.where(source==source[idx])[0].tolist()
        output["other_drawings"] = [root["sketch"]["paths"][int(split.query_ids[j])] for j in brothers if j!=idx]
        output["query_margin"] = own[idx].item()
        output["sibling_mean_margin"] = votes["other_drawing_same_source_mean"][idx].item()
        selected[kind] = output
    return selected


@torch.inference_mode()
def run(args):
    split,text,_ = load_split(args.cache,args.split)
    source = true_sources(split,args.cache)
    labels = split.sketch_labels.cuda()
    gallery = split.photo_labels.cuda()
    if (source<0).any() or not torch.equal(gallery[source],labels):
        raise ValueError("Known source mismatch")
    native = F.normalize(F.normalize(split.sketch.float(),dim=1)+F.normalize(split.mirror.float(),dim=1),dim=1)
    photos = F.normalize(split.photo.float(),dim=1)
    qs = torch.cat([query_posteriors(a,b,text[DEFAULT_PROMPTS[0]],split.scale)
                    for a,b in zip(split.sketch.split(128),split.mirror.split(128),strict=True)])
    ps = word_bank(split.photo,text[DEFAULT_PROMPTS[1]],split.scale).T.contiguous()
    readers = {"native":(native,photos), "semantic":(qs,ps)}
    result, cases = {}, {}
    manifest = read(args.cache/"manifest.json")["datasets"]["sketchy"]
    idx = torch.load(index_file(args.cache,"sketchy",manifest),weights_only=True)
    saved = {}
    for selector,(q,p) in readers.items():
        endpoints, own = choose_endpoints(q,p,labels,gallery,source)
        selected = {key:evidence(src,tgt,endpoints,source,labels,gallery) for key,(src,tgt) in readers.items()}
        if (selected[selector]["own"]-own).abs().max().item()>2e-5:
            raise ValueError("Pair differences do not replay the selected endpoints")
        named = selected[selector]
        masks = {"leader_wrong":own<0,"leader_right":own>0}
        result[selector] = {
            "queries":len(own),"ties":int((own==0).sum()),"wrong":int((own<0).sum()),"right":int((own>0).sum()),
            "comparisons":{group:{key:{k:actual_class_summary(m,v,labels) for k,m in measurements.items()}
                                  for key,measurements in selected.items()} for group,v in masks.items()}}
        cases[selector] = examples(split,idx,source,endpoints,own,named)
        saved[selector] = {"endpoints":endpoints.cpu(), "native_margin":selected["native"]["own"].cpu(),
                           "semantic_margin":selected["semantic"]["own"].cpu(),"own_selected_margin":own.cpu(),
                           "secondary":{name:val.cpu() for name,val in named.items() if name!="own"}}
    artifact = args.output/f"real-errors-{args.split}.pt"
    save_tensor({"query_ids":split.query_ids,"gallery_ids":split.gallery_ids,"labels":split.sketch_labels,
                 "source":source.cpu(),"measurements":saved},artifact)
    atomic_json(artifact.with_suffix(".json"),{"revision":args.revision,"split":args.split,"classes":len(UNSEEN_CLASSES[args.split]),
        "queries":len(source),"gallery":len(gallery),"results":result,"examples":cases,"per_query_sha256":checksum(artifact),
        "design":"Post-hoc original-source-excluded class leaders. Reference sketches, source associations and labels are diagnostic only. No ranking uses other sketches."})
    print(args.split,"done",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",choices=SKETCHY_SPLITS,required=True)
    p.add_argument("--cache",type=Path,default=DEFAULT_CACHE)
    p.add_argument("--output",type=Path,default=DST)
    p.add_argument("--revision",required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("highest")
    run(args)
