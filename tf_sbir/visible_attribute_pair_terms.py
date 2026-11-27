"""Account for precisely what a known pair of opposing sentences does on the same photo pairs."""

import argparse
import json

import torch

from .concept_benchmark import atomic_json,checksum
from .visible_attribute_panels import OUT
from .visible_attribute_readout import CONDITIONS,class_estimate,discriminate,half_credit,retrieve


def terms(scores,labels):
    a=scores[labels==1]
    b=scores[labels==-1]
    diff=a[:,None,:]-b[None,:,:]
    common=(diff[:,:,0]+diff[:,:,1])/2
    content=(diff[:,:,0]-diff[:,:,1])/2
    wins_a=half_credit(diff[:,:,0])
    wins_b=half_credit(-diff[:,:,1])
    both=wins_a*wins_b
    neither=(1-wins_a)*(1-wins_b)
    identical=wins_a*(1-wins_b)+(1-wins_a)*wins_b
    known=.5*(wins_a+wins_b).mean()
    contrast=half_credit(content).mean()
    if not torch.allclose(both+neither+identical,torch.ones_like(both)):raise ValueError("Missing outcome")
    if (torch.stack((content+common,common-content),2)-diff).abs().max()>1e-12:
        raise ValueError("Score decomposition failed")
    return {"both_correct":both.mean().item(),"both_wrong":neither.mean().item(),
            "same_photo_for_both":identical.mean().item(),"supplied_state_raw":known.item(),
            "supplied_state_contrast":contrast.item(),"spread_beats_common":(content.abs()>common.abs()).double().mean().item()}


def compare(split_entry):
    q=split_entry["scores"]["sketch"]
    p=split_entry["scores"]["photo"]
    sl=split_entry["labels"]["sketch"]
    pl=split_entry["labels"]["photo"]
    states=torch.where(sl==pl,sl,0)
    if min((states==i).sum().item() for i in (1,-1))<2:return None
    result={}
    for ix,key in enumerate(CONDITIONS):
        q_margin=q[:,ix,0]-q[:,ix,1]
        p_margin=p[:,ix,0]-p[:,ix,1]
        change=torch.stack((p_margin,-p_margin),dim=1)
        raw=retrieve(q_margin,p[:,ix],states)
        contrasting=retrieve(q_margin,change,states)
        outcome=terms(p[:,ix],states)
        photo_auc=discriminate(p_margin,states)["auc"]
        if max(abs(raw["known_state_text"]-outcome["supplied_state_raw"]),
               abs(photo_auc-outcome["supplied_state_contrast"]),
               abs(photo_auc-contrasting["known_state_text"]))>1e-12:
            raise ValueError("Source-exclusion identity failed")
        result[key]={**outcome,"sketch_selected_raw":raw["nominated_text"],"sketch_selected_contrast":contrasting["nominated_text"],
                     "known_minus_sketch_contrast":photo_auc-contrasting["nominated_text"],
                     "contrast_minus_single_known":photo_auc-raw["known_state_text"],
                     "contrast_minus_single_selected":contrasting["nominated_text"]-raw["nominated_text"]}
    return result


def run(revision):
    data=torch.load(OUT/"scores.pt",weights_only=True,mmap=True)
    ref=json.loads((OUT/"summary.json").read_bytes())
    if checksum(OUT/"scores.pt")!=ref["raw_sha256"]:raise ValueError("Unverified endpoints")
    rows={}
    for cls,item in data.items():
        result=compare(item)
        if result is not None:rows[cls]=result
    if set(rows)!=set(ref["summary"]["retrieval"]["sketch"]["eligible_classes"]):raise ValueError("Different population")
    metrics=next(iter(rows.values()))["alone"].keys()
    summaries={k:{m:class_estimate([r[k][m] for r in rows.values()]) for m in metrics} for k in CONDITIONS}
    atomic_json(OUT/"pair-terms.json",{"revision":revision,"score_revision":ref["revision"],"raw_sha256":ref["raw_sha256"],
                "classes":list(rows),"class_estimates":summaries,"rows":rows})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision",required=True)
    run(parser.parse_args().revision)
