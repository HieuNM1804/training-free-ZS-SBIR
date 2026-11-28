"""Reassign the completed index within each printed noun without changing its distribution."""

import argparse
from concurrent.futures import ThreadPoolExecutor

import torch

from .description_block_choices import order_key
from .description_block_results import Population
from .description_choice_blocks import inputs
from .native_block_probe import deterministic
from .photo_name_evaluation import KD, gallery, queries
from .photo_name_interaction import OUT, record, verified
from .replicated_simplex_study import evaluator
from .wordnet_renaming_pilot import SPLITS


def within_word(split, ids, names):
    result = torch.empty_like(ids)
    for word in names.unique():
        subset = torch.where(names==word)[0].tolist()
        ordered = torch.tensor(sorted(subset,key=lambda i:order_key("within-word-image",split,int(ids[i]))))
        result[ordered] = ordered.roll(1)
    if not torch.equal(names[result],names) or not torch.equal(result.sort().values,torch.arange(len(ids))):
        raise ValueError("Wrong conditional permutation")
    return result


@torch.inference_mode()
def run(split,part,revision,gallery_revision,native_revision):
    _,index,_=inputs()
    qi,labels=index["sketch"]["views"][split]
    gi,classes=index["photo"]["views"][split]
    take=torch.arange(part,len(qi),2)
    x=queries(qi[take],native_revision)
    f=gallery(split,gi,gallery_revision)
    perm=within_word(split,gi,f["name"].cpu())
    y=f["interaction"][perm.cuda()]
    kd=evaluator(KD)
    def measure(i):
        with torch.inference_mode():
            ap,precision,_,_=kd(x[i:i+1],y,labels[take[i:i+1]],classes,split)
        return torch.stack((ap,precision))
    with ThreadPoolExecutor(max_workers=4) as pool:
        metrics=torch.stack(list(pool.map(measure,range(len(take)))))
    record(OUT/f"reassign-{split}-{part}.pt",{"ids":qi[take],"metrics":metrics,"permutation":perm},
           revision,{"unchanged":int((perm==torch.arange(len(gi))).sum()),"gallery":len(gi),"queries":len(take)})


def summarize(split,revision,official_revision):
    main=verified(OUT/f"official-{split}.pt",official_revision)
    chunks=[verified(OUT/f"reassign-{split}-{i}.pt",revision) for i in range(2)]
    if not torch.equal(chunks[0]["permutation"],chunks[1]["permutation"]):raise ValueError("Different galleries")
    rows=torch.empty_like(main["scores"]["interaction"])
    for i,c in enumerate(chunks):
        if not torch.equal(c["ids"],main["query_ids"][i::2]):raise ValueError("Unequal populations")
        rows[i::2]=c["metrics"].double()
    pop=Population(main["classes"],main["source_ids"])
    perm=chunks[0]["permutation"]
    record(OUT/f"reassigned-{split}.pt",{"query_ids":main["query_ids"],"gallery_ids":main["gallery_ids"],
           "permutation":perm,"reassigned":rows},revision,{"split":split,"queries":len(rows),
           "gallery":len(perm),"unchanged_singletons":int((perm==torch.arange(len(perm))).sum()),
           "mean_percent":(100*rows.mean(0)).tolist(),
           "primary_minus_reassigned":pop.effect(main["scores"]["interaction"]-rows)})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",required=True,choices=SPLITS)
    p.add_argument("--part",type=int,choices=(0,1))
    p.add_argument("--revision",required=True)
    p.add_argument("--official-revision",required=True)
    p.add_argument("--gallery-revision",required=True)
    p.add_argument("--native-revision",required=True)
    a=p.parse_args()
    deterministic()
    if a.part is None:summarize(a.split,a.revision,a.official_revision)
    else:run(a.split,a.part,a.revision,a.gallery_revision,a.native_revision)
