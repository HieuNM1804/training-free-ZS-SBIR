"""Preserve every true category-by-noun count when randomizing gallery noun assignments."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib

import torch

from .branch_view_study import PrefixReplay
from .description_block_results import Population
from .description_choice_blocks import inputs
from .gallery_name_rank_decomposition import OUT, attribution, joined, semantic_inputs
from .native_block_probe import SEED, deterministic
from .photo_name_evaluation import KD
from .photo_name_interaction import record, verified
from .replicated_simplex_study import evaluator
from .wordnet_renaming_pilot import SPLITS


REPLICATES = 8


def random_groups(split,classes,names):
    before = (classes*601+names).bincount(minlength=(int(classes.max())+1)*601)
    groups = []
    for rep in range(REPLICATES):
        result = names.clone()
        for c in classes.unique():
            key = hashlib.sha256(f"{SEED}/{split}/{rep}/{int(c)}".encode()).digest()
            rng = torch.Generator().manual_seed(int.from_bytes(key[:8],"little")%(2**63))
            ids = torch.where(classes==c)[0]
            result[ids] = names[ids[torch.randperm(len(ids),generator=rng)]]
        after = (classes*601+result).bincount(minlength=len(before))
        if not torch.equal(before,after):
            raise ValueError("Label-by-word counts changed")
        groups.append(result)
    return torch.stack(groups)


@torch.inference_mode()
def run(split,part,revision,rank_revision,name_revision,native_revision):
    qi,labels,gi,classes,_,names,q,p,b=semantic_inputs(split,name_revision,native_revision)
    old=verified(OUT/f"{split}-{part}.pt",rank_revision)
    take=torch.arange(part,len(qi),2)
    if not torch.equal(qi[take],old["ids"]) or not torch.equal(gi,old["gallery_ids"]) or not torch.equal(names.cpu(),old["groups"]):
        raise ValueError("Previous ranking population changed")
    groups=random_groups(split,classes,names.cpu())
    banks=groups.cuda()
    replay=PrefixReplay(evaluator(KD),classes,split)
    outer,inner=[],[]
    pt,bt=p.T.contiguous(),b.T.contiguous()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0,len(take),192):
            ix=take[start:start+192]
            raw=(q[ix]@pt).topk(len(gi),dim=1).indices
            donor=(q[ix]@bt).topk(len(gi),dim=1).indices
            if start==0:
                for order,k in ((raw,"raw"),(donor,"donors")):
                    again=replay.score(order,labels[ix],pool)
                    if not torch.equal(again.double(),old["rows"][k][:len(ix)]):
                        raise ValueError("Original rank replay drift")
            outer.append(torch.stack([replay.score(joined(donor,raw,g),labels[ix],pool) for g in banks],1))
            inner.append(torch.stack([replay.score(joined(raw,donor,g),labels[ix],pool) for g in banks],1))
    record(OUT/f"classnull-{split}-{part}.pt",{"ids":qi[take],"outer":torch.cat(outer),"inner":torch.cat(inner),
           "group_assignments":groups,"gallery_ids":gi},revision,{"replicates":REPLICATES,"queries":len(take),
           "gallery":len(gi),"rank_revision":rank_revision})


def summarize(split,revision,rank_revision):
    old=verified(OUT/f"summary-{split}.pt",rank_revision)
    qi=old["query_ids"]
    rows=old["rows"]
    actual=attribution(rows,"named")[0]
    pop=Population(old["labels"],old["source_ids"])
    tensors={k:torch.empty((len(qi),REPLICATES,2),dtype=torch.float64) for k in ("outer","inner")}
    assignment=None
    for i in range(2):
        shard=verified(OUT/f"classnull-{split}-{i}.pt",revision)
        if not torch.equal(qi[i::2],shard["ids"]) or not torch.equal(old["gallery_ids"],shard["gallery_ids"]):
            raise ValueError("Wrong held population")
        if assignment is not None and not torch.equal(assignment,shard["group_assignments"]):
            raise ValueError("Random control changed between queries")
        assignment=shard["group_assignments"]
        for k in tensors:tensors[k][i::2]=shard[k].double()
    raw,full=rows["raw"][:,None,:],rows["donors"][:,None,:]
    outer=((tensors["outer"]-raw)+(full-tensors["inner"]))/2
    within=((tensors["inner"]-raw)+(full-tensors["outer"]))/2
    if not torch.allclose(outer+within,full-raw,atol=1e-12,rtol=0):
        raise ValueError("Attribution failed")
    info={"replicates":REPLICATES,"split":split,"queries":len(qi),"gallery":len(old["gallery_ids"]),
          "control_per_replicate":(100*outer.mean(0)).tolist(),"mean_control_outer":pop.effect(outer.mean(1)),
          "mean_control_within":pop.effect(within.mean(1)),
          "actual_minus_control_outer":pop.effect(actual-outer.mean(1)),
          "fraction_same_name":(assignment==old["name_groups"][None,:]).double().mean(1).tolist()}
    record(OUT/f"class-control-{split}.pt",{"query_ids":qi,"gallery_ids":old["gallery_ids"],"outer":tensors["outer"],
           "inner":tensors["inner"],"group_assignments":assignment},revision,info)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",choices=SPLITS,required=True)
    p.add_argument("--part",type=int,choices=(0,1))
    p.add_argument("--revision",required=True)
    p.add_argument("--rank-revision",required=True)
    p.add_argument("--name-revision",required=True)
    p.add_argument("--native-revision",required=True)
    a=p.parse_args()
    deterministic()
    if a.part is None:summarize(a.split,a.revision,a.rank_revision)
    else:run(a.split,a.part,a.revision,a.rank_revision,a.name_revision,a.native_revision)
