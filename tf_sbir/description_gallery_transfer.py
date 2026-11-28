"""Test whether the gallery descriptions chosen with a class oracle transfer to disjoint photos."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from types import SimpleNamespace

import open_clip
import torch
from torch.nn.functional import normalize

from .branch_view_study import PrefixReplay
from .data import UNSEEN_CLASSES
from .description_block_results import Population
from .description_choice_blocks import OUT as BLOCKS, PILOT, inputs
from .encoding import load_frozen_clip
from .gallery_name_rank_decomposition import semantic_inputs
from .kaggle_unseen_features import DEFAULT_CACHE
from .locked_geometry_words import score_rows
from .native_block_probe import SEED, deterministic
from .photo_name_evaluation import KD
from .photo_name_interaction import OUT as NAME_INDEX, PARTS, ordered_shards, record, verified
from .quickdraw_locked import KD_FUNCTION_SHA
from .replicated_simplex_study import evaluator
from .semantic_boxable import DEFAULT_PROMPTS
from .wordnet_gloss_ceiling import per_class_metrics
from .wordnet_renaming_pilot import SPLITS
from .wordnet_renaming_text import Sense, canonical, describe, encode


OUT=DEFAULT_CACHE.parent/"results/description-gallery-transfer"


def fold_indices(split,sources,gi,classes,randomized):
    chosen=torch.isin(gi,sources.unique())
    if int(chosen.sum())!=int(sources.unique().numel()) or len(gi.unique())!=len(gi):
        raise ValueError("All actual source photos must have unique gallery IDs")
    if randomized:
        alternative=torch.zeros_like(chosen)
        for c in classes.unique():
            ids=(classes==c).nonzero().flatten().tolist()
            keys=sorted(ids,key=lambda i:hashlib.sha256(f"{SEED}/matched-size/{split}/{int(gi[i])}".encode()).digest())
            alternative[keys[:int(chosen[ids].sum())]]=True
        if not torch.equal(classes[chosen].bincount(),classes[alternative].bincount()):
            raise ValueError("Gallery chance control changed the class counts")
        chosen=alternative
    if (classes[chosen].bincount()==0).any() or (classes[~chosen].bincount()==0).any():
        raise ValueError("Each side must have every true class")
    return [chosen.nonzero().flatten(),(~chosen).nonzero().flatten()]


def vocabulary(split,checkpoint):
    archive=torch.load(DEFAULT_CACHE/"text.pt",weights_only=True)
    words=archive["words"]
    catalog=json.loads((PILOT/"inventory.json").read_bytes())
    senses=[Sense(**x) for x in catalog["senses"]]
    glosses=[describe("photo",s.anonymous) for s in senses]
    distinct=list(dict.fromkeys(glosses))
    positions={s:i+len(words) for i,s in enumerate(distinct)}
    sense_row=torch.tensor([positions[s] for s in glosses])
    if (len(words),len(senses),len(distinct))!=(601,1135,1123):raise ValueError("Unexpected original dictionary")
    model=load_frozen_clip(checkpoint,torch.device("cuda"))
    descriptions=encode(model,open_clip.get_tokenizer("ViT-B-32"),distinct)
    vectors=torch.cat((archive["text"][DEFAULT_PROMPTS[1]].cuda(),descriptions))
    def spelling(s):return canonical(s).replace(" ","")
    labels={spelling(c) for c in UNSEEN_CLASSES[split]}
    forbidden_names={i for i,w in enumerate(words) if spelling(w) in labels}
    forbidden_glosses=set()
    for s,idx in zip(senses,sense_row.tolist(),strict=True):
        if any(spelling(alias) in labels for alias in s.aliases):
            forbidden_names.update(s.public_ids)
            forbidden_glosses.add(idx)
    keep=torch.ones(len(vectors),dtype=torch.bool)
    keep[sorted(forbidden_names|forbidden_glosses)]=False
    is_name=torch.arange(len(vectors))<len(words)
    candidates={"names":is_name,"glosses":~is_name,"names_filtered":is_name&keep,"glosses_filtered":~is_name&keep}
    name_prompts=[DEFAULT_PROMPTS[1].format(w) for w in words]
    return vectors,name_prompts+distinct,sense_row,candidates


def measure_baseline(q,p,positions,labels,classes,split):
    gallery=classes[positions]
    replay=PrefixReplay(evaluator(KD),gallery,split)
    gt=gallery.cuda()
    bank=p[positions].T.contiguous()
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0,len(q),384):
            sl=slice(start,start+384)
            rows.append(score_rows(q[sl]@bank,labels[sl],gt,split,replay,pool)[:,:2].double())
    return torch.cat(rows)


def winners(table,mask):
    return table[:,:,0].masked_fill(~mask[:,None],-torch.inf).argmax(0)


def score_transfer(utility,ref_utility,baseline_rows,labels,sources,candidates,prompt_list,picks):
    pop=Population(labels,sources)
    rows={"semantic":baseline_rows}
    selected={}
    for bank,mask in candidates.items():
        a=winners(ref_utility,mask)
        b=winners(utility,mask)
        if not mask[a].all() or not mask[b].all():raise ValueError("Selector escaped its declared bank")
        rows[bank+"_transfer"]=utility[a[labels],labels]
        rows[bank+"_test_oracle"]=utility[b[labels],labels]
        if (rows[bank+"_transfer"][:,0]>rows[bank+"_test_oracle"][:,0]+1e-12).any():
            raise ValueError("A transfer selection exceeded its same-gallery AP oracle")
        selected[bank]={"reference_indices":a.tolist(),"test_indices":b.tolist(),
                        "reference_winners":[prompt_list[i] for i in a],"test_winners":[prompt_list[i] for i in b],
                        "unchanged_id":int((a==b).sum())}
    for role,ids in picks.items():rows[role+"_first"]=utility[ids,labels]
    table={k:(100*v.mean(0)).tolist() for k,v in rows.items()}
    effects={k:pop.effect(v-baseline_rows) for k,v in rows.items() if k.endswith("_transfer")}
    regret={k:pop.effect(rows[k+"_test_oracle"]-rows[k+"_transfer"]) for k in candidates}
    return rows,{"percent":table,"transfer_minus_semantic":effects,"test_oracle_minus_transfer":regret,"selected":selected}


@torch.inference_mode()
def run(split,randomized,revision,name_revision,native_revision,choice_revision):
    meta,idx,_=inputs()
    qi,labels,gi,classes,sources,_,q,p,_=semantic_inputs(split,name_revision,native_revision)
    photos=ordered_shards([NAME_INDEX/f"names-{i}.pt" for i in range(PARTS)],name_revision,torch.arange(len(idx["photo"]["paths"])))
    image=normalize(photos["native"][gi].cuda(),dim=1)
    parts=fold_indices(split,sources,gi,classes,randomized)
    vec,sentences,meaning,masks=vocabulary(split,meta["checkpoint"])
    past=verified(BLOCKS/f"{split}-free.pt",choice_revision)
    if not torch.equal(past["query_ids"],qi) or not torch.equal(past["gallery_ids"],gi):
        raise ValueError("Incorrect archived sketch and photo choices")
    picks={r:meaning[past[r][:,-1,0].long()] for r in ("sketch_two","source")}
    tables=[]
    controls=[]
    for fold in parts:
        spec=SimpleNamespace(name=split,photo_labels=classes[fold],sketch_labels=labels)
        table=per_class_metrics(vec,image[fold],spec)
        tables.append(table)
        controls.append(measure_baseline(q,p,fold,labels,classes,split))
    sets=("random_small","random_large") if randomized else ("source_photos","other_photos")
    rows={}
    summary={}
    kd=evaluator(KD)
    checks=[]
    for target in range(2):
        arm=sets[1-target]+"_to_"+sets[target]
        fold=parts[target]
        r,s=score_transfer(tables[target],tables[1-target],controls[target],labels,sources,masks,sentences,picks)
        rows[arm],summary[arm]=r,s
        # Check actual CLIP vectors and class targets for all transferred winners, including filtered banks.
        errors=[]
        for bank in masks:
            for c,ind in enumerate(s["selected"][bank]["reference_indices"]):
                ap,prec,_,_=kd(vec[ind:ind+1],image[fold],torch.tensor([c]),classes[fold],split)
                errors.append((torch.stack((ap,prec)).double()-tables[target][ind,c]).abs())
        checks.append(torch.stack(errors).amax(0).tolist())
    kind="matched_random" if randomized else "source_membership"
    name=OUT/f"{split}-{kind}.pt"
    record(name,{"query_ids":qi,"gallery_ids":gi,"class_labels":labels,"gallery_labels":classes,"source_ids":sources,
                 "part_positions":parts,"sentence_table":tables,"candidate_masks":masks,"sentence_bank":sentences,
                 "nominated":picks,"metrics":rows},revision,{"split":split,"partition":kind,"queries":len(qi),
                 "gallery":len(gi),"seed":SEED,"gallery_counts":[len(f) for f in parts],
                 "class_counts":[classes[f].bincount().tolist() for f in parts],"classes":list(UNSEEN_CLASSES[split]),
                 "eligible_sentences":{k:int(v.sum()) for k,v in masks.items()},"directions":summary,
                 "direct_KD_winner_max_absolute_errors":checks,"evaluator":KD_FUNCTION_SHA,
                 "name_revision":name_revision,"native_revision":native_revision,"choice_revision":choice_revision})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",choices=SPLITS,required=True)
    p.add_argument("--randomized",action="store_true")
    p.add_argument("--revision",required=True)
    p.add_argument("--name-revision",required=True)
    p.add_argument("--native-revision",required=True)
    p.add_argument("--choice-revision",required=True)
    a=p.parse_args()
    deterministic()
    run(a.split,a.randomized,a.revision,a.name_revision,a.native_revision,a.choice_revision)
