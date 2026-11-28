"""Separate availability of transferable target descriptions from choosing among them."""

import argparse
import hashlib
import json

import open_clip
import torch
from torch.nn.functional import normalize

from .description_block_results import Population
from .description_choice_blocks import OUT as BLOCKS, PILOT, inputs
from .description_gallery_transfer import OUT, winners
from .encoding import load_frozen_clip
from .kaggle_unseen_features import DEFAULT_CACHE
from .native_block_probe import SEED, deterministic
from .photo_name_interaction import OUT as NAME_INDEX, PARTS, ordered_shards, record, verified
from .semantic_boxable import DEFAULT_PROMPTS
from .wordnet_renaming_pilot import SPLITS
from .wordnet_renaming_text import Sense, describe, encode


def source_bank(checkpoint,original):
    text=torch.load(DEFAULT_CACHE/"text.pt",weights_only=True)
    catalog=json.loads((PILOT/"inventory.json").read_bytes())
    senses=[Sense(**i) for i in catalog["senses"]]
    mapping={}
    for s in senses:
        source=describe("drawing",s.anonymous)
        target=describe("photo",s.anonymous)
        if target in mapping and mapping[target]!=source:raise ValueError("Inconsistent domain counterpart")
        mapping[target]=source
    if original[:601]!=[DEFAULT_PROMPTS[1].format(w) for w in text["words"]]:
        raise ValueError("Candidate names have changed")
    model=load_frozen_clip(checkpoint,torch.device("cuda"))
    tokenizer=open_clip.get_tokenizer("ViT-B-32")
    descriptions=encode(model,tokenizer,[mapping[target] for target in original[601:]])
    return torch.cat((text["text"][DEFAULT_PROMPTS[0]].cuda(),descriptions))


def observers(qi,source,photo_count,native_revision,name_revision):
    ids=[];orig=[];flips=[]
    for i in range(PARTS):
        a=verified(BLOCKS/f"native-sketch-{i}.pt",native_revision)
        ids.append(a["ids"]);orig.append(a["rgb"]["native"]);flips.append(a["mirror"]["native"])
    ids=torch.cat(ids)
    if not torch.equal(ids.sort().values,torch.arange(len(ids))):raise ValueError("Sketch IDs changed")
    idx=ids.argsort()[qi]
    actual=torch.cat(orig)[idx].cuda()
    flip=torch.cat(flips)[idx].cuda()
    pictures=ordered_shards([NAME_INDEX/f"names-{i}.pt" for i in range(PARTS)],name_revision,torch.arange(photo_count))
    return {"sketch":normalize(actual,dim=1),"sketch_two":normalize(actual+flip,dim=1),
            "source":normalize(pictures["native"][source].cuda(),dim=1)}


def read_menu(scores,indices):
    return indices[scores[:,indices].argmax(1)]


def random_menus(pool,n,bank,split):
    seed=int.from_bytes(hashlib.sha256(f"{SEED}/uniform-menus/{split}/{bank}".encode()).digest()[:8],"little")%(2**63)
    rng=torch.Generator().manual_seed(seed)
    return [pool[torch.randperm(len(pool),generator=rng)[:n]].sort().values for _ in range(16)]


@torch.inference_mode()
def run(split,revision,table_revision,name_revision,native_revision):
    path=OUT/f"{split}-source_membership.pt"
    prior=verified(path,table_revision)
    meta,idx,_=inputs()
    qi,labels,source=(prior[k] for k in ("query_ids","class_labels","source_ids"))
    pop=Population(labels,source)
    fold,test=prior["part_positions"]
    if torch.isin(prior["gallery_ids"][test],source).any() or len(fold)!=len(source.unique()):
        raise ValueError("A query source was included in the test gallery")
    ref,held=prior["sentence_table"]
    vectors=source_bank(meta["checkpoint"],prior["sentence_bank"])
    seen=observers(qi,source,len(idx["photo"]["paths"]),native_revision,name_revision)
    logits={k:v@vectors.T for k,v in seen.items()}
    sem=prior["metrics"]["source_photos_to_other_photos"]["semantic"]
    query_labels=labels.cuda()
    target=held.cuda()
    outputs={};summaries={}
    for bank,mask in prior["candidate_masks"].items():
        win=winners(ref,mask)
        nominated=win.unique().sort().values
        pool=mask.nonzero().flatten()
        choices=nominated.cuda()
        full=pool.cuda()
        controls=random_menus(pool,len(choices),bank,split)
        rand=[c.cuda() for c in controls]
        informed=held[win[labels],labels]
        uniform=held[nominated].mean(0)[labels]
        table={"semantic":sem,"class_winner":informed,"uniform_in_winner_menu":uniform}
        decisions={}
        effects={}
        for role,score in logits.items():
            ids=read_menu(score,choices)
            unrestricted=read_menu(score,full)
            metrics=target[ids,query_labels].cpu()
            ref_random=torch.stack([target[read_menu(score,c),query_labels] for c in rand]).mean(0).cpu()
            table[role+"_menu"]=metrics
            table[role+"_unrestricted"]=target[unrestricted,query_labels].cpu()
            table[role+"_random_menu"]=ref_random
            decisions[role]=dict(menu=ids.cpu(),unrestricted=unrestricted.cpu())
            effects[role]={k:pop.effect(metrics-table[v]) for k,v in (("minus_all",role+"_unrestricted"),
                           ("minus_random",role+"_random_menu"),("minus_semantic","semantic"),("minus_class_winner","class_winner"))}
            effects[role]["picks_class_associated_winner"]=pop.effect(ids.cpu()==win[labels])
        effects["source_minus_sketch_two_in_menu"]=pop.effect(table["source_menu"]-table["sketch_two_menu"])
        outputs[bank]=dict(winning_candidate_by_class=win,menu=nominated,random_menus=torch.stack(controls),
                           decisions=decisions,metrics=table)
        summaries[bank]={"available":len(pool),"menu_size":len(choices),"chance_class_winner_percent":100/len(choices),
                         "mean_percent":{k:(100*v.mean(0)).tolist() for k,v in table.items()},"effects":effects}
    if prior["candidate_masks"]["glosses"].sum()!=1123 or len(sem)!=len(qi):raise ValueError("Input census changed")
    record(OUT/f"menu-{split}.pt",{"query_ids":qi,"source_ids":source,"class_labels":labels,
           "held_gallery_ids":prior["gallery_ids"][test],"banks":outputs},revision,{"split":split,"queries":len(qi),
           "held_gallery":len(test),"banks":summaries,"table_revision":table_revision,
           "name_revision":name_revision,"native_revision":native_revision})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",choices=SPLITS,required=True)
    p.add_argument("--revision",required=True)
    p.add_argument("--table-revision",required=True)
    p.add_argument("--name-revision",required=True)
    p.add_argument("--native-revision",required=True)
    args=p.parse_args()
    deterministic()
    run(args.split,args.revision,args.table_revision,args.name_revision,args.native_revision)
