"""Cross disjoint visual category observers with explicitly independent target representations."""

import argparse
import hashlib
from types import SimpleNamespace

import torch
from torch.nn.functional import normalize

from .description_block_results import Population
from .description_choice_blocks import inputs
from .description_gallery_menu import observers, source_bank
from .description_gallery_transfer import OUT, vocabulary, winners
from .description_menu_bias import assignment, subject_sum
from .native_block_probe import SEED, deterministic
from .photo_name_evaluation import KD
from .photo_name_interaction import OUT as NAME_INDEX, PARTS, ordered_shards, record, verified
from .replicated_simplex_study import evaluator
from .wordnet_gloss_ceiling import per_class_metrics
from .wordnet_renaming_pilot import SPLITS


def shuffled_class_ids(labels,split,held):
    key=hashlib.sha256(f"{SEED}/centroid-permute/{split}/{held}".encode()).digest()
    rng=torch.Generator().manual_seed(int.from_bytes(key[:8],"little")%(2**63))
    return labels[torch.randperm(len(labels),generator=rng)]


def centers(z,cl,n):
    return normalize(torch.stack([z[cl==c].double().mean(0) for c in range(n)]).float(),dim=1)


def evaluate_vectors(vectors,bank,gallery_labels,classes,name,kd):
    vals=[]
    for c in classes:
        r=[kd(v[None,:],bank,c.reshape(1),gallery_labels,name) for v in vectors]
        vals.append(torch.stack([torch.stack(z[:2]).double() for z in r]))
    return torch.stack(vals,dim=1)


@torch.inference_mode()
def run(split,held,revision,table_revision,native_revision,name_revision):
    meta,idx,_=inputs()
    prior=verified(OUT/f"{split}-source_membership.pt",table_revision)
    qi,labels,src,gi,cl=(prior[k] for k in ("query_ids","class_labels","source_ids","gallery_ids","gallery_labels"))
    subjects,grp,cnt=src.unique(return_inverse=True,return_counts=True)
    subject_class,fold=assignment(split,grp,subjects,labels)
    source_ids=subjects[fold!=held]
    hold=fold[grp]==held
    gallery_ids=gi[prior["part_positions"][1]]
    gallery_classes=cl[prior["part_positions"][1]]
    if torch.isin(source_ids,src[hold]).any() or torch.isin(gallery_ids,subjects).any():
        raise ValueError("Reference, query sources, and scoring gallery must be disjoint")
    n=len(cl.unique())
    ref_classes=subject_class[fold!=held]
    if not torch.equal(ref_classes.bincount(),torch.full((n,),50)):raise ValueError("Expected 50 distinct reference subjects per class")
    rgb=ordered_shards([NAME_INDEX/f"names-{i}.pt" for i in range(PARTS)],name_revision,
                       torch.arange(len(idx["photo"]["paths"])))["native"]
    ref=normalize(rgb[source_ids].float().cuda(),dim=1)
    bank=normalize(rgb[gallery_ids].float().cuda(),dim=1)
    obs=observers(qi,src,len(rgb),native_revision,name_revision)
    source_drawings=normalize((subject_sum(obs["sketch_two"].double().cpu(),grp,len(subjects))/cnt[:,None]).float(),dim=1)
    sketch_cent=centers(source_drawings[fold!=held].cuda(),ref_classes,n)
    rgb_cent=centers(ref,ref_classes,n)
    shuffled=shuffled_class_ids(ref_classes,split,held)
    random_cent=centers(ref,shuffled,n)
    proto_ids=[]
    for c in range(n):
        eligible=(ref_classes==c).nonzero().flatten().tolist()
        proto_ids.append(min(eligible,key=lambda j:hashlib.sha256(f"{SEED}/one-gallery-ref/{int(source_ids[j])}".encode()).digest()))
    single_cent=ref[proto_ids]
    vec,sentences,_,masks=vocabulary(split,meta["checkpoint"])
    src_vec=source_bank(meta["checkpoint"],sentences)
    if sentences!=prior["sentence_bank"]:raise ValueError("Reference sentence bank differs")
    kt=evaluator(KD)
    util=per_class_metrics(vec,ref,SimpleNamespace(name=split,photo_labels=ref_classes,sketch_labels=labels))
    alternatives={k:winners(util,masks[k]) for k in ("names","glosses_filtered")}
    q=obs["sketch_two"][hold]
    p=obs["source"][hold]
    classes=torch.arange(n)
    estimates={
        "category_oracle":labels[hold], "one_rgb":(q@single_cent.T).argmax(1).cpu(),
        "rgb_centroid":(q@rgb_cent.T).argmax(1).cpu(),"drawing_centroid":(q@sketch_cent.T).argmax(1).cpu(),
        "photo_query_rgb_centroid":(p@rgb_cent.T).argmax(1).cpu(),
        "random_groups":(q@random_cent.T).argmax(1).cpu(),
    }
    for k,win in alternatives.items():
        estimates[k]=(q@src_vec[win].T).argmax(1).cpu()
    target_vectors={"rgb_centroid":rgb_cent,"one_rgb":single_cent,"random_groups":random_cent,
                    **{k:vec[v] for k,v in alternatives.items()}}
    tables={k:evaluate_vectors(v,bank,gallery_classes,classes,split,kt) for k,v in target_vectors.items()}
    dest=OUT/f"visual-reference-{split}-{held}.pt"
    record(dest,{"positions":hold.nonzero().flatten(),"query_ids":qi[hold],"source_ids":src[hold],"class_labels":labels[hold],
           "gallery_ids":gallery_ids,"reference_ids":source_ids,"reference_classes":ref_classes,"shuffled":shuffled,
           "single_ids":source_ids[proto_ids],"selected_texts":alternatives,"reference_utility":util,
           "predictions":estimates,"target_KD":tables},revision,{"split":split,"fold":held,"queries":int(hold.sum()),
           "subjects":len(src[hold].unique()),"gallery":len(gallery_ids),"reference_subjects":len(source_ids),
           "table_revision":table_revision,"native_revision":native_revision,"name_revision":name_revision})


def collect(split,revision,table_revision):
    prior=verified(OUT/f"{split}-source_membership.pt",table_revision)
    qi,labels,src=(prior[k] for k in ("query_ids","class_labels","source_ids"))
    p=[verified(OUT/f"visual-reference-{split}-{i}.pt",revision) for i in range(2)]
    if not torch.equal(p[0]["gallery_ids"],p[1]["gallery_ids"]) or torch.isin(p[0]["reference_ids"],p[1]["reference_ids"]).any():
        raise ValueError("Invalid fold roles")
    positions=torch.cat([v["positions"] for v in p])
    order=positions.argsort()
    if not torch.equal(positions[order],torch.arange(len(qi))) or not torch.equal(torch.cat([v["query_ids"] for v in p])[order],qi):
        raise ValueError("Every original query must occur exactly once")
    pop=Population(labels,src)
    baseline=prior["metrics"]["source_photos_to_other_photos"]["semantic"]
    summaries={}
    material={}
    choices={}
    for observer in p[0]["predictions"]:
        chosen=torch.cat([v["predictions"][observer] for v in p])[order]
        cells={}
        for decoder in p[0]["target_KD"]:
            pieces=[v["target_KD"][decoder][v["predictions"][observer],v["class_labels"]] for v in p]
            rows=torch.cat(pieces)[order]
            cells[decoder]={"mean_percent":(100*rows.mean(0)).tolist(),"minus_semantic":pop.effect(rows-baseline)}
            material[(observer,decoder)]=rows
        choices[observer]=chosen
        summaries[observer]={"class_accuracy":pop.effect(chosen==labels),"retrieval":cells}
    separate={}
    for decoder in ("rgb_centroid","names","glosses_filtered"):
        separate[decoder]=pop.effect(material[("drawing_centroid",decoder)]-material[("rgb_centroid",decoder)])
    record(OUT/f"visual-summary-{split}.pt",{"query_ids":qi,"source_ids":src,"classes":labels,"choices":choices,
           "metrics":{a:{b:material[(a,b)] for b in p[0]["target_KD"]} for a in choices},"semantic":baseline,
           "gallery_ids":p[0]["gallery_ids"]},revision,{"split":split,"queries":len(qi),"gallery":len(p[0]["gallery_ids"]),
           "subjects":len(src.unique()),"semantic_reference_percents":(100*baseline.mean(0)).tolist(),
           "summary":summaries,"drawing_minus_rgb_observer":separate})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",choices=SPLITS,required=True)
    p.add_argument("--fold",type=int,choices=(0,1))
    for k in ("revision","table-revision","native-revision","name-revision"):p.add_argument("--"+k,required=True)
    a=p.parse_args()
    deterministic()
    if a.fold is None:collect(a.split,a.revision,a.table_revision)
    else:run(a.split,a.fold,a.revision,a.table_revision,a.native_revision,a.name_revision)
