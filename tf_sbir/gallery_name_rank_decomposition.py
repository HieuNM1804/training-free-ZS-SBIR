"""Factor the existing neighbor-induced ranking change into noun slots and within-noun order."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json

import torch
from torch.nn.functional import normalize

from .branch_view_study import PrefixReplay
from .description_block_results import Population
from .description_choice_blocks import OUT as NATIVE, inputs
from .gallery_diffusion import nearest_gallery
from .kaggle_unseen_features import DEFAULT_CACHE
from .native_block_probe import SEED, deterministic
from .photo_name_evaluation import KD
from .photo_name_interaction import OUT as NAME_INDEX, PARTS, ordered_shards, record, verified, word_permutation
from .quickdraw_locked import KD_FUNCTION_SHA
from .replicated_simplex_study import evaluator
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_disputed_word_risk import query_posteriors, word_bank
from .semantic_donor_specificity import average_donors
from .wordnet_renaming_pilot import SPLITS


OUT = DEFAULT_CACHE.parent/"results/gallery-name-rank-decomposition"
REFERENCE = {"sketchy_1":((75.35,77.10),(80.15,79.94)), "sketchy_2":((72.66,71.07),(75.71,73.60))}


def semantic_inputs(split, name_revision, native_revision):
    _,idx,sources = inputs()
    qi,labels = idx["sketch"]["views"][split]
    gi,classes = idx["photo"]["views"][split]
    raw = ordered_shards([NAME_INDEX/f"names-{i}.pt" for i in range(PARTS)],name_revision,
                         torch.arange(len(idx["photo"]["paths"])))
    names = raw["name"][gi].cuda()
    photos = normalize(raw["native"][gi].cuda().float(),dim=1)
    qparts = []
    for i in range(PARTS):
        a = verified(NATIVE/f"native-sketch-{i}.pt",native_revision)
        qparts.append((a["ids"],a["rgb"]["native"],a["mirror"]["native"].half()))
    full_ids = torch.cat([z[0] for z in qparts])
    if not torch.equal(full_ids.sort().values,torch.arange(len(idx["sketch"]["paths"]))):
        raise ValueError("Unexpected sketch indexing")
    order = full_ids.argsort()[qi]
    q,mirror = (torch.cat([z[k] for z in qparts])[order].cuda().float() for k in (1,2))
    archive = torch.load(DEFAULT_CACHE/"text.pt",mmap=True,weights_only=True)
    scale = archive["scale"]
    src,dst = (archive["text"][key].cuda() for key in DEFAULT_PROMPTS)
    p = word_bank(photos,dst,scale).T.contiguous().double()
    p = (p/p.sum(1,keepdim=True)).float()
    q = torch.cat([query_posteriors(a,b,src,scale) for a,b in zip(q.split(384),mirror.split(384),strict=True)])
    candidates,_ = nearest_gallery(photos,256)
    b = average_donors(p,candidates[:,:64])
    return qi,labels,gi,classes,sources[qi],names,q,p,b


def joined(positions, ordered, names):
    # Stable sort groups occurrences by noun while preserving their order independently in each ranking.
    slots = names[positions].argsort(dim=1,stable=True)
    items = names[ordered].argsort(dim=1,stable=True)
    result = torch.empty_like(positions).scatter_(1,slots,ordered.gather(1,items))
    if not torch.equal(names[result],names[positions]):
        raise ValueError("A noun was assigned to the wrong slot")
    return result


def attribution(rows,kind):
    base,full=rows["raw"],rows["donors"]
    outer,inner=rows[kind+"_outer"],rows[kind+"_inner"]
    group=((outer-base)+(full-inner))/2
    within=((inner-base)+(full-outer))/2
    if not torch.allclose(group+within,full-base,atol=1e-12,rtol=0):
        raise ValueError("Attribution does not sum to the observed change")
    return group,within


@torch.inference_mode()
def run(split,part,revision,name_revision,native_revision):
    qi,labels,gi,classes,_,names,q,p,b=semantic_inputs(split,name_revision,native_revision)
    take=torch.arange(part,len(qi),2)
    keys=("raw","donors","named_outer","named_inner","random_outer","random_inner")
    rows={k:[] for k in keys}
    groups={"named":names,"random":names[word_permutation(split,len(names)).cuda()]}
    replay=PrefixReplay(evaluator(KD),classes,split)
    for start in range(0,len(take),192):
        ids=take[start:start+192]
        raw=(q[ids]@p.T.contiguous()).topk(len(gi),dim=1).indices
        donors=(q[ids]@b.T.contiguous()).topk(len(gi),dim=1).indices
        orders={"raw":raw,"donors":donors}
        for kind,group in groups.items():
            if not torch.equal(joined(raw,raw,group),raw) or not torch.equal(joined(donors,donors,group),donors):
                raise ValueError("Original ordering not recovered")
            orders[kind+"_outer"]=joined(donors,raw,group)
            orders[kind+"_inner"]=joined(raw,donors,group)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for k,mat in orders.items():
                rows[k].append(replay.score(mat,labels[ids],pool).double())
    record(OUT/f"{split}-{part}.pt",{"ids":qi[take],"gallery_ids":gi,"groups":names.cpu(),
           "rows":{k:torch.cat(v) for k,v in rows.items()}},revision,
           {"name_revision":name_revision,"native_revision":native_revision,"evaluator":KD_FUNCTION_SHA,
            "part":part,"split":split,"queries":len(take),"gallery":len(gi)})


def summarize(split,revision):
    _,idx,source=inputs()
    qi,labels=idx["sketch"]["views"][split]
    gi,_=idx["photo"]["views"][split]
    parts=[verified(OUT/f"{split}-{i}.pt",revision) for i in range(2)]
    rows={k:torch.empty((len(qi),2),dtype=torch.float64) for k in parts[0]["rows"]}
    for i,a in enumerate(parts):
        if not torch.equal(a["ids"],qi[i::2]) or not torch.equal(a["gallery_ids"],gi) or not torch.equal(a["groups"],parts[0]["groups"]):
            raise ValueError("Unequal evaluation population or group assignments")
        for k in rows:rows[k][i::2]=a["rows"][k]
    pop=Population(labels,source[qi])
    means={k:(100*v.mean(0)).tolist() for k,v in rows.items()}
    drift={k:[v-t for v,t in zip(means[k],r,strict=True)] for k,r in zip(("raw","donors"),REFERENCE[split],strict=True)}
    effects={k:attribution(rows,k) for k in ("named","random")}
    out={kind:{"between_group_slots":pop.effect(pair[0]),"within_group_order":pop.effect(pair[1])}
         for kind,pair in effects.items()}
    out["named_minus_random_within"]=pop.effect(effects["named"][1]-effects["random"][1])
    record(OUT/f"summary-{split}.pt",{"query_ids":qi,"gallery_ids":gi,"labels":labels,"source_ids":source[qi],
           "name_groups":parts[0]["groups"],"rows":rows},revision,{"queries":len(qi),"gallery":len(gi),
           "means":means,"endpoint_drift_percentage_points":drift,"endpoint_check":max(abs(v) for row in drift.values() for v in row)<.1,
           "effects":out,"total_change":pop.effect(rows["donors"]-rows["raw"]),"split":split})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",required=True,choices=SPLITS)
    p.add_argument("--revision",required=True)
    p.add_argument("--part",type=int,choices=(0,1))
    p.add_argument("--name-revision",required=True)
    p.add_argument("--native-revision",required=True)
    a=p.parse_args()
    deterministic()
    if a.part is None:summarize(a.split,a.revision)
    else:run(a.split,a.part,a.revision,a.name_revision,a.native_revision)
