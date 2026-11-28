"""Evaluate the approved gallery index with the unmodified KD evaluator on actual embeddings."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time

import torch
from torch.nn.functional import normalize

from .concept_benchmark import atomic_json, checksum
from .description_block_choices import order_key
from .description_block_results import Population
from .description_choice_blocks import PARTS as NATIVE_PARTS, OUT as NATIVE, inputs
from .description_choice_head_ceiling import baseline
from .kaggle_unseen_features import DEFAULT_CACHE
from .native_block_probe import SEED, deterministic
from .photo_name_interaction import OUT, ordered_shards, record, verified
from .quickdraw_locked import KD_FUNCTION_SHA
from .replicated_simplex_study import evaluator
from .semantic_boxable import DEFAULT_PROMPTS
from .wordnet_renaming_pilot import SPLITS


ARMS = ("interaction","native","canvas","joint","word_only","text_name","wrong_interaction")
KD = DEFAULT_CACHE.parent / "real-sketchy-smoke" / "evaluator.py"


def queries(qi, revision):
    results = []
    for i in range(NATIVE_PARTS):
        path = NATIVE/f"native-sketch-{i}.pt"
        shard = verified(path,revision)
        results.append((shard["ids"],shard["rgb"]["native"]))
    ids = torch.cat([v[0] for v in results])
    order = ids.argsort()
    if not torch.equal(ids[order],torch.arange(len(ids))):
        raise ValueError("Invalid original sketch order")
    return torch.cat([v[1] for v in results])[order[qi]].cuda()


def gallery(split, ids, revision):
    sorted_ids = ids.sort().values
    rows = ordered_shards([OUT/f"gallery-{split}-{i}.pt" for i in range(2)],revision,sorted_ids)
    order = torch.searchsorted(sorted_ids,ids)
    if not torch.equal(sorted_ids[order],ids):
        raise ValueError("Incorrect official gallery ordering")
    rows = {k:v[order].cuda() for k,v in rows.items()}
    if not torch.equal(rows["name"].sort().values,rows["wrong_name"].sort().values):
        raise ValueError("Wrong-name control changed noun frequencies")
    text = torch.load(DEFAULT_CACHE/"text.pt",mmap=True,weights_only=True)
    rows["text_name"] = text["text"][DEFAULT_PROMPTS[1]].cuda()[rows["name"]]
    return rows


def matched_same_word(split, q, qi, labels, gi, gc, source, features):
    # Positive selection precedes reading any of the competing rankings.
    eligible_photos = {int(k):torch.where(gc==k)[0] for k in gc.unique()}
    pos = []
    for i,cls,exact in zip(qi.tolist(),labels.tolist(),source.tolist(),strict=True):
        allowed = eligible_photos[cls]
        allowed = allowed[gi[allowed]!=exact]
        word = int.from_bytes(order_key("same-printed-name-positive",split,i)[:8],"little")
        pos.append(int(allowed[word%len(allowed)]))
    pos = torch.tensor(pos,device="cuda")
    native = normalize(features["native"],dim=1)
    names = features["name"]
    classes = gc.cuda()
    controls = {}
    details = {k:[] for k in ("pos","lower","upper","keep","cos_gap_lower","cos_gap_upper")}
    norms = {k:normalize(features[k],dim=1) for k in ARMS}
    for start in range(0,len(qi),256):
        sl = slice(start,start+256)
        hit = pos[sl]
        x = normalize(q[sl],dim=1)
        cosine = x@native.T
        margin = cosine.gather(1,hit[:,None])-cosine
        other = (names[None,:]==names[hit,None]) & (classes[None,:]!=labels[sl,None].cuda())
        below = margin.masked_fill(~(other & (margin>0)),float("inf"))
        above = (-margin).masked_fill(~(other & (margin<0)),float("inf"))
        lowgap,low = below.min(1)
        highgap,high = above.min(1)
        keep = lowgap.isfinite() & highgap.isfinite()
        for k,v in (("pos",hit),("lower",low),("upper",high),("keep",keep),
                    ("cos_gap_lower",lowgap),("cos_gap_upper",highgap)):
            details[k].append(v.cpu())
        for key,bank in norms.items():
            if key=="native":
                # Read the identical matrix that established the strict brackets, including its rounding.
                deltas=torch.stack((lowgap,-highgap),1)
            else:
                deltas=torch.stack(((x*(bank[hit]-bank[low])).sum(1),(x*(bank[hit]-bank[high])).sum(1)),dim=1)
            controls.setdefault(key,[]).append(deltas.cpu())
    return {k:torch.cat(v) for k,v in details.items()}, {k:torch.cat(v) for k,v in controls.items()}


@torch.inference_mode()
def run(split,part,revision,gallery_revision,native_revision):
    _,index,sources = inputs()
    qi,labels = index["sketch"]["views"][split]
    gi,classes = index["photo"]["views"][split]
    indices = torch.arange(part,len(qi),2)
    q = queries(qi[indices],native_revision)
    features = gallery(split,gi,gallery_revision)
    kd = evaluator(KD)
    required = (200,200) if split=="sketchy_2" else (0,100)
    truth = labels[indices]
    for key in ARMS:
        out = OUT/f"score-{split}-{part}-{key}.pt"
        if out.with_suffix(".json").is_file():
            old = verified(out,revision)
            if not torch.equal(old["ids"],qi[indices]):raise ValueError("Stale evaluation queries")
            continue
        bank = features[key]
        def one(i):
            with torch.inference_mode():
                ap,precision,m,p = kd(q[i:i+1],bank,truth[i:i+1],classes,split)
            if (m,p)!=required:raise ValueError("KD changed the cutoffs")
            return torch.stack((ap,precision))
        begin = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as pool:
            measurements = torch.stack(list(pool.map(one,range(len(q)))))
        record(out,{"ids":qi[indices],"metrics":measurements},revision,
               {"arm":key,"split":split,"part":part,"evaluator":KD_FUNCTION_SHA,
                "gallery_revision":gallery_revision,"native_revision":native_revision,
                "queries":len(q),"gallery":len(gi),"seconds":time.monotonic()-begin})
        print(split,part,key,(100*measurements.double().mean(0)).tolist(),flush=True)
    panel,deltas = matched_same_word(split,q,qi[indices],truth,gi,classes,sources[qi[indices]],features)
    record(OUT/f"same-word-{split}-{part}.pt",{"ids":qi[indices],"gallery_ids":gi,"panel":panel,"deltas":deltas},
           revision,{"queries":len(q),"eligible":int(panel["keep"].sum()),"part":part,"split":split})


def summarize(split,revision,gallery_revision):
    _,index,sources = inputs()
    qi,labels = index["sketch"]["views"][split]
    gi,_ = index["photo"]["views"][split]
    pop = Population(labels,sources[qi])
    scores = {"semantic":baseline(split,qi)}
    hashes = {}
    for key in ARMS:
        parts = [OUT/f"score-{split}-{i}-{key}.pt" for i in range(2)]
        p = [verified(f,revision) for f in parts]
        result = torch.empty((len(qi),2),dtype=torch.float64)
        for i,(shard,f) in enumerate(zip(p,parts,strict=True)):
            if not torch.equal(shard["ids"],qi[i::2]):raise ValueError("Query order changed")
            result[i::2]=shard["metrics"].double()
            hashes[f.name]=checksum(f)
        scores[key]=result
    values={k:(100*v.mean(0)).tolist() for k,v in scores.items()}
    effects={k:pop.effect(scores["interaction"]-v) for k,v in scores.items() if k!="interaction"}
    panels=[verified(OUT/f"same-word-{split}-{i}.pt",revision) for i in range(2)]
    packed_ids = torch.cat([p["ids"] for p in panels])
    positions = packed_ids.argsort()[torch.searchsorted(packed_ids.sort().values,qi)]
    if not torch.equal(packed_ids[positions],qi):raise ValueError("Bad panel ordering")
    keep=torch.cat([p["panel"]["keep"] for p in panels])[positions]
    conditional={"eligible":int(keep.sum()),"total":len(qi)}
    if keep.any():
        sub=Population(labels[keep],sources[qi][keep])
        conditional["subjects"]=len(sub.counts)
        conditional["classes"]=len(sub.classes)
        tests={}
        for key in ARMS:
            margin=torch.cat([p["deltas"][key] for p in panels])[positions][keep]
            wins=(margin>0).double()+.5*(margin==0)
            if key in ("native","word_only","text_name") and not (wins.mean(1)==.5).all():
                raise ValueError(f"Broken matched control: {key}")
            tests[key]=wins
        conditional["results"]={k:{"each_side":sub.effect(v),"average":sub.effect(v.mean(1)),
                                    "primary_minus_comparator":sub.effect((tests["interaction"]-v).mean(1))}
                                for k,v in tests.items()}
    features=gallery(split,gi,gallery_revision)
    g=features["interaction"]
    qs=torch.tensor([0,.1,.5,.9,1.],device="cuda")
    diagnostics={"norm_quantiles":torch.quantile(g.norm(dim=1),qs).tolist(),
                 "word_cosine_quantiles":torch.quantile((normalize(g,dim=1)*normalize(features["word_only"],dim=1)).sum(1),qs).tolist(),
                 "zero_rows":int((g.norm(dim=1)==0).sum()),"unmodified_permutation_percent":
                 100*(features["name"]==features["wrong_name"]).double().mean().item(),
                 "distinct_nominated_names":len(features["name"].unique())}
    dest=OUT/f"official-{split}.pt"
    record(dest,{"query_ids":qi,"gallery_ids":gi,"classes":labels,"source_ids":sources[qi],"scores":scores,
                 "same_word_parts":panels,"photo_names":features["name"].cpu(),"shuffled_names":features["wrong_name"].cpu()},
           revision,{"split":split,"queries":len(qi),"gallery":len(gi),"gallery_revision":gallery_revision,
                     "official_percent":values,"paired_primary_minus_comparator":effects,"same_word":conditional,
                     "index_diagnostics":diagnostics,"evaluator":KD_FUNCTION_SHA,"input_score_hashes":hashes})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",choices=SPLITS,required=True)
    p.add_argument("--part",type=int,choices=(0,1))
    p.add_argument("--revision",required=True)
    p.add_argument("--gallery-revision",required=True)
    p.add_argument("--native-revision")
    a=p.parse_args()
    deterministic()
    with torch.inference_mode():
        if a.part is None:summarize(a.split,a.revision,a.gallery_revision)
        else:run(a.split,a.part,a.revision,a.gallery_revision,a.native_revision)
