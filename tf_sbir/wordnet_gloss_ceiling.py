"""Measure the best possible RGB retrieval using one of the pilot's available descriptions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json

import open_clip
import torch
from torch.nn.functional import normalize

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .data import UNSEEN_CLASSES
from .encoding import load_frozen_clip, save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE, load_split
from .native_block_probe import SEED, deterministic
from .replicated_simplex_study import evaluator
from .semantic_boxable import DEFAULT_PROMPTS
from .wordnet_renaming_pilot import OUT, SPLITS
from .wordnet_renaming_text import anonymize, describe, encode, inventory


def all_targets(senses,pairs):
    original = [describe("photo",s.anonymous) for s in senses]
    sentences = list(dict.fromkeys(original))
    index = {s:i for i,s in enumerate(sentences)}
    unique,inverse = pairs.unique(dim=0,return_inverse=True)
    mapping = []
    for ai,bi in unique.tolist():
        a,b = senses[ai],senses[bi]
        aliases = tuple(sorted(set(a.aliases)|set(b.aliases)))
        ga,gb = anonymize(a.definition,aliases),anonymize(b.definition,aliases)
        if not ga or not gb:
            ga,gb = a.anonymous,b.anonymous
        row = []
        for gloss in (ga,gb):
            sentence = describe("photo",gloss)
            if sentence not in index:
                index[sentence] = len(sentences)
                sentences.append(sentence)
            row.append(index[sentence])
        mapping.append(row)
    return sentences,len(set(original)),torch.tensor(mapping)[inverse]


def per_class_metrics(vectors,photos,split):
    kd = evaluator(DEFAULT_CACHE.parent/"real-sketchy-smoke/evaluator.py")
    replay = PrefixReplay(kd,split.photo_labels,split.name)
    classes = split.sketch_labels.unique().sort().values
    if not torch.equal(classes,torch.arange(len(classes),dtype=classes.dtype)):
        raise ValueError("Unexpected class-index order")
    rows = []
    bank = photos.T.contiguous()
    cut = replay.cutoffs[0] or len(photos)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for batch in vectors.split(192):
            rank = (batch @ bank).topk(cut).indices.to("cpu",torch.int32)
            measured = [replay.score(rank,c.expand(len(batch)),pool).double() for c in classes]
            rows.append(torch.stack(measured,dim=1))
    return torch.cat(rows)


@torch.inference_mode()
def run(split_name, revision, pilot_revision, half):
    split,text,words = load_split(DEFAULT_CACHE,split_name)
    pieces = []
    for shard in (0,1):
        p = OUT/f"{split_name}-{shard}.pt"
        m = json.loads(p.with_suffix(".json").read_bytes())
        if m["revision"] != pilot_revision or checksum(p) != m["tensor_sha256"]:
            raise ValueError("Unverified pilot shard")
        pieces.append(torch.load(p,mmap=True,weights_only=True))
    order = torch.cat([p["positions"] for p in pieces]).argsort()
    pair = torch.cat([p["pair"] for p in pieces])[order]
    senses,_ = inventory(words)
    sentences,n_base,pair_text = all_targets(senses,pair)
    model = load_frozen_clip(json.loads((DEFAULT_CACHE/"manifest.json").read_bytes())["checkpoint"],torch.device("cuda"))
    strings = [DEFAULT_PROMPTS[1].format(w) for w in words]+sentences
    ids = torch.arange(half,len(strings),2)
    n_words = len(words)
    selected = ids[ids>=n_words]-n_words
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    # Use the original cached embeddings for the public names and the pilot's target templates for definitions.
    vectors = torch.cat((text[DEFAULT_PROMPTS[1]][ids[ids<n_words]],encode(model,tokenizer,[sentences[j] for j in selected.tolist()])))
    table = per_class_metrics(vectors,normalize(split.photo.float(),dim=1),split)
    dest = OUT/"ceiling"/f"{split_name}-{half}.pt"
    save_tensor({"indices":ids,"classes":split.sketch_labels,"values":table,"query_pair_text":pair_text+n_words,
                 "words":n_words,"senses":len(senses),"base":n_base,"strings":strings},dest)
    atomic_json(dest.with_suffix(".json"),{"split":split_name,"half":half,"pilot":pilot_revision,
                "revision":revision,"sha256":checksum(dest),"seed":SEED,"device":torch.cuda.get_device_name()})


def summarize(name,revision,pilot_revision):
    parts = []
    for half in (0,1):
        path = OUT/"ceiling"/f"{name}-{half}.pt"
        m = json.loads(path.with_suffix(".json").read_bytes())
        if m["revision"] != revision or m["pilot"] != pilot_revision or checksum(path) != m["sha256"]:
            raise ValueError("Invalid ceiling run")
        parts.append(torch.load(path,weights_only=True,mmap=True))
    a,b = parts
    if a["strings"]!=b["strings"] or not torch.equal(a["query_pair_text"],b["query_pair_text"]) or not torch.equal(a["classes"],b["classes"]):
        raise ValueError("Mismatched vocabulary or query population")
    ids = torch.cat([p["indices"] for p in parts])
    if not torch.equal(ids.sort().values,torch.arange(len(a["strings"]))):
        raise ValueError("Sentence coverage is incomplete")
    table = torch.cat([p["values"] for p in parts])[ids.argsort()]
    counts = a["classes"].bincount().double()
    classes = a["classes"]
    n,n_base = a["words"],a["base"]
    means,winners = {},{}
    for key,start,stop in (("all_names",0,n),("canonical_glosses",n,n+n_base),("all_observed_glosses",n,len(ids))):
        val,idx = table[start:stop].max(0)
        means[key] = (100*(val*counts[:,None]).sum(0)/counts.sum()).tolist()
        winners[key] = {label:{"AP_sentence":a["strings"][start+idx[i,0].item()],"AP_P_ceiling":(100*val[i]).tolist()}
                        for i,label in enumerate(UNSEEN_CLASSES[name])}
    read = table[a["query_pair_text"],classes[:,None]]
    pilot = [torch.load(OUT/f"{name}-{s}.pt",mmap=True,weights_only=True) for s in (0,1)]
    order = torch.cat([p["positions"] for p in pilot]).argsort()
    prior = torch.cat([p["candidate_metrics"] for p in pilot])[order]
    error = (read-prior).abs()
    if error.mean() > .0005:
        raise ValueError("Metrics for the actual pilot candidates were not reproduced")
    upper = table[n:].amax(0)[classes]
    per_query = read.max(1).values
    if (per_query > upper+1e-8).any():
        raise ValueError("Attainable conditional ceiling is larger than the bank")
    path = OUT/"ceiling"/f"summary-{name}.json"
    atomic_json(path,{"split":name,"revision":revision,"pilot_revision":pilot_revision,"seed":SEED,
                "queries":len(classes),"gallery":len(pilot[0]["gallery_ids"]),"n_words":n,"n_senses":a["senses"],
                "different_canonical_sentences":n_base,"different_observed_glosses":len(ids)-n,
                "bank_ceiling_AP_P":means,"query_selected_two_oracle_replay":(100*per_query.mean(0)).tolist(),
                "replay_mean_absolute_error_points":(100*error.mean((0,1))).tolist(),
                "replay_max_absolute_error_points":(100*error.amax((0,1))).tolist(),
                "winning_descriptions_by_class":winners})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",required=True,choices=SPLITS)
    parser.add_argument("--half",type=int,choices=(0,1))
    parser.add_argument("--revision",required=True)
    parser.add_argument("--pilot-revision",required=True)
    args=parser.parse_args()
    deterministic()
    if args.half is None:
        summarize(args.split,args.revision,args.pilot_revision)
    else:
        run(args.split,args.revision,args.pilot_revision,args.half)
