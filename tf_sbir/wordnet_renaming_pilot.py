"""Evaluate the approved two-description test and matched controls on frozen Sketchy CLIP inputs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json

import open_clip
import torch
from torch.nn.functional import normalize

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .conditional_completion_study import evaluate
from .crossdomain_inputs import CHECKPOINT
from .encoding import load_frozen_clip, save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE, load_split
from .named_word_axis_audit import OUT as AUDIT, logits, text_readouts
from .native_block_probe import SEED, deterministic
from .replicated_simplex_study import evaluator
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_donor_specificity import boot
from .wordnet_renaming_text import WORDNET_SHA256, describe, encode, inventory, pair_language


SPLITS = ("sketchy_1","sketchy_2")
ARMS = ("name","definition","attached","certified")
OUT = DEFAULT_CACHE.parent/"results"/"wordnet-description-renaming"


def nominate(x, meanings, heads):
    pairs = []
    for batch in x.split(384):
        similarity = batch @ meanings.T
        first = similarity.argmax(1)
        other = similarity.masked_fill(heads[None, :] == heads[first,None], -torch.inf).argmax(1)
        pairs.append(torch.stack((first,other),dim=1))
    result = torch.cat(pairs)
    if (heads[result[:,0]] == heads[result[:,1]]).any():
        raise ValueError("Hypotheses should have different public names")
    return result


def named_margins(x, bank, inverse, pair_heads, text, usable):
    # Two matched readings change just which name accompanies each description.
    vectors = bank[inverse]
    scores = torch.einsum("nd,nrd->nr",x,vectors[:,:6])
    coupled = scores[:,0] - scores[:,1]
    swapped = scores[:,2] - scores[:,3]
    definition = scores[:,4] - scores[:,5]
    name = (x*(text[pair_heads[:,0]]-text[pair_heads[:,1]])).sum(1)
    active = usable[inverse] & (((coupled > 0) & (swapped > 0)) | ((coupled < 0) & (swapped < 0)))
    choices = {"name":name < 0, "definition":definition < 0, "attached":coupled < 0,
               "certified":torch.where(active,coupled < 0,definition < 0)}
    additive = usable[inverse] & (definition.abs() > name.abs())
    margins = torch.stack((name,definition,coupled,swapped),1)
    return choices, active, additive, margins


def semantic_opponents(split, qs, ps):
    ids,same = [], []
    gallery_labels = split.photo_labels.cuda()
    gallery_words = ps.argmax(1)
    bank = ps.T.contiguous()
    for start in range(0,len(qs),384):
        read = qs[start:start+384]
        truth = split.sketch_labels[start:start+len(read)].cuda()
        scores = read @ bank
        relevant = gallery_labels[None,:] == truth[:,None]
        pos = scores.masked_fill(~relevant,-torch.inf).argmax(1)
        neg = scores.masked_fill(relevant,-torch.inf).argmax(1)
        first = scores.argmax(1)
        originally_right = gallery_labels[first] == truth
        noun = read.argmax(1)
        same.append(torch.stack((originally_right, (~originally_right) & (gallery_words[neg]==noun)
                                  & (gallery_words[pos]!=noun)),1))
        ids.append(torch.stack((pos,neg),1))
    return torch.cat(ids),torch.cat(same)


@torch.inference_mode()
def run(name, shard, revision):
    split,text,words = load_split(DEFAULT_CACHE,name)
    inputs = json.loads((DEFAULT_CACHE/"manifest.json").read_bytes())
    if inputs["sha256"] != CHECKPOINT:
        raise ValueError("Invalid pretrained weights")
    old_file = AUDIT/f"{name}-core.pt"
    old_meta = json.loads(old_file.with_suffix(".json").read_bytes())
    if old_meta["tensor_sha256"] != checksum(old_file):
        raise ValueError("Unverified Semantic Dot reference")
    old = torch.load(old_file,mmap=True,weights_only=True)
    if not torch.equal(old["query_ids"],split.query_ids) or not torch.equal(old["gallery_ids"],split.gallery_ids):
        raise ValueError("Benchmark order changed")
    local_ids = torch.arange(shard,len(split.sketch),2)
    all_labels = split.sketch_labels
    original_ids = split.query_ids
    split = replace(split, sketch=split.sketch[local_ids], mirror=split.mirror[local_ids],
                    query_ids=original_ids[local_ids],sketch_labels=all_labels[local_ids])
    photos = normalize(split.photo.float(),dim=1)
    x = normalize(normalize(split.sketch.float(),dim=1)+normalize(split.mirror.float(),dim=1),dim=1)
    source,target = (text[k] for k in DEFAULT_PROMPTS)
    model = load_frozen_clip(inputs["checkpoint"],torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    senses,catalog = inventory(words)
    signature = hashlib.sha256(json.dumps(catalog,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if name==SPLITS[0] and shard==0:
        atomic_json(OUT/"inventory.json",catalog)
    print("senses",len(senses),"names",catalog["usable_public_names"],flush=True)
    heads = torch.tensor([s.head for s in senses],device="cuda")
    anonymous = encode(model,tokenizer,[describe("drawing",s.anonymous) for s in senses])
    pair = nominate(x,anonymous,heads)
    unique,inverse = pair.unique(dim=0,return_inverse=True)
    print("query pairs",len(pair),"different ordered pairs",len(unique),flush=True)
    bank,usable = pair_language(model,tokenizer,unique.tolist(),senses,words)
    choices,active,additive,margins = named_margins(x,bank,inverse,heads[pair],source,usable)
    target_a,target_b = bank[inverse,6],bank[inverse,7]
    all_name = (x @ source.T).argmax(1)
    del model
    zs,zp = logits(split,photos,source,target)
    _,_,semantic_q,semantic_p = text_readouts(zs,zp)
    opponent,baseline_status = semantic_opponents(split,semantic_q,semantic_p)
    diff = photos[opponent[:,0]]-photos[opponent[:,1]]
    candidate_delta = torch.stack(((target_a*diff).sum(1),(target_b*diff).sum(1)),dim=1).cpu()
    kd_path = DEFAULT_CACHE.parent/"real-sketchy-smoke/evaluator.py"
    replay = PrefixReplay(evaluator(kd_path),split.photo_labels,name)
    with ThreadPoolExecutor(max_workers=4) as pool:
        cand_a = evaluate(split,target_a.contiguous(),photos,replay,pool).double()
        cand_b = evaluate(split,target_b.contiguous(),photos,replay,pool).double()
        direct = evaluate(split,target[all_name].contiguous(),photos,replay,pool).double()
    rows = {"semantic":old["rows"]["NN"][local_ids],"direct_name_to_rgb":direct}
    for arm,selected_b in choices.items():
        rows[arm] = torch.where(selected_b.cpu()[:,None],cand_b,cand_a)
    if not torch.equal(rows["certified"][~active.cpu()],rows["definition"][~active.cpu()]):
        raise ValueError("Uncertified queries must retain their anonymous definition")
    dest = OUT/f"{name}-{shard}.pt"
    save_tensor({"positions":local_ids,"query_ids":split.query_ids,"gallery_ids":split.gallery_ids,
                 "classes":split.sketch_labels,"pair":pair.cpu(),"public_heads":heads[pair].cpu(),
                 "choices":{k:v.cpu() for k,v in choices.items()},"active":active.cpu(),"additive":additive.cpu(),
                 "usable":usable[inverse].cpu(),"margins":margins.cpu(),"candidate_metrics":torch.stack((cand_a,cand_b),1),
                 "rows":rows,"opponent":opponent.cpu(),"original_status":baseline_status.cpu(),
                 "candidate_deltas":candidate_delta,"direct_name":all_name.cpu()}, dest)
    atomic_json(dest.with_suffix(".json"), {"revision":revision,"seed":SEED,"split":name,"shard":shard,
                "checkpoint":CHECKPOINT,"wordnet":WORDNET_SHA256,"inventory_signature":signature,"senses":len(senses),
                "usable_public_names":catalog["usable_public_names"],"queries":len(x),"gallery":len(photos),
                "candidate_pairs":len(unique),"evaluator_sha256":checksum(kd_path),"text_sha256":checksum(DEFAULT_CACHE/"text.pt"),
                "device":torch.cuda.get_device_name(),"tensor_sha256":checksum(dest)})
    print(name,shard,"done",flush=True)


def summarize(name, revision):
    metas,parts = [],[]
    for shard in (0,1):
        path = OUT/f"{name}-{shard}.pt"
        meta = json.loads(path.with_suffix(".json").read_bytes())
        if meta["revision"] != revision or checksum(path) != meta["tensor_sha256"]:
            raise ValueError("Unverified shard")
        metas.append(meta)
        parts.append(torch.load(path,weights_only=True,mmap=True))
    for key in ("inventory_signature","text_sha256","evaluator_sha256","checkpoint","wordnet"):
        if metas[0][key] != metas[1][key]:
            raise ValueError("Different inference inputs across workers")
    pos = torch.cat([p["positions"] for p in parts])
    order = pos.argsort()
    if not torch.equal(pos[order],torch.arange(len(pos))) or not torch.equal(parts[0]["gallery_ids"],parts[1]["gallery_ids"]):
        raise ValueError("Unequal gallery or missing queries")
    def merged(key,sub=None):
        values = [p[key] if sub is None else p[key][sub] for p in parts]
        return torch.cat(values)[order]
    rows = {k:merged("rows",k) for k in parts[0]["rows"]}
    choices = {k:merged("choices",k) for k in ARMS}
    classes,active,additive = merged("classes"),merged("active"),merged("additive")
    candidates = merged("candidate_metrics")
    original,hard = merged("original_status").T
    correct = {}
    deltas = merged("candidate_deltas")
    for k,select in choices.items():
        d = deltas.gather(1,select.long()[:,None]).flatten()
        correct[k] = (d>0).double()+.5*(d==0).double()
    def estimates(delta,mask):
        cls = classes[mask].unique()
        if not len(cls):
            return None
        means = torch.stack([delta[mask & (classes==c)].mean(0) for c in cls])
        return {"mean_query_points":(100*delta[mask].mean(0)).tolist(),"class_balanced":boot((100*means).numpy())}
    groups = {"all":torch.ones_like(active),"certified":active,"target_errors":hard,"original_correct":original}
    panel = {}
    for group,mask in groups.items():
        arms = {}
        for k in ARMS:
            v = correct[k]
            arms[k] = {"pair_accuracy":(100*v[mask].mean()).item() if mask.any() else None,
                       "repairs":int((mask & ~original & (v==1)).sum()),
                       "damage":int((mask & original & (v==0)).sum()),"ties":int((mask & (v==.5)).sum())}
        panel[group] = {"queries":int(mask.sum()),"arms":arms,
                        "certified_minus_definition":estimates(correct["certified"]-correct["definition"],mask)}
    summary = {"split":name,"revision":revision,"seed":SEED,"queries":len(pos),"gallery":len(parts[0]["gallery_ids"]),
               "classes":len(classes.unique()),"senses":metas[0]["senses"],"public_names":metas[0]["usable_public_names"],
               "inventory_signature":metas[0]["inventory_signature"],"wordnet_sha256":WORDNET_SHA256,
               "raw_metrics":{k:(100*v.mean(0)).tolist() for k,v in rows.items()},
               "best_of_two_by_metric_oracle":(100*candidates.amax(1).mean(0)).tolist(),
               "certified_queries":int(active.sum()),"usable_queries":int(merged("usable").sum()),
               "certified_vs_definition_changed":int((choices["certified"]!=choices["definition"]).sum()),
               "additive_certified":int(additive.sum()),"certified_under_both":int((active & additive).sum()),
               "query_win_loss_tie_AP_vs_definition":[int(((rows["certified"][:,0]-rows["definition"][:,0])*sign>0).sum())
                                                   for sign in (1,-1)]+[int((rows["certified"][:,0]==rows["definition"][:,0]).sum())],
               "official_effects":{k:estimates(rows["certified"]-rows[k],groups["all"]) for k in ("name","definition","attached","semantic")},
               "official_metrics_on_certified":{k:(100*v[active].mean(0)).tolist() for k,v in rows.items()} if active.any() else {},
               "matched_pairs":panel}
    atomic_json(OUT/f"summary-{name}.json",summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",required=True,choices=SPLITS)
    parser.add_argument("--revision",required=True)
    parser.add_argument("--shard",type=int,choices=(0,1))
    args = parser.parse_args()
    deterministic()
    if args.shard is None:
        summarize(args.split,args.revision)
    else:
        run(args.split,args.shard,args.revision)
