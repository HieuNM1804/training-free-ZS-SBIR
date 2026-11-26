"""Test whether the same completion law can work entirely on the query side."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import random

import numpy as np
import open_clip
import torch

from .concept_benchmark import atomic_json, checksum
from .conditional_completion import random_frames
from .conditional_completion_study import (
    BATCH, DEFAULT_PROMPTS, DOCS, FAMILY, NEIGHBORS, SEED, SKETCHY, Split, VOCABULARY_STUDY,
    class_interval, context, encode_prompt, evaluate, image_inputs, mean_percent, select,
)
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT, runtime
from .encoding import load_frozen_clip, save_tensor
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .source_completion import paired_axes, transfer
from .semantic_boxable_prompt_search import posteriors


RANKS = (32, 64, 128, 256)


def sweep(root, split, text, kd, axis, protocol):
    target = root / f"grid-{split.name}.json"
    if target.exists():
        record = json.loads(target.read_bytes())
        if record["protocol"] != protocol:
            raise ValueError("Protocol changed on resume")
        return record["metrics"]
    _, native, photos, t, replay = context(split, text, kd)
    table = {}
    for rank in RANKS:
        controls = ("completion", "matched_donors", "full_donors") if rank == RANKS[0] else ("completion", "matched_donors")
        banks = transfer((split.sketch, split.mirror), photos, t, split.scale, axis, rank, NEIGHBORS, controls)
        for arm, configs in banks.items():
            for k, query in configs.items():
                name = f"{arm}-{k}" if arm == "full_donors" else f"{arm}-{rank}-{k}"
                table[name] = mean_percent(evaluate(split, query, native, replay))
        print(split.name, "source-only", rank, {k: [round(v,3) for v in table[f"completion-{rank}-{k}"]]
                                               for k in NEIGHBORS}, flush=True)
        del banks
    atomic_json(target, {"protocol": protocol, "queries": len(split.query_ids), "gallery": len(photos), "metrics": table})
    return table


def official(root, split, text, kd, pool, axes, lock, selection, grid=None):
    target = root / f"{split.name}.json"
    if target.exists():
        saved = json.loads(target.read_bytes())
        if saved["selection"] != selection or checksum(target.with_suffix(".pt")) != saved["per_query_sha256"]:
            raise ValueError("Invalid source-side result")
        return
    q, native, photos, t, replay = context(split, text, kd)
    source = (split.sketch, split.mirror)
    r, k = lock["rank"], lock["k"]
    rows = {}
    def record(name, query, grid_key=None):
        rows[name] = evaluate(split, query, native, replay, pool)
        values = mean_percent(rows[name])
        if grid is not None and grid_key is not None:
            if max(abs(a-b) for a,b in zip(values, grid[grid_key], strict=True)) > .00025:
                raise ValueError(f"Original KD replay differs for {name}")
        print(split.name, name, values, flush=True)

    record("dot", q)
    p = DEFAULT_PROMPTS[1]
    alternate, _ = posteriors(split, text, [(p,p)])
    record("target_prompt_only", alternate[p])
    head = transfer(source, photos, t, split.scale, axes["paired"], r, [k],
                    ("completion", "matched_donors", "full_donors", "mean_logits"))
    for arm, banks in head.items():
        key = f"{arm}-{k}" if arm=="full_donors" else f"{arm}-{r}-{k}"
        record(arm, banks[k], None if arm=="mean_logits" else key)
    del head
    for name, arm in (("best_matched", "matched_donors"), ("best_full", "full_donors")):
        config = lock[arm].split("-")
        cr, ck = (int(config[1]), int(config[2])) if len(config)==3 else (r,int(config[1]))
        if (cr,ck)==(r,k):
            rows[name] = rows[arm]
        else:
            values = transfer(source, photos, t, split.scale, axes["paired"], cr, [ck], (arm,))
            record(name, values[arm][ck], lock[arm])
    for name in ("random", "midpoint", "shuffled"):
        values = transfer(source, photos, t, split.scale, axes[name], r, [k], ("completion",))
        record(name, values["completion"][k])
    artifact = target.with_suffix(".pt")
    save_tensor({"qi":split.query_ids,"pi":split.gallery_ids,"labels":split.sketch_labels,"rows":rows}, artifact)
    atomic_json(target, {"commit":revision(),"selection":selection,"cutoffs":replay.cutoffs,"queries":len(q),
        "gallery":len(photos),"classes":len(split.sketch_labels.unique()),"device":torch.cuda.get_device_name(),
        "metrics":{name:mean_percent(values) for name,values in rows.items()},
        "completion_minus_control_class_macro_ap":{name:class_interval(100*(rows["completion"][:,0]-values[:,0]),
                                                                    split.sketch_labels,seed=SEED)
                                                  for name,values in rows.items()},
        "per_query_sha256":checksum(artifact)})


@torch.inference_mode()
def run():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    workspace = Path("/content/sbir")
    root = workspace / "results" / FAMILY / "source"
    vocab = json.loads((DOCS / VOCABULARY_STUDY / "freeze.json").read_bytes())
    sha = vocab.pop("sha256")
    words = vocab["subsets"]["boxable"]["words"]
    if len(words)!=601 or digest(vocab)!=sha:
        raise ValueError("Invalid vocabulary")
    kd, _ = load_kd(workspace)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Evaluator changed")
    workers = min(4,len(os.sched_getaffinity(0)))
    protocol = save_lock(root / "protocol.json", {"commit":revision(),"seed":SEED,"ranks":RANKS,"k":NEIGHBORS,
        "prompts":DEFAULT_PROMPTS,"vocabulary":sha,"clip":CHECKPOINT,"evaluator":KD_FUNCTION_SHA,
        "selection":"Full unseen Sketchy mAP mean, precision tie-break. Optimize both donor controls independently.",
        "hypothesis":"Complete a partial sketch in target space and keep every gallery feature native. The same-donor control removes only the completion."})
    cfg = runtime(workspace,"sketchy",batch=BATCH,workers=workers)
    model = load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = {p:encode_prompt(model,tokenizer,words,p) for p in DEFAULT_PROMPTS}
    s,t = (text[p] for p in DEFAULT_PROMPTS)
    axes = {name:paired_axes(s,t,name,seed=SEED)[0] for name in ("paired","midpoint","shuffled")}
    axes["random"] = random_frames(t.size(1),1,SEED,t.device)[0]
    data = image_inputs(cfg,SKETCHY,model)
    grids = [sweep(root,Split.from_inputs(data,name),text,kd,axes["paired"],protocol) for name in SKETCHY]
    winner = select(grids,"completion")
    _,r,k = winner.split("-")
    lock = {"commit":revision(),"protocol":protocol,"rank":int(r),"k":int(k),"winner":winner,
            "grids":{name:checksum(root/f"grid-{name}.json") for name in SKETCHY},
            "matched_donors":select(grids,"matched_donors"),"full_donors":select(grids,"full_donors"),
            "sketchy":[g[winner] for g in grids]}
    selection=save_lock(root/"freeze.json",lock)
    print("SOURCE FROZEN before transfer",lock,flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name,grid in zip(SKETCHY,grids,strict=True):
            official(root,Split.from_inputs(data,name),text,kd,pool,axes,lock,selection,grid)
        del data
        cfg=runtime(workspace,"tuberlin",batch=BATCH,workers=workers)
        split=Split.from_inputs(image_inputs(cfg,("tuberlin",),model),"tuberlin")
        official(root,split,text,kd,pool,axes,lock,selection)


if __name__=="__main__":
    run()
