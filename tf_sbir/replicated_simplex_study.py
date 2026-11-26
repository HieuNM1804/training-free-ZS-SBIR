"""Complete unseen selection, mechanism controls and frozen transfer on Kaggle."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .concept_benchmark import atomic_json, checksum
from .conditional_completion import full_donor_controls
from .conditional_completion_study import context, evaluate
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .depiction_path_diagnostic import class_interval
from .encoding import save_tensor
from .gallery_diffusion import nearest_gallery
from .joint_gallery_witnesses import load_reference
from .kaggle_unseen_features import DEFAULT_CACHE, SEED, load_split
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .replicated_concept_simplex import barycentric, fixture, index, measurements, select_anchors
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import SKETCHY, mean_percent


SIZES = (12, 24, 36, 48, 72, 96, 144, 216)
PRECISIONS = (16, 32, 64, 128)
KINDS = ("replicated", "unreplicated", "within_fold", "photo_only", "frequency", "random")
DONORS = (8, 16, 32, 64, 128, 256)
DEFAULT_ROOT = DEFAULT_CACHE.parent / "results" / "replicated-concept-simplex"


def read(path):
    return json.loads(path.read_bytes())


def evaluator(path):
    spec = importlib.util.spec_from_file_location("kd_for_simplex_study", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kd = module._retrieval_metrics
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("KD source mismatch")
    return kd


def setup(args):
    split, text, words = load_split(args.cache, args.split)
    kd = evaluator(args.evaluator)
    q, native, photos, t, replay = context(split, text, kd)
    obs = measurements(photos, text[DEFAULT_PROMPTS[0]], t, split.scale, seed=SEED)
    return split, text, words, q, native, photos, t, replay, obs


def protocol(args):
    config = {
        "commit": args.revision, "seed": SEED, "checkpoint": CHECKPOINT, "prompts": DEFAULT_PROMPTS,
        "evaluator": KD_FUNCTION_SHA, "gallery_only": True, "queries_independent": True,
        "sizes": SIZES, "precisions": PRECISIONS, "controls": KINDS[1:], "donors": DONORS,
        "reference": "Locked semantic Dot with all 601 public words and the native prompt per domain.",
        "selection": "Equal weights on the two full unseen Sketchy mAPs, then mean precision. Freeze before TU.",
        "realization": fixture(), "vocabulary_file": checksum(args.cache / "text.pt"),
        "input_manifest": checksum(args.cache / "manifest.json"),
    }
    return save_lock(args.output / "protocol.json", config)


def sweep(args):
    split, _, words, q, native, photos, t, replay, obs = setup(args)
    root = args.output
    signature = read(root / "protocol.json")["sha256"]
    rows, pivots = {}, {}
    for kind in KINDS:
        sequence = select_anchors(obs, max(SIZES), kind)
        pivots[kind] = [words[i] for i in sequence.tolist()]
        for size in SIZES[args.part::args.parts]:
            if len(sequence) < size:
                continue
            ids = sequence[:size]
            source, evidence, prior, _, _ = index(obs, photos, q, ids)
            for precision in PRECISIONS:
                gallery = (precision * evidence + prior).softmax(1)
                rows[f"{kind}:{size}:{precision}"] = mean_percent(evaluate(split, source, gallery, replay))
        scores = [(v[0],key) for key,v in rows.items() if key.startswith(kind+":")]
        print(split.name, "shard", args.part, kind, max(scores) if scores else None, flush=True)
        atomic_json(root/f"progress-{split.name}-{args.part}.json", {"protocol":signature,"metrics":rows})
    if args.part == 0:
        rows["dot"] = mean_percent(evaluate(split, q, native, replay))
        ids, _ = nearest_gallery(photos, max(DONORS))
        banks = full_donor_controls(photos, t, split.scale, ids, DONORS)
        for k, gallery in banks.items():
            rows[f"neighbors:{k}"] = mean_percent(evaluate(split, q, gallery, replay))
    atomic_json(root/f"grid-{split.name}-{args.part}.json", {
        "protocol": signature, "part":args.part, "parts":args.parts,"metrics":rows,"anchors":pivots,
        "queries":len(q),"gallery":len(photos),"device":torch.cuda.get_device_name(),
    })


def lock(args):
    root = args.output
    signature = read(root/"protocol.json")["sha256"]
    tables, hashes = [], {}
    for name in SKETCHY:
        table = {}
        for part in range(args.parts):
            grid = root/f"grid-{name}-{part}.json"
            record = read(grid)
            if record["protocol"] != signature or record["parts"] != args.parts or record["part"] != part:
                raise ValueError("Incompatible grid")
            if table.keys() & record["metrics"].keys():
                raise ValueError("Duplicate grid entry")
            table.update(record["metrics"])
            hashes[grid.name] = checksum(grid)
        tables.append(table)
    def key(n):
        return tuple(sum(tab[n][c] for tab in tables)/len(tables) for c in (0,1))
    both = tables[0].keys() & tables[1].keys()
    selected = {kind: max(sorted(n for n in both if n.startswith(kind+":")), key=key)
                for kind in (*KINDS, "neighbors")}
    spec = {"commit":args.revision,"protocol":signature,"selected":selected,"grids":hashes,
            "selection_metrics":{kind:[table[n] for table in tables] for kind,n in selected.items()},
            "baseline":[table["dot"] for table in tables]}
    save_lock(root/"freeze.json",spec)
    print("LOCK", json.dumps(spec["selected"]),spec["selection_metrics"]["replicated"],flush=True)


def heldout_test(photos, text, scale, size):
    """Use no labels or sketches. Discover using A/B and predict word-conditional means on C."""
    perm = torch.randperm(len(photos),generator=torch.Generator().manual_seed(SEED+1)).to(photos.device)
    n = len(photos)//3
    train, test = photos[perm[n:]], photos[perm[:n]]
    s,t = (text[p] for p in DEFAULT_PROMPTS)
    obs = measurements(train,s,t,scale,seed=SEED)
    held = measurements(test,s,t,scale,seed=SEED)
    uniform = test.mean(0)
    weights = held.target_prior
    denom = (weights * (held.target_mean-uniform).square().sum(1)).sum()
    result = {}
    for kind in KINDS:
        ids = select_anchors(obs,size,kind)
        source_mix = barycentric(obs.source_mean,ids)
        forecast = source_mix @ obs.target_mean[ids]
        error = (held.target_mean-forecast).square().sum(1)
        result[kind] = {"explained_vs_mean":(1 - (weights*error).sum()/denom).item(),
                        "nonanchor_mse":error[~torch.isin(torch.arange(len(error),device=ids.device),ids)].mean().item(),
                        "effective_anchors":len(ids)}
    return result


def official(args):
    root = args.output
    locked = read(root/"freeze.json")
    config = locked["config"]
    if config["commit"] != args.revision:
        raise ValueError("Official runner is not the frozen revision")
    choices = config["selected"]
    split, text, words, q, native, photos, t, replay, obs = setup(args)
    _, size, precision = choices["replicated"].split(":")
    size, precision = int(size), int(precision)
    rows, build = {}, {}
    workers = max(1,min(8,len(os.sched_getaffinity(0))//3))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows["dot"] = evaluate(split, q, native, replay, pool)
        reference, _ = load_reference(split)
        drift = [a-b for a,b in zip(mean_percent(rows["dot"]),mean_percent(reference["dot"][:,:2]),strict=True)]
        if max(map(abs,drift)) > .10:
            raise ValueError(f"Recomputed Dot differs materially from reference in percentage points {drift}")
        def record(name, source, gallery):
            rows[name] = evaluate(split, source, gallery, replay, pool)
            print(split.name,name,mean_percent(rows[name]),flush=True)
        for kind in KINDS:
            _, ck, ct = choices[kind].split(":")
            ck, ct = int(ck), int(ct)
            sequence = select_anchors(obs,max(ck,size),kind)
            for tag,k,tau in ((kind,size,precision),(kind+"_tuned",ck,ct)):
                if tag.endswith("_tuned") and (k,tau)==(size,precision):
                    rows[tag] = rows[kind]
                    continue
                tic = time.monotonic()
                ids = sequence[:k]
                source,evidence,prior,_,p_map = index(obs,photos,q,ids)
                gallery = (tau*evidence+prior).softmax(1)
                torch.cuda.synchronize()
                build[tag] = time.monotonic()-tic
                record(tag, source, gallery)
                if tag == "replicated":
                    r = q[:,ids]
                    record("restrict_query_to_anchor_words",r/r.sum(1,keepdim=True).clamp_min(1e-25),gallery)
                    record("replace_visual_assignment_by_word_mixture",source,native @ p_map)
        k = int(choices["neighbors"].split(":")[1])
        ids, _ = nearest_gallery(photos,k)
        donor = full_donor_controls(photos,t,split.scale,ids,[k])[k]
        record("rgb_neighbor_mean",q,donor)
    checks = heldout_test(photos,text,split.scale,size)
    if split.name in SKETCHY:
        position = SKETCHY.index(split.name)
        for kind in KINDS:
            expected = config["selection_metrics"][kind][position]
            actual = mean_percent(rows[kind+"_tuned"])
            if max(abs(a-b) for a,b in zip(expected,actual,strict=True)) > .0003:
                raise ValueError(f"Original KD replay differs from sweep for {kind}")
    chosen = select_anchors(obs,size)
    target = root/f"{split.name}.pt"
    save_tensor({"qi":split.query_ids,"pi":split.gallery_ids,"labels":split.sketch_labels,"rows":rows},target)
    atomic_json(target.with_suffix(".json"),{
        "commit":args.revision,"selection":locked["sha256"],"device":torch.cuda.get_device_name(),
        "queries":len(q),"gallery":len(photos),"classes":len(split.sketch_labels.unique()),
        "cutoffs":replay.cutoffs,"metrics":{n:mean_percent(v) for n,v in rows.items()},
        "delta_class_macro_ap":{n:class_interval(100*(rows["replicated"][:,0]-v[:,0]),split.sketch_labels,seed=SEED)
                               for n,v in rows.items()},
        "build_seconds_after_measurements":build,"heldout_visual_reference":checks,
        "anchors":[words[i] for i in chosen.tolist()],"historical_dot_drift_points":drift,
        "per_query_sha256":checksum(target),
    })


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("protocol","sweep","lock","official"))
    parser.add_argument("--revision",required=True)
    parser.add_argument("--cache",type=Path,default=DEFAULT_CACHE)
    parser.add_argument("--output",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--evaluator",type=Path,default=DEFAULT_CACHE.parent/"real-sketchy-smoke/evaluator.py")
    parser.add_argument("--split",choices=(*SKETCHY,"tuberlin"))
    parser.add_argument("--part",type=int,default=0)
    parser.add_argument("--parts",type=int,default=2)
    args=parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.set_float32_matmul_precision("highest")
    if args.stage=="sweep" and args.split not in SKETCHY:
        parser.error("Only Sketchy is used for selection")
    {"protocol":protocol,"sweep":sweep,"lock":lock,"official":official}[args.stage](args)
