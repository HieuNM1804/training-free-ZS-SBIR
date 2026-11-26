"""Vocabulary design for repeated anonymous visual reference tasks, using one frozen CLIP."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import torch

from .concept_benchmark import atomic_json, checksum
from .conditional_completion_study import context, evaluate
from .crossdomain_inputs import CHECKPOINT
from .depiction_path_diagnostic import class_interval
from .encoding import save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE, SEED, load_split
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .reference_lexicon import fixture, reference_task
from .replicated_simplex_study import evaluator, read
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import SKETCHY, mean_percent


SIZES = (12, 24, 36, 48, 72, 96, 144, 192)
GAMES = 16
DRAWS = 8
SWEEPS = 2
ARMS = ("joint","one_to_one","free_names","common_names","mean_cell","soft_cell")
DEFAULT_ROOT = DEFAULT_CACHE.parent/"results"/"shared-reference-choice"


def inputs(args):
    split,text,words=load_split(args.cache,args.split)
    q,native,photos,t,replay=context(split,text,evaluator(args.evaluator))
    return split,q,native,photos,(split.scale*(photos@text[DEFAULT_PROMPTS[0]].T),split.scale*(photos@t.T)),replay,words


def book(photos,logits,q,size,*,audit=False,fit_steps=15):
    matrices,labels,details,codes=[],[],[],[]
    for game in range(GAMES):
        readouts,owner,names,stats=reference_task(photos,*logits,q,size,SEED+2*game,DRAWS,SWEEPS,
                                                 audit=audit,fit_steps=fit_steps)
        matrices.append(readouts)
        labels.append(owner)
        details.append(stats)
        codes.append({arm:v.cpu().tolist() for arm,v in names.items()})
    starts,offset= [],0
    for result in matrices:
        starts.append(offset)
        offset+=result["joint"].size(1)
    index=photos.new_zeros(len(photos),offset)
    for start,owner in zip(starts,labels,strict=True):
        index.scatter_(1,(start+owner)[:,None],1.)
    return {arm:torch.cat([m[arm] for m in matrices],1)/GAMES for arm in matrices[0]},index,details,codes


def protocol(args):
    save_lock(args.output/"protocol.json",{"commit":args.revision,"seed":SEED,"sizes":SIZES,"games":GAMES,
        "listener_draws_per_cell":DRAWS,"coordinate_sweeps":SWEEPS,"visual_lloyd_steps":15,
        "inputs":checksum(args.cache/"manifest.json"),"evaluator":KD_FUNCTION_SHA,"backbone":CHECKPOINT,
        "prompts":DEFAULT_PROMPTS,"toy":fixture(),"selection":"Full unseen Sketchy mean mAP, then mean precision. Same lock on TU.",
        "hypothesis":"Labels that identify an image among alternatives form a code; changing a competitor label changes the utility of the current label. Anonymous reference tasks probe this without benchmark names."})


def sweep(args):
    split,q,_,photos,logits,replay,_=inputs(args)
    root=args.output
    sign=read(root/"protocol.json")["sha256"]
    rows={}
    for size in SIZES[args.part::args.parts]:
        t0=time.monotonic()
        matrices,index,stats,_=book(photos,logits,q,size)
        for arm in ARMS:
            rows[f"{arm}:{size}"]=mean_percent(evaluate(split,matrices[arm],index,replay))
        print(split.name,size,{arm:round(rows[f"{arm}:{size}"][0],3) for arm in ARMS},
              "mean edits",sum(s["edits"] for s in stats)/GAMES,"seconds",round(time.monotonic()-t0,1),flush=True)
        atomic_json(root/f"progress-{split.name}-{args.part}.json",{"protocol":sign,"metrics":rows})
    atomic_json(root/f"grid-{split.name}-{args.part}.json",{"protocol":sign,"metrics":rows,"part":args.part,
                "parts":args.parts,"queries":len(q),"gallery":len(photos)})


def lock(args):
    root=args.output
    sign=read(root/"protocol.json")["sha256"]
    grids,hashes=[],{}
    for name in SKETCHY:
        merged={}
        for part in range(args.parts):
            path=root/f"grid-{name}-{part}.json"
            blob=read(path)
            if blob["protocol"]!=sign or (blob["part"],blob["parts"])!=(part,args.parts) or merged.keys()&blob["metrics"].keys():
                raise ValueError("Unexpected sweep shard")
            hashes[path.name]=checksum(path)
            merged.update(blob["metrics"])
        if len(merged)!=len(SIZES)*len(ARMS):raise ValueError("Incomplete sweep")
        grids.append(merged)
    def mean_at(key):
        return tuple(sum(g[key][c] for g in grids)/2 for c in (0,1))
    best={arm:max(SIZES,key=lambda n:mean_at(f"{arm}:{n}")) for arm in ARMS}
    values={arm:[tab[f"{arm}:{k}"] for tab in grids] for arm,k in best.items()}
    save_lock(root/"freeze.json",{"commit":args.revision,"protocol":sign,"selected":best,"grids":hashes,
                                  "selected_on_sketchy":values})
    print("FROZEN",best,values,flush=True)


def official(args):
    split,q,native,photos,logits,replay,words=inputs(args)
    root=args.output
    lock=read(root/"freeze.json")
    if args.revision!=lock["config"]["commit"]:raise ValueError("Frozen implementation changed")
    best=lock["config"]["selected"]
    rows,details,selected_codes={},None,None
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows["dot"]=evaluate(split,q,native,replay,pool)
        for size in sorted(set(best.values())):
            matrices,index,stats,codes=book(photos,logits,q,size,audit=(size==best["joint"]))
            if size==best["joint"]:
                details=stats
                selected_codes=codes
            for arm,x in matrices.items():
                at_primary=size==best["joint"]
                at_optimum=size==best.get(arm)
                if at_primary or at_optimum:
                    result=evaluate(split,x,index,replay,pool)
                    if at_primary:rows[arm]=result
                    if at_optimum:rows[arm+"_tuned"]=result
                    print(split.name,arm,size,mean_percent(result),flush=True)
            del matrices,index
        unrefined,index,_,_=book(photos,logits,q,best["joint"],fit_steps=0)
        rows["random_voronoi"]=evaluate(split,unrefined["joint"],index,replay,pool)
        rows["voronoi_mean_cell"]=evaluate(split,unrefined["mean_cell"],index,replay,pool)
    if split.name in SKETCHY:
        j=SKETCHY.index(split.name)
        for arm in ARMS:
            expected=lock["config"]["selected_on_sketchy"][arm][j]
            values=mean_percent(rows[arm+"_tuned"])
            if max(abs(a-b) for a,b in zip(expected,values,strict=True))>.00035:
                raise ValueError(f"Sweep and original KD disagree for {arm}")
    target=root/f"{split.name}.pt"
    save_tensor({"qi":split.query_ids,"pi":split.gallery_ids,"labels":split.sketch_labels,"rows":rows,
                 "game_codes":selected_codes},target)
    atomic_json(target.with_suffix(".json"),{"commit":args.revision,"selection":lock["sha256"],
        "queries":len(q),"gallery":len(photos),"classes":len(split.sketch_labels.unique()),"cutoffs":replay.cutoffs,
        "game_diagnostics":details,"first_code":[words[i] for i in selected_codes[0]["joint"]],
        "metrics":{n:mean_percent(v) for n,v in rows.items()},
        "joint_minus_control_class_macro_ap":{n:class_interval(100*(rows["joint"][:,0]-v[:,0]),split.sketch_labels,seed=SEED)
                                             for n,v in rows.items()},"per_query_sha256":checksum(target)})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("protocol","sweep","lock","official"))
    parser.add_argument("--split",choices=(*SKETCHY,"tuberlin"))
    parser.add_argument("--revision",required=True)
    parser.add_argument("--cache",type=Path,default=DEFAULT_CACHE)
    parser.add_argument("--output",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--evaluator",type=Path,default=DEFAULT_CACHE.parent/"real-sketchy-smoke/evaluator.py")
    parser.add_argument("--part",type=int,default=0)
    parser.add_argument("--parts",type=int,default=2)
    args=parser.parse_args()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if args.stage=="sweep" and args.split not in SKETCHY:parser.error("Only tune on Sketchy")
    with torch.inference_mode():
        {"protocol":protocol,"sweep":sweep,"lock":lock,"official":official}[args.stage](args)
