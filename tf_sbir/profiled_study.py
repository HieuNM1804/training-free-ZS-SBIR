"""Full unseen profile coding, matched mixtures, and prospective dataset transfer."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,checksum
from .conditional_completion_study import context,evaluate
from .crossdomain_inputs import CHECKPOINT
from .depiction_path_diagnostic import class_interval
from .encoding import save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE,SEED,load_split,index_file
from .profiled_prediction import bank_models,fixture,forecast
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .replicated_simplex_study import evaluator,read
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import SKETCHY,mean_percent


DEFAULT_ROOT=DEFAULT_CACHE.parent/"results"/"profiled-lexical-prediction"
COMPONENTS=(32,64,128,256,512,1024)
SCALES=(8,16,32,64,128,256)
LAWS=("reports","mixture")
ARMS=("profile","bayes","unnormalized")


def inputs(args):
    split,text,words=load_split(args.cache,args.split)
    q,native,photos,_,replay=context(split,text,evaluator(args.evaluator))
    return split,q,native,photos,replay,words


def protocol(args):
    save_lock(args.output/"protocol.json",{"commit":args.revision,"components":COMPONENTS,"scales":SCALES,
        "laws":LAWS,"seed":SEED,"prompts":DEFAULT_PROMPTS,"checkpoint":CHECKPOINT,"evaluator":KD_FUNCTION_SHA,
        "manifest":checksum(args.cache/"manifest.json"),"fixture":fixture(),
        "selection":"Select each estimator on the mean full unseen Sketchy mAP, precision breaks ties. TU only after lock.",
        "hypothesis":"Under partial observation different visual populations can supply the strongest explanation for different words. Conditional minimax regret yields a unique normalized forecast."})


def sweep(args):
    split,q,native,photos,replay,_=inputs(args)
    root=args.output
    sig=read(root/"protocol.json")["sha256"]
    table={}
    for k in COMPONENTS[args.part::args.parts]:
        models=bank_models(photos,native,k,SEED)
        for law in LAWS:
            for scale in SCALES:
                banks,_=forecast(models,scale,law,arms=ARMS)
                for arm,b in banks.items():
                    table[f"{arm}:{law}:{k}:{scale}"]=mean_percent(evaluate(split,q,b,replay))
        peaks={arm:max((v[0],key) for key,v in table.items() if key.startswith(arm+":") and key.split(":")[2]==str(k))
               for arm in ARMS}
        print(split.name,k,peaks,flush=True)
        atomic_json(root/f"progress-{split.name}-{args.part}.json",{"metrics":table,"protocol":sig})
    atomic_json(root/f"grid-{split.name}-{args.part}.json",{"metrics":table,"protocol":sig,"part":args.part,
                "parts":args.parts,"queries":len(q),"gallery":len(native)})


def lock(args):
    root=args.output
    sig=read(root/"protocol.json")["sha256"]
    grids,hashes=[],{}
    for s in SKETCHY:
        table={}
        for part in range(args.parts):
            path=root/f"grid-{s}-{part}.json"
            blob=read(path)
            if blob["protocol"]!=sig or blob["part"]!=part or table.keys() & blob["metrics"].keys():
                raise ValueError("Invalid grid shard")
            table.update(blob["metrics"])
            hashes[path.name]=checksum(path)
        if len(table)!=len(ARMS)*len(LAWS)*len(COMPONENTS)*len(SCALES):
            raise ValueError("Incomplete grid")
        grids.append(table)
    def key(id):
        return tuple(sum(table[id][col] for table in grids)/2 for col in (0,1))
    arms={arm:arm+":" for arm in ARMS}
    arms.update({law:"profile:"+law+":" for law in LAWS})
    best={arm:max(sorted(n for n in grids[0] if n.startswith(prefix)),key=key) for arm,prefix in arms.items()}
    selected={arm:[tab[id] for tab in grids] for arm,id in best.items()}
    save_lock(root/"freeze.json",{"protocol":sig,"commit":args.revision,"grids":hashes,"selected":best,
                                  "selected_on_sketchy":selected})
    print("FROZEN",best,selected,flush=True)


def true_sources(split,cache):
    spec=read(cache/"manifest.json")
    record=spec["datasets"]["sketchy"]
    order=torch.load(index_file(cache,"sketchy",record),weights_only=True)
    photo_paths=order["photo"]["paths"]
    sketch_paths=order["sketch"]["paths"]
    lookup={}
    for j,i in enumerate(split.gallery_ids.tolist()):
        p=Path(photo_paths[i])
        if (p.parent.name,p.stem) in lookup: raise ValueError("Source key repeats")
        lookup[(p.parent.name,p.stem)]=j
    ids=[]
    for i in split.query_ids.tolist():
        p=Path(sketch_paths[i])
        stem,sep,suffix=p.stem.rpartition("-")
        ids.append(lookup.get((p.parent.name,stem),-1) if sep and suffix.isdigit() else -1)
    return torch.tensor(ids,device="cuda")


def official(args):
    root=args.output
    freeze=read(root/"freeze.json")
    if freeze["config"]["commit"]!=args.revision:raise ValueError("Frozen code mismatch")
    best=freeze["config"]["selected"]
    split,q,native,photos,replay,words=inputs(args)
    rows,diagnostics,predictive={},{},{}
    if split.name in SKETCHY:
        ids=true_sources(split,args.cache)
        paired=ids>=0
    def score(name,b,pool):
        rows[name]=evaluate(split,q,b,replay,pool)
        if split.name in SKETCHY:
            normalized=b/b.sum(1,keepdim=True).clamp_min(1e-30)
            predictive[name]=-(q[paired]*normalized[ids[paired]].clamp_min(1e-30).log()).sum(1).cpu()
        print(split.name,name,mean_percent(rows[name]),flush=True)
    primary=best["profile"].split(":")
    settings={(int(n.split(":")[2]),n.split(":")[1],int(n.split(":")[3])) for n in best.values()}
    with ThreadPoolExecutor(max_workers=6) as pool:
        score("dot",native,pool)
        last_k=-1
        for k,law,scale in sorted(settings):
            if k!=last_k:
                models=bank_models(photos,native,k,SEED)
                last_k=k
            matched=(k,law,scale)==(int(primary[2]),primary[1],int(primary[3]))
            banks,diag=forecast(models,scale,law,arms=(*ARMS,"map","empirical_bayes","half_profile") if matched else ARMS)
            for arm,b in banks.items():
                name=f"{arm}:{law}:{k}:{scale}"
                aliases=[key+"_tuned" for key,winner in best.items() if winner==name]
                if matched: aliases=[arm,*aliases]
                if not aliases:continue
                score(aliases[0],b,pool)
                for alias in aliases[1:]:
                    rows[alias]=rows[aliases[0]]
                    if split.name in SKETCHY:predictive[alias]=predictive[aliases[0]]
            if matched:diagnostics["folds"]=diag
    if split.name in SKETCHY:
        j=SKETCHY.index(split.name)
        for n in best:
            expected=freeze["config"]["selected_on_sketchy"][n][j]
            if max(abs(a-b) for a,b in zip(mean_percent(rows[n+"_tuned"]),expected,strict=True))>.00035:
                raise ValueError("Sweep differs from KD replay for "+n)
        diagnostics["paired_source_queries"]=int(paired.sum())
        diagnostics["paired_name_cross_entropy"]={n:v.double().mean().item() for n,v in predictive.items()}
    dest=root/f"{split.name}.pt"
    save_tensor({"qi":split.query_ids,"pi":split.gallery_ids,"labels":split.sketch_labels,"rows":rows,
                 "paired_predictive_loss":predictive},dest)
    atomic_json(dest.with_suffix(".json"),{"commit":args.revision,"selection":freeze["sha256"],
        "queries":len(q),"gallery":len(native),"classes":len(split.sketch_labels.unique()),"cutoffs":replay.cutoffs,
        "metrics":{n:mean_percent(v) for n,v in rows.items()},"diagnostics":diagnostics,
        "primary_minus_control_class_macro_ap":{n:class_interval(100*(rows["profile"][:,0]-v[:,0]),
                                    split.sketch_labels,seed=SEED) for n,v in rows.items()},
        "per_query_sha256":checksum(dest)})


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
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if args.stage=="sweep" and args.split not in SKETCHY:parser.error("Select only on Sketchy")
    with torch.inference_mode():
        {"protocol":protocol,"sweep":sweep,"lock":lock,"official":official}[args.stage](args)
