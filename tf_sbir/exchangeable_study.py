"""Bounded test of latent concept repetition in five soft observations of one query."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum
from .conditional_completion_study import context
from .crossdomain_inputs import CHECKPOINT
from .depiction_path_diagnostic import class_interval
from .encoding import save_tensor
from .exchangeable_evidence import coefficients, example, interpolate
from .joint_gallery_witnesses import load_reference
from .kaggle_query_windows import SCALES, load_windows
from .kaggle_unseen_features import DEFAULT_CACHE, SEED, load_split
from .locked_geometry_words import score_rows
from .profiled_study import true_sources
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .replicated_simplex_study import evaluator, read
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import SKETCHY, mean_percent, metric_rows


DEFAULT_ROOT = DEFAULT_CACHE.parent / "results" / "exchangeable-concept-observations"
ALPHAS = (.125, .5, 2., 8., 32.)
ESTIMATORS = ("partial", "two_extremes", "one_draw", "independent", "average")
BATCH = 96


def protocol(args):
    return save_lock(args.output / "protocol.json", {
        "commit": args.revision, "checkpoint": CHECKPOINT, "seed": SEED, "prompts": DEFAULT_PROMPTS,
        "evaluator": KD_FUNCTION_SHA, "feature_manifest": checksum(args.cache / "manifest.json"),
        "observations": "Whole sketch and four corner windows, average native and reflected word posterior within view",
        "gallery": "Each unmodified, native photo word posterior is its own Dirichlet base measure",
        "scales": SCALES, "alpha": ALPHAS, "controls": ESTIMATORS[1:], "example": example(),
        "selection": "Select on equal-weight full unseen Sketchy mAP. Use mean precision as tie-break. Freeze before transfer.",
        "stop": "Require at least four absolute mAP points on each Sketchy split and an advantage over controls for partial sharing. Otherwise stop this line.",
    })


def setup(args):
    split, text, _ = load_split(args.cache, args.split)
    q, p, _, _, replay = context(split, text, evaluator(args.evaluator))
    return split, q, p, text[DEFAULT_PROMPTS[0]], load_windows(args.cache, split), replay


def view_evidence(q, windows, src, scale, window):
    crop = windows[:,4*window:4*window+4].float()
    prob = (scale*(crop @ src.T)).softmax(-1).mean(2)
    return torch.cat((q[:,None],prob),dim=1)


def describe(key):
    arm,size,*rest = key.split(":")
    return arm, int(size), float(rest[0]) if rest else None


def simplex_score(kind, alpha, c, t):
    if kind == "one_draw":
        return c[:,0]
    if kind == "independent":
        return c[:,-1]
    if kind == "two_extremes":
        # Same marginal probability of sharing a latent draw for EVERY pair of views.
        return (c[:,0] + alpha*c[:,-1]) / (1+alpha)
    if kind == "partial":
        return interpolate(c,t,alpha)
    raise ValueError(kind)


def scores_for(keys, views, p, coeff, totals):
    for key in keys:
        arm,_,alpha = describe(key)
        if arm == "average":
            yield key, views.mean(1) @ p.T
        elif arm == "dot":
            yield key, views[:,0] @ p.T
        else:
            yield key, simplex_score(arm,alpha,coeff,totals)


def keys_for(size):
    return [f"{arm}:{size}:{a}" for arm in ("partial","two_extremes") for a in ALPHAS] + [
        f"{arm}:{size}" for arm in ("one_draw","independent","average")]


def official_keys(winners):
    _,size,alpha = describe(winners["partial"])
    aliases = {arm: (f"{arm}:{size}:{alpha}" if arm in ("partial","two_extremes") else f"{arm}:{size}")
               for arm in ESTIMATORS}
    aliases["dot"] = f"dot:{size}"
    aliases.update({arm+"_tuned":key for arm,key in winners.items()})
    return aliases


def sweep(args):
    split,q,p,t,windows,replay = setup(args)
    root = args.output
    signature = read(root/"protocol.json")["sha256"]
    size = args.part
    keys = keys_for(size)
    truth, gallery = split.sketch_labels.cuda(), split.photo_labels.cuda()
    m, cutoff = replay.cutoffs
    positions = torch.arange(1,(m or len(p))+1,device=p.device)
    collected = defaultdict(list)
    for start in range(0,len(q),BATCH):
        sl=slice(start,start+BATCH)
        views=view_evidence(q[sl],windows[sl],t,split.scale,size)
        c,totals=coefficients(views,p)
        for key,score in scores_for(keys,views,p,c,totals):
            collected[key].append(metric_rows(score,truth[sl],gallery,positions,cutoff).cpu())
    table={key:mean_percent(torch.cat(v)) for key,v in collected.items()}
    atomic_json(root/f"grid-{split.name}-{size}.json",{
        "commit":args.revision,"protocol":signature,"part":size,"crop_fraction":SCALES[size],
        "queries":len(q),"gallery":len(p),"metrics":table,
    })
    print(split.name, "window fraction",SCALES[size],table,flush=True)


def lock(args):
    root=args.output
    sig=read(root/"protocol.json")["sha256"]
    grids,hashes=[],{}
    for name in SKETCHY:
        table={}
        for i in range(2):
            path=root/f"grid-{name}-{i}.json"
            blob=read(path)
            if blob["protocol"]!=sig or blob["part"]!=i or set(blob["metrics"])!=set(keys_for(i)):
                raise ValueError("Invalid Sketchy selection table")
            hashes[path.name]=checksum(path)
            table.update(blob["metrics"])
        grids.append(table)
    def best_key(key):
        return tuple(sum(tab[key][c] for tab in grids)/2 for c in (0,1))
    best={arm:max(sorted(k for k in grids[0] if k.startswith(arm+":")),key=best_key) for arm in ESTIMATORS}
    values={arm:[tab[key] for tab in grids] for arm,key in best.items()}
    save_lock(root/"freeze.json",{"commit":args.revision,"protocol":sig,"grids":hashes,"best":best,
                                  "selected_on_sketchy":values})
    print("FROZEN",best,values,flush=True)


def official(args):
    root=args.output
    frozen=read(root/"freeze.json")
    if frozen["config"]["commit"]!=args.revision:
        raise ValueError("Frozen revision differs from original KD replay")
    split,q,p,t,windows,replay=setup(args)
    best=frozen["config"]["best"]
    aliases=official_keys(best)
    groups=defaultdict(list)
    for key in sorted(set(aliases.values())):
        groups[describe(key)[1]].append(key)
    _,primary_size,alpha=describe(best["partial"])
    gallery,truth=split.photo_labels.cuda(),split.sketch_labels.cuda()
    baseline,_=load_reference(split)
    paired_ids=true_sources(split,args.cache) if split.name in SKETCHY else None
    diagnostic=defaultdict(list)
    collected=defaultdict(list)
    tiny_at_cutoff=0
    with ThreadPoolExecutor(max_workers=6) as pool:
        for size,keys in groups.items():
            for start in range(0,len(q),BATCH):
                sl=slice(start,min(start+BATCH,len(q)))
                views=view_evidence(q[sl],windows[sl],t,split.scale,size)
                c,totals=coefficients(views,p)
                for key,score in scores_for(keys,views,p,c,totals):
                    rows=score_rows(score,split.sketch_labels[sl],gallery,split.name,replay,pool)
                    collected[key].append(rows)
                    if key==best["partial"]:
                        threshold=score.topk(replay.cutoffs[1]).values[:,-1]
                        tiny_at_cutoff+=int((threshold<=torch.finfo(score.dtype).tiny).sum())
                if paired_ids is None or size!=primary_size:
                    continue
                sid=paired_ids[sl]
                ok=sid>=0
                if not ok.any():continue
                where=torch.where(ok)[0]
                sid=sid[ok]
                if not torch.equal(gallery[sid],truth[sl][ok]):raise ValueError("Mismatched sketch/photo identifier")
                dot=q[sl][ok]@p.T
                gap=(dot-dot.gather(1,sid[:,None])).abs()
                gap.masked_fill_(gallery[None]!=truth[sl][ok,None],torch.inf)
                gap.scatter_(1,sid[:,None],torch.inf)
                rival=gap.argmin(1)
                # Read the whole-sketch description only after observing the four partial views.
                crops,crop_totals=coefficients(views[ok,1:],p)
                full=c[ok]
                allrow=torch.arange(len(sid),device=q.device)
                diagnostic["query_rows"].append((start+where).cpu())
                for arm in ("partial","two_extremes","independent","one_draw"):
                    num=simplex_score(arm,alpha,full,totals)[allrow[:,None],torch.stack((sid,rival),1)]
                    den=simplex_score(arm,alpha,crops,crop_totals)[allrow[:,None],torch.stack((sid,rival),1)]
                    # Restore the whole-view scaling. All earlier scalings cancel in the ratio.
                    actual=num.clamp_min(1e-38).log()-den.clamp_min(1e-38).log()+q[sl][ok].amax(1,keepdim=True).log()
                    diagnostic[arm].append(actual.cpu())
            print(split.name,"official crop fraction",SCALES[size],"finished",flush=True)
    unique={key:torch.cat(v) for key,v in collected.items()}
    rows={name:unique[key] for name,key in aliases.items()}
    prior=mean_percent(baseline["dot"][:,:2])
    reproduced=mean_percent(rows["dot"][:,:2])
    drift=[x-y for x,y in zip(reproduced,prior,strict=True)]
    if max(map(abs,drift))>.1 or tiny_at_cutoff:
        raise ValueError(f"Unverified ranks, baseline drift={drift}, underflowed prefixes={tiny_at_cutoff}")
    if split.name in SKETCHY:
        pos=SKETCHY.index(split.name)
        for name,values in frozen["config"]["selected_on_sketchy"].items():
            error=max(abs(a-b) for a,b in zip(mean_percent(rows[name+"_tuned"][:,:2]),values[pos],strict=True))
            if error>.0004:raise ValueError(f"Original KD disagrees for {name}: {error}")
    diagnostics={key:torch.cat(v) for key,v in diagnostic.items()}
    summaries={}
    for arm,values in diagnostics.items():
        if arm=="query_rows":continue
        true,foil=values.double().T
        summaries[arm]={"true_log_predictive":true.mean().item(),"foil_log_predictive":foil.mean().item(),
                        "mean_paired_advantage":(true-foil).mean().item(),"queries":len(true)}
    dest=root/f"{split.name}.pt"
    save_tensor({"qi":split.query_ids,"pi":split.gallery_ids,"labels":split.sketch_labels,
                 "rows":rows,"whole_view_prediction":diagnostics},dest)
    atomic_json(dest.with_suffix(".json"),{
        "commit":args.revision,"selection":frozen["sha256"],"queries":len(q),"gallery":len(p),
        "classes":len(split.sketch_labels.unique()),"cutoffs":replay.cutoffs,"device":torch.cuda.get_device_name(),
        "metrics":{arm:mean_percent(v[:,:2]) for arm,v in rows.items()},
        "partial_minus_control_class_macro_ap":{arm:class_interval(100*(rows["partial"][:,0]-v[:,0]),
                                           split.sketch_labels,seed=SEED) for arm,v in rows.items()},
        "observed_delta_top1_counts":{arm:{"fixed":int(((rows["partial"][:,3]>0)&(v[:,3]==0)).sum()),
                                           "damaged":int(((rows["partial"][:,3]==0)&(v[:,3]>0)).sum())}
                                      for arm,v in rows.items()},
        "heldout_whole_description":summaries,"baseline_drift_points":drift,
        "primary_near_zero_at_cutoff":tiny_at_cutoff,"per_query_sha256":checksum(dest),
    })


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("protocol","sweep","lock","official"))
    parser.add_argument("--revision",required=True)
    parser.add_argument("--split",choices=(*SKETCHY,"tuberlin"))
    parser.add_argument("--cache",type=Path,default=DEFAULT_CACHE)
    parser.add_argument("--output",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--evaluator",type=Path,default=DEFAULT_CACHE.parent/"real-sketchy-smoke/evaluator.py")
    parser.add_argument("--part",type=int,choices=(0,1),default=0)
    args=parser.parse_args()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    if args.stage=="sweep" and args.split not in SKETCHY:
        parser.error("Sweep on Sketchy only")
    with torch.inference_mode():
        {"protocol":protocol,"sweep":sweep,"lock":lock,"official":official}[args.stage](args)
