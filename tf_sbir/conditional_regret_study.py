"""One conditional coding rule for sketch and photo, under an unknown mixture of visual subjects."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,checksum
from .conditional_completion_study import context,evaluate
from .conditional_regret import fixture,pair_codes,population_parameters,source_code,source_likelihood,target_code
from .crossdomain_inputs import CHECKPOINT
from .depiction_path_diagnostic import class_interval
from .encoding import save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE,SEED,load_split
from .profiled_prediction import bank_models
from .profiled_study import DEFAULT_ROOT,true_sources
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .replicated_simplex_study import evaluator,read
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import SKETCHY,mean_percent


SIZES=(64,128,256,512)
SOURCE_TEMP=(32,64,100,160)
TARGET_TEMP=(16,32,64)
ODDS=(1,2,8,32)
LAWS=("reports","mixture")


def dataset(args):
    split,text,_=load_split(args.cache,args.split)
    q,p,photos,_,replay=context(split,text,evaluator(args.evaluator))
    return split,text[DEFAULT_PROMPTS[0]],q,p,photos,replay


def protocol(args):
    save_lock(args.output/"protocol.json",{"commit":args.revision,"seed":SEED,"sizes":SIZES,"odds":ODDS,
        "source_temperatures":SOURCE_TEMP,"target_temperatures":TARGET_TEMP,"laws":LAWS,"evaluator":KD_FUNCTION_SHA,
        "prompts":DEFAULT_PROMPTS,"checkpoint":CHECKPOINT,"manifest":checksum(args.cache/"manifest.json"),
        "exact_small_example":fixture(torch.device("cuda")),
        "parameter_selection":"Equal-weight mean mAP on both complete unseen Sketchy splits with precision tie-break. Lock before TU.",
        "hypothesis":"The ratio of two possible latent subject odds can move by at most Gamma. Predict the name with minimal worst conditional log-loss regret after observing the image in its own domain."})


def fold_sources(split,text,models,law):
    result={}
    for t in SOURCE_TEMP:
        obs=source_likelihood(split,text,t)
        for odds in ODDS:
            qs=[]
            for model in models:
                theta,base=population_parameters(model,law)
                qs.append(source_code(obs,theta,base,odds)[0])
            result[t,odds]=torch.cat(qs,1)
    return result


def fold_photos(n,models,law,scale,odds):
    v=models[0].observed.size(1)
    values=models[0].observed.new_zeros(n,2*v)
    for i,model in enumerate(models):
        theta,base=population_parameters(model,law)
        codes,_=target_code(model,theta,base,scale,odds)
        values[model.rows,i*v:(i+1)*v]=codes
    return values


def sweep(args):
    root=args.output
    sig=read(root/"protocol.json")["sha256"]
    split,text,_,native,photos,replay=dataset(args)
    table={}
    for k in SIZES[args.part::args.parts]:
        models=bank_models(photos,native,k,SEED)
        for law in LAWS:
            qs=fold_sources(split,text,models,law)
            for scale in TARGET_TEMP:
                for odds in ODDS:
                    ys=fold_photos(len(photos),models,law,scale,odds)
                    for temp in SOURCE_TEMP:
                        key=f"{law}:{k}:{temp}:{scale}:{odds}"
                        table[key]=mean_percent(evaluate(split,qs[temp,odds],ys,replay))
            peaks={tag:max((v[0],key) for key,v in table.items()
                           if key.startswith(f"{law}:{k}:") and (int(key.split(":")[-1])>1)==robust)
                   for tag,robust in (("robust",True),("average",False))}
            print(split.name,k,law,peaks,flush=True)
            atomic_json(root/f"progress-{split.name}-{args.part}.json",{"protocol":sig,"metrics":table})
            del qs
    atomic_json(root/f"grid-{split.name}-{args.part}.json",{"protocol":sig,"metrics":table,"split":split.name,
                "part":args.part,"parts":args.parts,"queries":len(split.query_ids),"gallery":len(photos)})


def lock(args):
    root=args.output
    signature=read(root/"protocol.json")["sha256"]
    grids,hashes=[],{}
    for name in SKETCHY:
        tab={}
        for part in range(args.parts):
            path=root/f"grid-{name}-{part}.json"
            rec=read(path)
            if rec["protocol"]!=signature or (rec["parts"],rec["part"])!=(args.parts,part) or tab.keys()&rec["metrics"].keys():
                raise ValueError("Grid mismatch")
            tab.update(rec["metrics"])
            hashes[path.name]=checksum(path)
        if len(tab)!=len(SIZES)*len(SOURCE_TEMP)*len(TARGET_TEMP)*len(ODDS)*len(LAWS):raise ValueError("Incomplete grid")
        grids.append(tab)
    def score(key):
        return tuple(sum(g[key][i] for g in grids)/2 for i in (0,1))
    keys=sorted(grids[0])
    picks={"regret":max((k for k in keys if not k.endswith(":1")),key=score),
           "bayes":max((k for k in keys if k.endswith(":1")),key=score)}
    vals={arm:[tab[key] for tab in grids] for arm,key in picks.items()}
    save_lock(root/"freeze.json",{"commit":args.revision,"protocol":signature,"grids":hashes,"selected":picks,
                                  "selected_on_sketchy":vals})
    print("FROZEN",picks,vals,flush=True)


def official(args):
    root=args.output
    locked=read(root/"freeze.json")
    if locked["config"]["commit"]!=args.revision:raise ValueError("Frozen code mismatch")
    split,text,q0,p0,photos,replay=dataset(args)
    rows,losses,diagnostics={},{},{}
    if split.name in SKETCHY:
        ids=true_sources(split,args.cache)
        valid=ids>=0
    def measure(name,q,p,pool):
        rows[name]=evaluate(split,q,p,replay,pool)
        if split.name in SKETCHY:
            # Independent diagnostic of actual sketch/photo name agreement at the known source.
            paired=(q[valid]*p[ids[valid]]).sum(1)
            losses[name]=-paired.clamp_min(1e-30).log().cpu()
        print(split.name,name,mean_percent(rows[name]),flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        measure("dot",q0,p0,pool)
        for arm,key in locked["config"]["selected"].items():
            law,k,st,tt,odds=key.split(":")
            k,st,tt,odds=map(int,(k,st,tt,odds))
            models=bank_models(photos,p0,k,SEED)
            variants={arm:(odds,odds)}
            if arm=="regret":
                variants.update({"source_only":(odds,1),"target_only":(1,odds),"matched_bayes":(1,1)})
            for name,(os,ot) in variants.items():
                q,p,diagnostics[name]=pair_codes(split,text,models,law,st,tt,os,ot)
                measure(name,q,p,pool)
            if arm=="regret":
                raw=sum(source_likelihood(split,text,st))/2
                q,p,_=pair_codes(split,text,models,law,st,tt,odds,odds)
                measure("no_source_population",torch.cat((raw,raw),1),p,pool)
                v=p0.size(1)
                direct=p.new_zeros(p.shape)
                for fold,model in enumerate(models):
                    direct[model.rows,fold*v:(fold+1)*v]=p0[model.rows]
                measure("no_target_population",q,direct,pool)
    if split.name in SKETCHY:
        j=SKETCHY.index(split.name)
        for arm,values in locked["config"]["selected_on_sketchy"].items():
            if max(abs(a-b) for a,b in zip(mean_percent(rows[arm]),values[j],strict=True))>.00035:
                raise ValueError("Original KD differs from selection for "+arm)
        diagnostics["paired_source_queries"]=int(valid.sum())
        diagnostics["true_photo_name_disagreement_nats"]={n:v.double().mean().item() for n,v in losses.items()}
    dest=root/f"{split.name}.pt"
    save_tensor({"qi":split.query_ids,"pi":split.gallery_ids,"labels":split.sketch_labels,
                 "rows":rows,"source_pair_loss":losses},dest)
    atomic_json(dest.with_suffix(".json"),{"commit":args.revision,"selection":locked["sha256"],
        "queries":len(q0),"gallery":len(p0),"classes":len(split.sketch_labels.unique()),"cutoffs":replay.cutoffs,
        "metrics":{n:mean_percent(v) for n,v in rows.items()},"diagnostics":diagnostics,
        "primary_minus_control_class_macro_ap":{n:class_interval(100*(rows["regret"][:,0]-v[:,0]),
                                  split.sketch_labels,seed=SEED) for n,v in rows.items()},
        "per_query_sha256":checksum(dest)})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("protocol","sweep","lock","official"))
    parser.add_argument("--split",choices=(*SKETCHY,"tuberlin"))
    parser.add_argument("--revision",required=True)
    parser.add_argument("--cache",type=Path,default=DEFAULT_CACHE)
    parser.add_argument("--output",type=Path,default=DEFAULT_ROOT/"conditional")
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
