"""Measure whether population signatures resolve low-probability identifying nouns."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from .concept_benchmark import atomic_json, checksum
from .conditional_completion_study import context, evaluate
from .crossdomain_inputs import CHECKPOINT
from .depiction_path_diagnostic import class_interval
from .encoding import save_tensor
from .gallery_diffusion import nearest_gallery
from .kaggle_unseen_features import DEFAULT_CACHE, SEED, load_split
from .lexical_population import LANDMARKS, aggregate_native, codes, signature, split_visual_experiment, toy
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .replicated_concept_simplex import select_anchors
from .replicated_simplex_study import DEFAULT_ROOT, evaluator, read
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import SKETCHY, mean_percent


COUNTS = (8, 16, 32, 64, 128, 256)
SIZES = (12, 16, 24, 32, 48, 64, 96, 128)
ARMS = ("replicated","frequency","single_reader","raw","donor")


def inputs(args):
    split,text,words=load_split(args.cache,args.split)
    kd=evaluator(args.evaluator)
    q,p,photos,t,replay=context(split,text,kd)
    source=(split.scale*(photos @ text[DEFAULT_PROMPTS[0]].T)).softmax(1)
    draws=split_visual_experiment(photos,max(COUNTS),SEED)
    ids,_=nearest_gallery(photos,max(COUNTS))
    return split,words,q,p,photos,replay,(source,p),draws,ids


def anchors(pop):
    return {kind:select_anchors(pop.obs,max(SIZES),selector,SEED) for kind,selector in (
        ("replicated","replicated"),("single_reader","unreplicated"),("frequency","frequency"))}


def protocol(args):
    spec={"commit":args.revision,"seed":SEED,"counts":COUNTS,"sizes":SIZES,"probe_images":LANDMARKS,
          "prompts":DEFAULT_PROMPTS,"checkpoint":CHECKPOINT,"evaluator":KD_FUNCTION_SHA,"toy":toy(),
          "input_manifest":checksum(args.cache/"manifest.json"),"lexical_bank":checksum(args.cache/"text.pt"),
          "selection":"Maximize equal-weight full Sketchy mAP, then mean precision. TU only after freeze.",
          "hypothesis":"One conditional mean loses multiple visual modes and outside-simplex nouns need not encode a known subject. Recover latent mixture masses from both probe populations, then invert anchor reporting rates.",
          "arms":ARMS}
    save_lock(args.output/"protocol.json",spec)


def sweep(args):
    split,_,q,p,_,replay,readers,(_,draws),neighbors=inputs(args)
    sig=read(args.output/"protocol.json")["sha256"]
    table={}
    for k in COUNTS[args.part::args.parts]:
        pop=signature(readers,draws,k)
        selected=anchors(pop)
        gallery=aggregate_native(p,neighbors,k)
        table[f"donor:{k}"]=mean_percent(evaluate(split,q,gallery,replay))
        for kind,seq in selected.items():
            for size in SIZES:
                if size>len(seq):
                    continue
                ids=seq[:size]
                source,target,_=codes(pop,q,gallery,ids)
                table[f"{kind}:{k}:{size}"]=mean_percent(evaluate(split,source,target,replay))
                if kind=="replicated":
                    source=F.normalize(q[:,ids],p=1,dim=1)
                    target=F.normalize(gallery[:,ids],p=1,dim=1)
                    table[f"raw:{k}:{size}"]=mean_percent(evaluate(split,source,target,replay))
        print(split.name,k,{a:max((v[0],key) for key,v in table.items() if key.startswith(a+":"+str(k)+":"))
                            for a in ARMS[:-1]},flush=True)
        atomic_json(args.output/f"progress-{split.name}-{args.part}.json",{"metrics":table,"protocol":sig})
    atomic_json(args.output/f"grid-{split.name}-{args.part}.json",{"metrics":table,"protocol":sig,
                "part":args.part,"parts":args.parts,"queries":len(q),"gallery":len(p)})


def lock(args):
    root=args.output
    signature=read(root/"protocol.json")["sha256"]
    grids,hashes=[],{}
    for name in SKETCHY:
        combined={}
        for part in range(args.parts):
            p=root/f"grid-{name}-{part}.json"
            blob=read(p)
            if blob["protocol"]!=signature or blob["parts"]!=args.parts or combined.keys() & blob["metrics"].keys():
                raise ValueError("Incompatible grids")
            combined.update(blob["metrics"])
            hashes[p.name]=checksum(p)
        grids.append(combined)
    keys=grids[0].keys() & grids[1].keys()
    def score(key):
        return tuple(sum(tab[key][col] for tab in grids)/2 for col in (0,1))
    best={arm:max(sorted(k for k in keys if k.startswith(arm+":")),key=score) for arm in ARMS}
    selected={arm:[grid[key] for grid in grids] for arm,key in best.items()}
    save_lock(root/"freeze.json",{"commit":args.revision,"protocol":signature,"grids":hashes,"best":best,
                                  "metrics_selected_on_sketchy":selected})
    print("FROZEN",best,selected,flush=True)


def name_diagnostics(source, corrected, target, ids, split, rates, prior):
    # All calculations here occur after retrieval. Class IDs are never inputs to the index.
    gl=F.one_hot(split.photo_labels.cuda()).float()
    ql=F.one_hot(split.sketch_labels.cuda(),num_classes=gl.size(1)).float()
    counts=target[:,ids].T @ gl
    purity,category=(counts/counts.sum(1,keepdim=True)).max(1)
    class_means=(ql.T @ source[:,ids])/ql.sum(0)[:,None]
    index=torch.arange(len(ids),device=ids.device)
    observed=class_means[category,index]
    expected=rates[ids]/prior.clamp_min(1e-20)
    before=category[source[:,ids].argmax(1)]==split.sketch_labels.cuda()
    after=category[corrected.argmax(1)]==split.sketch_labels.cuda()
    finite=(expected>1e-10)&(observed>1e-10)
    correlation=None
    if finite.sum()>2:
        correlation=torch.corrcoef(torch.stack((expected[finite].log(),observed[finite].log())))[0,1].item()
    return {"anchor_purity_avg":purity.mean().item(),"above_80pct_purity":int((purity>.8).sum()),
            "distinct_dominant_classes":int(category.unique().numel()),"query_top1_fixed":int((~before&after).sum()),
            "query_top1_damaged":int((before&~after).sum()),"log_reporting_rate_corr":correlation,
            "rows":[{"purity":float(purity[i]),"class":int(category[i]),"sketch_measured":float(observed[i]),
                     "predicted":float(expected[i])} for i in range(len(ids))]}


def official(args):
    freeze=read(args.output/"freeze.json")
    if freeze["config"]["commit"]!=args.revision:
        raise ValueError("Frozen code mismatch")
    best=freeze["config"]["best"]
    split,words,q,p,_,replay,readers,(_,draws),neighbors=inputs(args)
    _,ktext,stext=best["replicated"].split(":")
    k,size=int(ktext),int(stext)
    rows,summary={},{}
    settings={int(key.split(":")[1]) for key in best.values()}
    def score(name,x,y,pool):
        rows[name]=evaluate(split,x,y,replay,pool)
        print(split.name,name,mean_percent(rows[name]),flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        score("dot",q,p,pool)
        for count in sorted(settings):
            pop=signature(readers,draws,count)
            sequences=anchors(pop)
            gallery=aggregate_native(p,neighbors,count)
            for arm in ARMS:
                parts=best[arm].split(":")
                if count!=int(parts[1]):
                    continue
                if arm=="donor":
                    score("donor_tuned",q,gallery,pool)
                    continue
                n=int(parts[2])
                seq=sequences["replicated" if arm=="raw" else arm]
                ids=seq[:n]
                x,y,info=codes(pop,q,gallery,ids,correction="raw" if arm=="raw" else "both")
                score(arm+"_tuned",x,y,pool)
                if arm=="replicated":
                    summary["anchor_words"]=[words[i] for i in ids.tolist()]
                    summary["fit"]=info
                    summary["proxy_to_sketch"]=name_diagnostics(q,x,p,ids,split,pop.rates[0],q.new_tensor(info["prior"]))
            if count!=k:
                continue
            for arm in ("replicated","frequency","single_reader","random"):
                seq=select_anchors(pop.obs,size,"random",SEED) if arm=="random" else sequences[arm]
                ids=seq[:size]
                x,y,_=codes(pop,q,gallery,ids)
                if best.get(arm)==f"{arm}:{k}:{size}":
                    rows[arm]=rows[arm+"_tuned"]
                else:
                    score(arm,x,y,pool)
            ids=sequences["replicated"][:size]
            for correction in ("raw","target_only","source_only","target_fit","flat_prior"):
                x,y,_=codes(pop,q,gallery,ids,correction=correction)
                score(correction,x,y,pool)
            x,y,_=codes(pop,q,p,ids)
            score("native_target_readout",x,y,pool)
            all_ids=torch.arange(q.size(1),device=q.device)
            x,y,_=codes(pop,q,gallery,all_ids)
            score("all_601",x,y,pool)
            score("donor_matched_k",q,gallery,pool)
    if split.name in SKETCHY:
        j=SKETCHY.index(split.name)
        for arm in ARMS:
            scan=freeze["config"]["metrics_selected_on_sketchy"][arm][j]
            replayed=mean_percent(rows[arm+"_tuned"])
            if max(abs(a-b) for a,b in zip(scan,replayed,strict=True))>.0003:
                raise ValueError(f"Original KD disagrees with scan for {arm}")
    target=args.output/f"{split.name}.pt"
    save_tensor({"qi":split.query_ids,"pi":split.gallery_ids,"labels":split.sketch_labels,"rows":rows},target)
    atomic_json(target.with_suffix(".json"),{"commit":args.revision,"selection":freeze["sha256"],
        "queries":len(q),"gallery":len(p),"classes":len(split.sketch_labels.unique()),"cutoffs":replay.cutoffs,
        "metrics":{n:mean_percent(v) for n,v in rows.items()},"mechanism":summary,
        "primary_minus_control_class_macro_ap":{n:class_interval(100*(rows["replicated"][:,0]-v[:,0]),
                                  split.sketch_labels,seed=SEED) for n,v in rows.items()},
        "per_query_sha256":checksum(target)})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("protocol","sweep","lock","official"))
    parser.add_argument("--split",choices=(*SKETCHY,"tuberlin"))
    parser.add_argument("--revision",required=True)
    parser.add_argument("--cache",type=Path,default=DEFAULT_CACHE)
    parser.add_argument("--evaluator",type=Path,default=DEFAULT_CACHE.parent/"real-sketchy-smoke/evaluator.py")
    parser.add_argument("--output",type=Path,default=DEFAULT_ROOT/"populations")
    parser.add_argument("--part",type=int,default=0)
    parser.add_argument("--parts",type=int,default=2)
    args=parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    if args.stage=="sweep" and args.split not in SKETCHY:
        parser.error("Select only on Sketchy")
    {"protocol":protocol,"sweep":sweep,"lock":lock,"official":official}[args.stage](args)
