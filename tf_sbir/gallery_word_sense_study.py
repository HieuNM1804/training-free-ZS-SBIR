"""Test whether a single public word can transfer distinct gallery-supported readings."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES, bare_words
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .gallery_word_senses import SenseIndex
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs, stratified_sample
from .retrieval_evidence import load_kd, paired_delta
from .semantic import read_words

FAMILY = "gallery-conditioned-word-senses"
SKETCHY = ("sketchy_1","sketchy_2")


def key(config):
    return json.dumps(config, separators=(",",":"))


def grid():
    return [("phrase", share, temp, mode, mix)
            for share in (0., .5, 1.) for mode in ("conditional","pooled")
            for temp in ((.5, 1., 2.) if mode=="conditional" else (1.,))
            for mix in (.4, .8, 1.)]


def pick(means, mode, count, native=False):
    chosen = [k for k in means if json.loads(k)[3]==mode
              and (not native or json.loads(k)[1]==0.)]
    return [tuple(json.loads(k)) for k in sorted(chosen, key=means.get, reverse=True)[:count]]


def shortlist(means):
    return sorted(set(pick(means,"conditional",3) + pick(means,"pooled",2)
                      + pick(means,"pooled",2,native=True)))


def lock_candidates(means):
    main = pick(means,"conditional",1)[0]
    pooled = pick(means,"pooled",1)[0]
    native = pick(means,"pooled",1,native=True)[0]
    return {"primary": main, "optimized_pooled": pooled, "optimized_native": native}


def controls(lock):
    kind, share, temp, _, mix = lock["primary"]
    return sorted(set(tuple(v) for v in lock.values()) | {
        (kind, share, 1., "pooled", mix), (kind, share, temp, "reverse", mix),
        ("sum",share,temp,"conditional",mix), ("context",share,temp,"conditional",mix),
        (kind,0.,1.,"pooled",mix), ("phrase",0.,1.,"pooled",0.),
    })


def build(w, dataset, split, cfg, words, photos, gallery_ids, source, target, scale, runid):
    cache = w/"cache"/FAMILY
    paths = cache/(split+"-"+runid+".pt")
    notes = paths.with_suffix(".json")
    if paths.exists() and notes.exists():
        state = torch.load(paths, weights_only=True, mmap=True, map_location="cpu")
        index = object.__new__(SenseIndex)
        from .gallery_word_senses import Fold
        index.folds = [Fold(x["ids"].cuda(),x["weights"].cuda(),x["difference"].cuda(),x["fraction"].cuda(),
                            {k:tuple(t.cuda() for t in v) for k,v in x["texts"].items()}) for x in state["folds"]]
        index.scale = scale
        index.source = state["source"].cuda()
        index.photo_count = len(gallery_ids)
        index.diagnostics = read(notes)
        return index
    index = SenseIndex(cfg, words, photos[gallery_ids].cuda(),source,target,scale)
    folds = [{"ids":fold.ids.cpu(),"weights":fold.weights.cpu(), "difference":fold.difference.cpu(),
              "fraction":fold.fraction.cpu(), "texts":{k:[x.cpu() for x in v] for k,v in fold.source_texts.items()}}
             for fold in index.folds]
    save_tensor({"folds":folds,"source":index.source.cpu()},paths)
    atomic_json(notes,index.diagnostics)
    return index


def measure(index, historical, q, y, g, indices, variants, split, reference, workers, primary):
    per = {key(c):[] for c in variants}
    errors = {}
    first = []
    parts = sorted({tuple(c[:4]) for c in variants})
    ctx = ThreadPoolExecutor(max_workers=workers) if reference else nullcontext()
    with ctx as pool:
        for group in indices.split(72):
            qs = q[group].cuda()
            hist = historical[group].cuda().clamp_min(1e-30).log()
            for which in parts:
                compact = index.raw_scores(qs, *which).log()
                for cell in (c for c in variants if tuple(c[:4])==which):
                    name = key(cell)
                    mix = cell[4]
                    est,order,_,_ = map_values((1-mix)*hist+mix*compact,y[group].cuda(),g.cuda(),split)
                    if reference:
                        obs = reference.score(order,y[group],pool)
                        err = float((est.cpu()-obs).abs().max())
                        if err > 2e-6:raise ValueError("Official metric replay differs")
                        errors[name] = max(errors.get(name,0.),err)
                        if name == primary:first.append(order[:,:25].cpu().int())
                    else:
                        obs = est.cpu()
                    per[name].append(obs)
    return {k:torch.cat(v) for k,v in per.items()}, errors, torch.cat(first) if first else None


@torch.inference_mode()
def run(w, phase, workers):
    folder = w/"results"/FAMILY
    current = revision()
    protocol = {"primary":"candidate-dependent contextual reading of a shared word",
                "revision":current,"grid":grid(),"pilot_queries_per_class":40,"seed":39109,
                "compact_rgb_words":64,"selection":"mean official mAP across full Sketchy splits, then mean precision"}
    signature = digest(protocol)
    if (folder/"protocol.json").exists():
        if read(folder/"protocol.json")["sha256"]!=signature:raise ValueError("Different source or protocol in output")
    else:
        atomic_json(folder/"protocol.json",protocol|{"sha256":signature})
    kd,_ = load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Original KD function checksum mismatch")
    stages = {"pilot":("pilot",), "lock":("shortlist",), "audit":("final",),
              "all":("pilot","shortlist","final")}[phase]
    for stage in stages:
        if stage=="pilot":
            variants=grid()
        elif stage=="shortlist":
            variants=shortlist(read(folder/"pilot.json")["means"])
        else:
            lock=read(folder/"freeze.json")
            if lock["full_sketchy_sha256"]!=checksum(folder/"shortlist.json") or lock["revision"]!=current:
                raise ValueError("Incomplete or changed full Sketchy freeze")
            chosen=lock["selection"]
            variants=controls(chosen)
        table={}
        for ds in (("sketchy","tuberlin") if stage=="final" else ("sketchy",)):
            cfg,_,qs,paths,gs,qall,pall=global_inputs(w,ds)
            words=read_words(cfg.vocabulary)
            if len(words)!=20931:raise ValueError("Wrong vocabulary")
            _,scale=bare_words(cfg,words)
            source,target=(x.cuda() for x in load_words(cfg,words))
            for split,(qi,y) in qs.items():
                pi,g=gs[split]
                if (len(qi),len(pi),len(g.unique()))!=SIZES[split] or digest(UNSEEN_CLASSES[split])!=SPLIT_HASHES[split]:
                    raise ValueError("Wrong evaluation split")
                out=folder/(split+".json" if stage=="final" else stage+"-"+split+".json")
                if out.exists():
                    j=read(out)
                    if j["run_signature"]!=signature:raise ValueError("Checkpoint identity changed")
                    if stage=="final" and checksum(folder/(split+".pt"))!=j["per_query_sha256"]:
                        raise ValueError("Incomplete query metrics")
                    table[split]=j["values"]
                    continue
                ident=digest({"protocol":signature,"paths":digest([str(paths[i]) for i in pi.tolist()]),
                              "checkpoint":CHECKPOINT,"vocabulary":checksum(cfg.vocabulary)})[:16]
                index=build(w,ds,split,cfg,words,pall,pi,source,target,scale,ident)
                historical=torch.load(w/"cache/pragmatic-gallery-competition"/(split+"-named-rgb16.pt"),
                                      weights_only=True,mmap=True,map_location="cpu")
                if historical.shape!=(len(qi),len(pi)):raise ValueError("Incomplete historical gallery")
                ids=stratified_sample(y,40,39109) if stage=="pilot" else torch.arange(len(qi))
                primary=key(chosen["primary"]) if stage=="final" else None
                rows,errors,top=measure(index,historical,qall[qi],y,g,ids,variants,split,
                                        Replay(kd,g,split) if stage=="final" else None,workers,primary)
                vals={k:v.mean(0).tolist() for k,v in rows.items()}
                rec={"stage":stage,"run_signature":signature,"values":vals,"diagnostics":index.diagnostics,
                     "queries":len(ids),"gallery":len(pi),"classes":len(g.unique())}
                if stage=="final":
                    path=folder/(split+".pt")
                    save_tensor({"metrics":rows,"query_labels":y,"top25":top},path)
                    rec.update({"protocol":{"split":split,"queries":len(qi),"gallery":len(g),"classes":len(g.unique()),
                                          "checkpoint":CHECKPOINT,"vocabulary_sha256":checksum(cfg.vocabulary),
                                          "gpu":torch.cuda.get_device_name(),"revision":current,"kd_sha256":KD_FUNCTION_SHA,
                                          "map_k":200 if split=="sketchy_2" else 0,"p_k":200 if split=="sketchy_2" else 100},
                                "scores":{k:dict(zip(("mAP","precision"),v)) for k,v in vals.items()},
                                **{name:key(c) for name,c in chosen.items()},
                                "vs_controls":{k:paired_delta(rows[primary],v,y) for k,v in rows.items()},
                                "original_kd_queries_per_condition":len(qi),"kd_max_abs_error":errors,
                                "per_query_sha256":checksum(path)})
                atomic_json(out,rec)
                table[split]=vals
                print(stage,split,"winner",max(vals,key=vals.get),max(vals.values()),flush=True)
        if stage!="final":
            means={k:[sum(table[s][k][i] for s in SKETCHY)/2 for i in (0,1)] for k in table[SKETCHY[0]]}
            atomic_json(folder/(stage+".json"),{"means":means,"splits":table})
            if stage=="shortlist":
                obj={"selection":lock_candidates(means),"revision":current,"full_sketchy_sha256":checksum(folder/"shortlist.json")}
                atomic_json(folder/"freeze.json",obj|{"sha256":digest(obj)})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--phase",choices=("pilot","lock","audit","all"),default="all")
    parser.add_argument("--workers",type=int,default=min(10,len(os.sched_getaffinity(0))))
    args=parser.parse_args()
    run(args.workspace,args.phase,args.workers)
