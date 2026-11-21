"""Evaluate invariance and natural-vocabulary retrieval on a single Colab GPU."""

from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import inspect
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES, bare_words
from .correlated_lexical_choice import WordChoices
from .cross_diffusion_map import Index, Config
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import load_kd, paired_delta
from .semantic import read_words
from .shared_concept_likelihood import emission


FAMILY = "correlated-lexical-choice"


def stability(reference, changed):
    # Search each index against sorted photo ids in the other top 100.
    ordered = changed.sort(1).values
    where = torch.searchsorted(ordered, reference.contiguous()).clamp_max(ordered.size(1)-1)
    overlap = (ordered.gather(1, where) == reference).float().mean(1)
    top1 = (reference[:, 0] != changed[:, 0]).float()
    return torch.stack((top1, overlap), 1).cpu()


@torch.inference_mode()
def measure(folder, meta, q, p, ql, pl, words, source, target, scale, kd, workers):
    split = meta["split"]
    path, per_query = folder / (split + ".json"), folder / (split + ".pt")
    if path.is_file() and per_query.is_file():
        old = read(path)
        if old["protocol"] == meta and old["per_query_sha256"] == checksum(per_query):
            return
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    config = Config(neighbors=16, bandwidth=.36, iterations=2, mode="rgb")
    graph = Index(SimpleNamespace(p=p), rgb_only=True)
    graph.prepare(config)
    bank = WordChoices(p, source, target, scale)
    if bank.clone_check["different_semantic_winners"]:
        raise ValueError("Exact copies changed the latent concept in paired draws")
    torch.cuda.synchronize()
    index_seconds = time.perf_counter() - start
    print(split, "index", round(index_seconds,1), "seconds, matched temperature multiplier", bank.beta, flush=True)
    replay = Replay(kd, pl, split)
    gl = pl.cuda()
    per_method, errors = {}, {}
    stability_pairs = {"softmax_copies": ("diffusion", "baseline_clones"),
                       "softmax_thinning": ("diffusion", "baseline_thin"),
                       "correlated_thinning": ("correlated", "correlated_thin"),
                       "two_mc_seeds": ("correlated", "second_seed")}
    stability_data = {key: [] for key in stability_pairs}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(0, len(ql), 192):
            y = ql[i:i+192]
            likelihood = emission(q[i:i+192], source, scale)
            top_lists = {}
            for name, lexical in bank.seeds(likelihood):
                scores = graph.score(config, lexical)
                if not torch.isfinite(scores).all():
                    raise ValueError("Nonfinite " + name)
                calc, order, _, _ = map_values(scores, y.cuda(), gl, split)
                top_lists[name] = order[:, :100].contiguous()
                observed = replay.score(order, y, pool)
                error = float((calc.cpu() - observed).abs().max())
                if error > 2e-6:
                    raise ValueError("KD verification failed: " + name)
                errors[name] = max(errors.get(name, 0.), error)
                per_method.setdefault(name, []).append(observed)
            for key,(ref, alt) in stability_pairs.items():
                stability_data[key].append(stability(top_lists[ref], top_lists[alt]))
            if i % 1920 == 0:
                print(split, i+len(y), "/", len(q), flush=True)
    values = {k:torch.cat(v) for k,v in per_method.items()}
    change = {k:torch.cat(v) for k,v in stability_data.items()}
    if any(v.shape != (len(ql),2) for v in values.values()):
        raise ValueError("Incomplete query set")
    save_tensor({"metrics":values,"stability":change,"query_labels":ql},per_query)
    result = {"protocol":meta,"map_k":replay.cutoffs[0],"p_k":replay.cutoffs[1],
              "scores":{k:dict(zip(("mAP","precision"),v.mean(0).tolist())) for k,v in values.items()},
              "paired_vs_diffusion":{k:paired_delta(v,values["diffusion"],ql) for k,v in values.items()},
              "correlated_vs_controls":{k:paired_delta(values["correlated"],values[k],ql) for k in
                                        ("matched_temperature","iid_sample","wrong_geometry","gallery_mle","second_seed")},
              "stability":{k:{"top1_changes":int(v[:,0].sum()),"top100_retained":float(v[:,1].mean())}
                           for k,v in change.items()},
              "thinning": {"pairs":len(bank.pairs),"min_cosine":min((v for _,_,v in bank.pairs), default=None),
                           "examples_by_similarity":[[words[a], words[b], v] for a,b,v in bank.pairs[::max(1,len(bank.pairs)//15)]]},
              "copy_insertion":bank.clone_check,"matched_beta":bank.beta,"collision":bank.collision,
              "index_seconds":index_seconds, "elapsed_seconds":time.perf_counter()-start,
              "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
              "original_kd_queries_per_method":len(ql),"kd_max_abs_error":errors,
              "per_query_sha256":checksum(per_query)}
    atomic_json(path,result)
    print(split,result["scores"],result["stability"],flush=True)


def run(w,workers):
    folder = w/"results"/FAMILY
    kd,_ = load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("KD-SBIR reference changed")
    for dataset in ("sketchy","tuberlin"):
        cfg,_,queries,_,galleries,qraw,praw = global_inputs(w,dataset)
        words = read_words(cfg.vocabulary)
        if len(words)!=20931:
            raise ValueError("Vocabulary changed")
        _,scale = bare_words(cfg,words)
        source,target = (x.cuda() for x in load_words(cfg,words))
        if dataset=="tuberlin":
            lock={"candidate":"correlated","parameters":{"draws":WordChoices.samples,"seed":WordChoices.first_seed},
                  "revision":revision(),"scores_before_transfer":{s:checksum(folder/(s+".json")) for s in ("sketchy_1","sketchy_2")}}
            atomic_json(folder/"freeze.json",lock|{"sha256":digest(lock)})
        for split,(qi,ql) in queries.items():
            pi,pl=galleries[split]
            if (len(qi),len(pi),len(pl.unique()))!=SIZES[split] or digest(UNSEEN_CLASSES[split])!=SPLIT_HASHES[split]:
                raise ValueError("Incomplete split " + split)
            meta={"split":split,"checkpoint":CHECKPOINT,"kd_function_sha256":KD_FUNCTION_SHA,
                  "vocabulary_sha256":checksum(cfg.vocabulary),"revision":revision(),"gpu":torch.cuda.get_device_name(),
                  "queries":len(ql),"gallery":len(pl),"classes":len(pl.unique()),"samples":WordChoices.samples,
                  "primary_seed":WordChoices.first_seed,"secondary_seed":WordChoices.second_seed}
            measure(folder,meta,qraw[qi].cuda(),praw[pi].cuda(),ql,pl,words,source,target,scale,kd,workers)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--workers",type=int,default=min(10,len(os.sched_getaffinity(0))))
    args=parser.parse_args()
    with torch.inference_mode():
        run(args.workspace,args.workers)
