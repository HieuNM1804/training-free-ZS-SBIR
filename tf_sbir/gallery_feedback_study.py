"""Full frozen CLIP-B/32 evaluation of lexical feedback and explanations."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import os
from pathlib import Path
import time

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES, bare_words
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .gallery_lexical_feedback import LexicalFeedback
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import load_kd, paired_delta
from .semantic import read_words
from .shared_concept_likelihood import emission


FAMILY = "gallery-lexical-feedback"
CANDIDATES = ("inverse_2", "inverse_4", "distinct_witness")
CONTROLS = ("factorized", "shared_word", "independent_words", "translated",
            "sharp_words", "shuffled_word_response", "inverse_residual_only")


@torch.inference_mode()
def measure(folder, meta, q, p, ql, pl, source, target, scale, kd, workers):
    split = meta["split"]
    path = folder / (split + ".json")
    details = folder / (split + ".pt")
    if path.is_file() and details.is_file():
        old = read(path)
        if old["protocol"] == meta and old["per_query_sha256"] == checksum(details):
            return
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    index = LexicalFeedback(p, target, scale)
    torch.cuda.synchronize()
    index_seconds = time.perf_counter() - start
    print(split, "index in", round(index_seconds, 1), flush=True)
    replay = Replay(kd, pl, split)
    device_labels = pl.cuda()
    values, best, errors, kl = {}, {}, {}, []
    loo_error = first_pass_error = 0.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(0,len(ql),192):
            labels = ql[i:i+192]
            likelihood = emission(q[i:i+192], source, scale)
            scores, divergence = index.evaluate(likelihood)
            if i == 0:
                ids, explicit = index.validate(likelihood)
                lhs = scores["distinct_witness"][0, ids]
                loo_error = float(((lhs - explicit).abs()/explicit.abs().clamp_min(1e-14)).max())
                if loo_error > 1e-4:
                    raise ValueError("Explicit photo removal disagrees with factorization " + str(loo_error))
            first = scores["diffusion"]
            delta = (first - scores["factorized"]).abs() / first.amax(1, keepdim=True).clamp_min(1e-30)
            first_pass_error = max(first_pass_error,float(delta.max()))
            if (divergence[:,1:] - divergence[:,:-1]).max() > 2e-5:
                raise ValueError("EM increased the query reconstruction divergence")
            kl.append(divergence.cpu())
            for name, score in scores.items():
                if not torch.isfinite(score).all():
                    raise ValueError("Nonfinite " + name)
                fast, order, _, _ = map_values(score, labels.cuda(), device_labels, split)
                observed = replay.score(order, labels, pool)
                error = float((fast.cpu() - observed).abs().max())
                if error > 2e-6:
                    raise ValueError("Original KD metrics mismatch " + name)
                errors[name] = max(errors.get(name,0.),error)
                values.setdefault(name,[]).append(observed)
                best.setdefault(name,[]).append(order[:,0].int().cpu())
            if i % 1920 == 0:
                print(split, i+len(labels), "/", len(q), flush=True)
    values = {k:torch.cat(v) for k,v in values.items()}
    best = {k:torch.cat(v) for k,v in best.items()}
    kl = torch.cat(kl)
    if any(v.shape != (len(q),2) for v in values.values()):
        raise ValueError("Incomplete query set")
    base = values["diffusion"]
    correct = {k:pl[v.long()] == ql for k,v in best.items()}
    save_tensor({"query_labels":ql,"metrics":values,"top1":best,"reconstruction_kl":kl},details)
    result = {"protocol":meta, "map_k":replay.cutoffs[0], "p_k":replay.cutoffs[1],
              "scores":{k:dict(zip(("mAP","precision"),v.mean(0).tolist())) for k,v in values.items()},
              "vs_diffusion":{k:paired_delta(v,base,ql) for k,v in values.items()},
              "vs_controls":{key:{k:paired_delta(values[key],values[k],ql) for k in CONTROLS}
                             for key in CANDIDATES},
              "top1":{k:{"changed":int((v != best["diffusion"]).sum()),
                         "wrong_to_right":int((correct[k] & ~correct["diffusion"]).sum()),
                         "right_to_wrong":int((~correct[k] & correct["diffusion"]).sum())}
                      for k,v in best.items()},
              "kl_by_step":kl.mean(0).tolist(),
              "lower_kl_and_lower_ap_at_step4":int(((kl[:,-1]<kl[:,0]) &
                                                      (values["inverse_4"][:,0] < base[:,0])).sum()),
              "index_mass_max_abs_before_normalization":index.mass_error,
              "factorization_max_abs_as_fraction_of_query_peak":first_pass_error,
              "loo_explicit_max_rel_error":loo_error,
              "kd_queries_per_method":len(q),"kd_max_abs_error":errors,
              "index_seconds":index_seconds,"elapsed_seconds":time.perf_counter()-start,
              "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,"per_query_sha256":checksum(details)}
    atomic_json(path,result)
    print(split,result["scores"],flush=True)


def lock(folder):
    sketchy = {s: read(folder/(s+".json")) for s in ("sketchy_1","sketchy_2")}
    means = {k:[sum(d["scores"][k][metric] for d in sketchy.values())/2
                for metric in ("mAP","precision")] for k in CANDIDATES}
    selected = max(CANDIDATES, key=means.get)
    decision = {"means":means, "selected":selected, "revision":revision(),
                "rule":"mean Sketchy mAP, then precision, between the three prespecified choices",
                "source_sha256":{s:checksum(folder/(s+".json")) for s in sketchy}}
    atomic_json(folder/"freeze.json",decision | {"sha256":digest(decision)})


def run(w, workers):
    folder = w/"results"/FAMILY
    kd,_ = load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("KD evaluator changed")
    for dataset in ("sketchy","tuberlin"):
        if dataset == "tuberlin":
            lock(folder)
        cfg,_,queries,_,galleries,qraw,praw = global_inputs(w,dataset)
        words = read_words(cfg.vocabulary)
        if len(words) != 20931:
            raise ValueError("Vocabulary changed")
        _,scale = bare_words(cfg,words)
        source,target = (v.cuda() for v in load_words(cfg,words))
        for split,(qi,ql) in queries.items():
            pi,pl = galleries[split]
            if (len(ql),len(pl),len(pl.unique()))!=SIZES[split] or digest(UNSEEN_CLASSES[split])!=SPLIT_HASHES[split]:
                raise ValueError("Incomplete split")
            meta = {"split":split,"checkpoint":CHECKPOINT,"kd_function_sha256":KD_FUNCTION_SHA,
                    "vocabulary_sha256":checksum(cfg.vocabulary),"revision":revision(),"gpu":torch.cuda.get_device_name(),
                    "queries":len(ql),"gallery":len(pl),"classes":len(pl.unique())}
            measure(folder,meta,qraw[qi].cuda(),praw[pi].cuda(),ql,pl,source,target,scale,kd,workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--workers",type=int,default=min(10,len(os.sched_getaffinity(0))))
    args = parser.parse_args()
    with torch.inference_mode():
        run(args.workspace,args.workers)
