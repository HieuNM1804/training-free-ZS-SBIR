"""Full chance-corrected switch attribution and a single-query exact-index test."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
from statistics import median
import time

import open_clip
import torch

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .counterfactual_naming_index import NamingIndex, switch_scores
from .counterfactual_naming_study import FAMILY, estimate, seed_run
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT, runtime
from .depiction_path_diagnostic import class_interval
from .encoding import load_frozen_clip, save_tensor
from .gallery_convex_support import SEED
from .gallery_convex_support_audit import matched_counts, newcomer_hits, prefix, summarize
from .locked_geometry_words import score_rows
from .naming_entropy_control import audit_entropy, fixture, maximum_entropy
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import BATCH, SKETCHY, Split, mean_percent
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt
from .semantic_boxable_sparse import image_inputs


def accumulated_switches(x, y, method):
    n = x.source.shape[1] if method == "switched" else x.levels.shape[1]
    rows = x.source if method == "switched" else x.from_log
    cols = y.source if method == "switched" else y.from_log
    total = rows.new_zeros((len(rows),len(cols)))
    pairs = torch.triu_indices(n,n,1,device=rows.device)
    left, right = getattr(x,method), getattr(y,method)
    for indices in pairs.split(2048,dim=1):
        u, v = indices
        a = torch.cat((left(u,v),left(v,u)),1)
        b = torch.cat((right(u,v),right(v,u)),1)
        total.addmm_(a,b.T)
        del a,b
    return total


def timed(f):
    torch.cuda.synchronize()
    tick = time.perf_counter()
    value = f()
    torch.cuda.synchronize()
    return value, 1000*(time.perf_counter()-tick)


def index_check(x,y,truth_scores,k):
    index, build = timed(lambda: NamingIndex(y))
    ids = torch.randperm(len(x.source),generator=torch.Generator().manual_seed(SEED))[:64].tolist()
    all_rows = torch.arange(len(y.source), device=y.source.device)
    _ = index.topk(x,ids[0],k)
    pruned = []
    compact_ms, literal_ms, scan_ms = [], [], []
    same_set, same_order = 0,0
    max_semantic_gap, max_dense_discrepancy = 0.,0.
    for t,row in enumerate(ids):
        expected = truth_scores[row].argsort(descending=True,stable=True)[:k]
        (actual, pool), ms = timed(lambda: index.topk(x,row,k))
        compact_ms.append(ms)
        pruned.append(len(pool))
        if not torch.isin(expected,pool).all():
            raise ValueError("A full-score top-K candidate was dropped by the analytic bound")
        same = torch.equal(actual.sort().values,expected.sort().values)
        same_set += int(same)
        same_order += int(torch.equal(actual,expected))
        if not same:
            max_semantic_gap = max(max_semantic_gap,float(
                (truth_scores[row,expected].min()-truth_scores[row,actual].min()).abs()))
        _, ms = timed(lambda: (y.target@x.source[row]).topk(k))
        literal_ms.append(ms)
        if t < 4:
            scan, ms = timed(lambda: y.target@x.source[row] + switch_scores(x,y,row,all_rows))
            scan_ms.append(ms)
            max_dense_discrepancy = max(max_dense_discrepancy,float((scan-truth_scores[row]).abs().max()))
    if max_semantic_gap > 2e-5 or max_dense_discrepancy > 2e-5:
        raise ValueError("Compact ranker differs materially from full benchmark evaluation")
    return {
        "k":k,"query_rows":ids,"gallery":len(y.source),"exact_candidate_counts":pruned,
        "mean_fraction_requiring_transition_score":sum(pruned)/len(ids)/len(y.source),
        "full_reference_topk_survives_bound":len(ids),"same_membership":same_set,"same_complete_order":same_order,
        "max_full_score_gap_at_mismatched_cutoff":max_semantic_gap,"max_dense_vs_compact_error":max_dense_discrepancy,
        "index_setup_ms_excluding_law":build,"index_float32_megabytes":5*y.source.numel()*4/1e6,
        "median_index_ms_64":median(compact_ms),"median_dot_ms_64":median(literal_ms),
        "median_compact_arithmetic_exhaustive_ms_4":median(scan_ms),
    }


def audit(root, split, text, kd, pool, parent_lock, new_lock):
    folder = root/"entropy-audit"
    scorefile = folder/(split.name+".json")
    datafile = scorefile.with_suffix(".pt")
    if scorefile.exists():
        pred = json.loads(scorefile.read_bytes())
        if pred["protocol"] != new_lock or pred["per_query_sha256"] != checksum(datafile):
            raise ValueError("Invalid saved audit")
        return
    parent = json.loads((root/scorefile.name).read_bytes())
    prior_file = root/datafile.name
    if parent["protocol"] != parent_lock or checksum(prior_file) != parent["per_query_sha256"]:
        raise ValueError("Primary archive differs")
    archived = torch.load(prior_file,weights_only=True)
    if any(not torch.equal(archived[k],v) for k,v in (
        ("qi",split.query_ids),("pi",split.gallery_ids),("labels",split.sketch_labels),
    )):
        raise ValueError("Primary sample order changed")
    x,y,qn,pn = estimate(split,text)
    true = accumulated_switches(x,y,"switched")
    nx,ny = maximum_entropy(x),maximum_entropy(y)
    changes = {}
    for name, model, null in (("sketch",x,nx),("photo",y,ny)):
        check_rows = torch.randperm(len(model.source),generator=torch.Generator().manual_seed(SEED))[:128].cuda()
        changes[name] = audit_entropy(model,null,check_rows)
    ref = accumulated_switches(nx,ny,"mass")
    # Match the shape of the primary exhaustive matrix multiply for replay.
    literal = x.source@y.target.T
    true += literal
    ref += literal
    del literal
    replay = PrefixReplay(kd,split.photo_labels,split.name)
    indexing = index_check(x,y,true,replay.cutoffs[1])
    labels, gallery = split.sketch_labels.cuda(), split.photo_labels.cuda()
    rows, pairs, discord = [], [], []
    for start in range(0,len(qn),BATCH):
        sl = slice(start,min(start+BATCH,len(qn)))
        dot = qn[sl]@pn
        original = prefix(dot,*replay.cutoffs)
        inside = torch.zeros_like(dot,dtype=torch.bool).scatter_(1,original,True)
        newcomer_hits(original,inside,gallery,labels[sl],archived["rows"]["dot"][sl,1])
        ordered = prefix(true[sl],*replay.cutoffs)
        main_new = newcomer_hits(ordered,inside,gallery,labels[sl],archived["rows"]["reciprocal"][sl,1])
        val = score_rows(ref[sl],split.sketch_labels[sl],gallery,split.name,replay,pool)
        rows.append(val)
        null_top = prefix(ref[sl],*replay.cutoffs)
        null_new = newcomer_hits(null_top,inside,gallery,labels[sl],val[:,1])
        pairs.append(matched_counts(main_new,null_new))
        # Compare the two outputs directly at equal budget. Most of the retrieved
        # photos are shared when the maximum-entropy plan closely resembles the model.
        in_null = torch.zeros_like(dot,dtype=torch.bool).scatter_(1,null_top,True)
        in_main = torch.zeros_like(dot,dtype=torch.bool).scatter_(1,ordered,True)
        only_main = ~in_null.gather(1,ordered)
        only_null = ~in_main.gather(1,null_top)
        discord.append(torch.stack((only_main.sum(1), (only_main&main_new[1]).sum(1),
                                   (only_null&null_new[1]).sum(1)),1).cpu())
        if start == 0 or sl.stop == len(qn) or (start//BATCH+1)%10 == 0:
            print(split.name,sl.stop,"/",len(qn),"entropy controls",flush=True)
    full_rows, matches, disagreement = torch.cat(rows).double(),torch.cat(pairs),torch.cat(discord)
    fit = archived["rows"]["reciprocal"]
    all_rows = {"entropy":full_rows,"reciprocal":fit,"dot":archived["rows"]["dot"]}
    n, selected, alternative = disagreement.T
    active = n>0
    save_tensor({"qi":split.query_ids,"pi":split.gallery_ids,"labels":split.sketch_labels,"rows":all_rows,
                 "matched":matches,"direct_disagreements":disagreement}, datafile)
    output = {
        "protocol":new_lock,"source_protocol":parent_lock,"source_per_query_sha256":parent["per_query_sha256"],
        "queries":len(qn),"gallery":len(y.source),"classes":len(split.sketch_labels.unique()),"cutoffs":list(replay.cutoffs),
        "metrics":{k:mean_percent(v[:,:2]) for k,v in all_rows.items()},
        "reciprocal_minus_entropy_macro_ap":class_interval(100*(fit[:,0]-full_rows[:,0]),split.sketch_labels,seed=SEED),
        "matched_promotions":summarize(matches,split.sketch_labels),
        "direct_disagreement":{"queries":int(active.sum()),"photos_per_arm":int(n.sum()),
            "primary_correct":int(selected.sum()),"control_correct":int(alternative.sum()),
            "macro_purity_delta":class_interval(100*(selected[active]-alternative[active]).double()/n[active],
                                                split.sketch_labels[active],seed=SEED)},
        "coupling_entropy":changes,"retrieval_index":indexing,"per_query_sha256":checksum(datafile),
    }
    atomic_json(scorefile,output)
    print(split.name,"verified entropy-control",output["metrics"],flush=True)


@torch.inference_mode()
def run(dataset):
    seed_run()
    w = Path("/content/sbir")
    root = w/"results"/FAMILY
    parent = json.loads((root/"freeze.json").read_bytes())
    cfg_words = json.loads((DOCS/VOCABULARY_STUDY/"freeze.json").read_bytes())
    vocab_hash = cfg_words.pop("sha256")
    words = cfg_words["subsets"]["boxable"]["words"]
    if (digest(cfg_words) != vocab_hash or vocab_hash != parent["config"]["vocabulary_sha256"]
        or parent["sha256"] != digest(parent["config"]) or len(words)!=601):
        raise ValueError("Protocol mismatch")
    kd,_ = load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("KD evaluator changed")
    lock = save_lock(root/"entropy-audit"/"freeze.json",{
        "commit":revision(),"source_commit":parent["config"]["commit"],"source_protocol":parent["sha256"],
        "clip_sha256":CHECKPOINT,"evaluator_sha256":KD_FUNCTION_SHA,"vocabulary_sha256":vocab_hash,
        "gpu":torch.cuda.get_device_name(),"torch":str(torch.__version__),"seed":SEED,
        "null":"maximum entropy holding all margins, diagonal and switch directions on each image",
        "fixture":fixture(torch.device("cuda")),
    })
    names = SKETCHY if dataset=="sketchy" else ("tuberlin",)
    workers = min(4,len(os.sched_getaffinity(0)))
    cfg = runtime(w,dataset,workers=workers)
    clip = load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = {s:encode_prompt(clip,tokenizer,words,s) for s in DEFAULT_PROMPTS}
    data = image_inputs(cfg,names,clip)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name in names:
            audit(root,Split.from_inputs(data,name),text,kd,pool,parent["sha256"],lock)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset",choices=("sketchy","tuberlin"))
    run(parser.parse_args().dataset)
