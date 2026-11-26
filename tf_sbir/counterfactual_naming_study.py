"""Full unseen retrieval and matched-name diagnostics for domain-prompt substitutions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np
import open_clip
import torch

from .benchmark import merge_inputs
from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json, checksum
from .counterfactual_naming import NamingLaw, audit_law, dominance, fixture, law, matrices
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT, runtime
from .depiction_path_diagnostic import class_interval
from .encoding import load_frozen_clip, save_tensor
from .gallery_convex_support import SEED
from .gallery_convex_support_audit import matched_counts, newcomer_hits, prefix, summarize
from .joint_gallery_witnesses import load_reference
from .locked_geometry_words import score_rows
from .query_partition_study import save_lock
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import BATCH, SKETCHY, Split, mean_percent, posteriors
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt
from .semantic_boxable_sparse import image_inputs


FAMILY = "counterfactual-naming"
ARMS = ("reciprocal", "ordered_coupling", "independent_prompt_choices", "source_verifies",
        "target_verifies", "require_both_names_in_all_cases")


def seed_run():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")


def estimate(split, text):
    s, t = DEFAULT_PROMPTS
    qs, ps = posteriors(split, text, [(s,s), (t,t)])
    x = law(qs[s], qs[t])
    y = law(ps[s].T.contiguous(), ps[t].T.contiguous())
    return x, y, qs[s], ps[t]


def original_photos(split, paths):
    lookup = {}
    for j, index in enumerate(split.gallery_ids.tolist()):
        path = Path(paths[1][index])
        key = path.parent.name, path.stem
        if key in lookup:
            raise ValueError("Ambiguous photo reference")
        lookup[key] = j
    indices = []
    for index in split.query_ids.tolist():
        path = Path(paths[0][index])
        stem, separator, suffix = path.stem.rpartition("-")
        indices.append(lookup.get((path.parent.name, stem), -1) if separator and suffix.isdigit() else -1)
    ids = torch.tensor(indices)
    valid = ids >= 0
    if not torch.equal(split.sketch_labels[valid], split.photo_labels[ids[valid]]):
        raise ValueError("Photo reference has the wrong category")
    return ids


def paired_natural_names(split, law_x, law_y, literal, change, placebo, paths):
    source_ids = original_photos(split, paths).to(literal.device)
    head_x = law_x.source.argmax(1)
    head_y = law_y.target.argmax(1)
    cats = split.photo_labels.cuda()
    stats = {kind: [] for kind in ("same_class", "other_class")}
    totals = {"verified": int((source_ids >= 0).sum()), "different_native_names": 0}
    for start in range(0,len(source_ids),BATCH):
        stop = min(start+BATCH, len(source_ids))
        idx = torch.arange(start,stop,device=literal.device)
        sid = source_ids[idx]
        valid = sid >= 0
        idx, sid = idx[valid], sid[valid]
        u, v = head_x[idx], head_y[sid]
        mismatch = u != v
        idx, sid, u, v = (a[mismatch] for a in (idx,sid,u,v))
        totals["different_native_names"] += len(idx)
        if not len(idx):
            continue
        gaps = (literal[idx] - literal[idx,sid,None]).abs()
        same_word = head_y[None, :] == v[:, None]
        same_cat = cats[sid,None] == cats[None, :]
        for kind in stats:
            mask = same_word & (same_cat if kind == "same_class" else ~same_cat)
            mask.scatter_(1, sid[:,None], False)
            ok = mask.any(1)
            if not ok.any():
                continue
            nearest = gaps.masked_fill(~mask, torch.inf).argmin(1)
            i, positive, other, uu, vv = (value[ok] for value in (idx,sid,nearest,u,v))
            source_switch = (law_x.target[i,vv] * (law_x.primitive[i,uu] - law_x.primitive[i,vv]).clamp_min(0))
            def conditional(g):
                return source_switch * law_y.source[g,uu] * (law_y.primitive[g,uu] - law_y.primitive[g,vv]).clamp_min(0)
            own, rival = conditional(positive), conditional(other)
            # Conditional quantities condition on the same two native names for each comparison.
            diffs = torch.stack((own, rival, change[i,positive]-change[i,other],
                                 placebo[i,positive]-placebo[i,other],
                                 (literal[i,positive]-literal[i,other]).abs()), dim=1).double().cpu()
            stats[kind].append((i.cpu(), diffs))
    for kind, blocks in stats.items():
        if not blocks:
            totals[kind] = None
            continue
        ids = torch.cat([a for a,_ in blocks])
        own, other, change_gap, null_gap, base_gap = torch.cat([b for _,b in blocks]).T
        comparison = {}
        for name, values in (("native_name_reciprocal_probability", own-other),
                             ("all_switches", change_gap), ("placebo_switches", null_gap)):
            win = (values > 0).double() + .5*(values == 0).double()
            comparison[name] = {"mean_gap": float(values.mean()), "wins_percent": float(100 * win.mean()),
                                "macro_win_minus_chance": class_interval(100*win-50,split.sketch_labels[ids],seed=SEED)}
        totals[kind] = {"pairs":len(ids), "mean_matched_dot_gap":float(base_gap.mean()),
                        "native_switch_positive_for_true_false":[int((a>0).sum()) for a in (own,other)],
                        "comparison":comparison}
    return totals


def trace_pair(x, y, i, j, words):
    ids = torch.arange(len(words),device=x.source.device)
    u, v = ids.repeat_interleave(len(ids)), ids.repeat(len(ids))
    def mass(l, row):
        return l.source[row,u]*l.target[row,v]*(l.primitive[row,u]-l.primitive[row,v]).clamp_min(0)
    a, b = mass(x,i), mass(y,j)
    product = a*b
    k = int(product.argmax())
    return {"native_dot":float(x.source[i] @ y.target[j]), "total_switch_mass":float(product.sum()),
            "dominant_switched_names":[words[int(u[k])],words[int(v[k])]], "pair_mass":float(product[k]),
            "source_and_target_images_transition_probability":[float(a[k]),float(b[k])]}


def score_methods(root, split, words, text, paths, kd, pool, lock):
    target = root/(split.name+".json")
    artifact = target.with_suffix(".pt")
    if target.exists():
        report = json.loads(target.read_bytes())
        if report["protocol"] != lock or checksum(artifact) != report["per_query_sha256"]:
            raise ValueError("Saved study failed provenance verification")
        return
    baseline, _ = load_reference(split)
    x, y, qnative, pnative = estimate(split, text)
    check = {"sketch":audit_law(x), "photo":audit_law(y)}
    tick = time.perf_counter()
    literal, changed, placebo = matrices(x, y)
    torch.cuda.synchronize()
    matrix_seconds = time.perf_counter()-tick
    rx, cx = x.source-x.stable, x.target-x.stable
    ry, cy = y.source-y.stable, y.target-y.stable
    # Verify the two independent upper bounds without keeping large matrices alive.
    violation = 0.
    for start in range(0,len(x.source),BATCH):
        sl = slice(start,start+BATCH)
        cap = (rx[sl]@ry.T).minimum(cx[sl]@cy.T)
        violation = max(violation,float((changed[sl]-cap).max()))
    if violation > 2e-6 or float((literal+changed).max()) > 1.00001:
        raise ValueError("Invalid agreement probability")
    for i,j in ((0,0),(3,15),(6,101),(19,571)):
        a,b,h = (v[i].double().cpu().numpy() for v in (x.source,x.target,x.primitive))
        c,d,l = (v[j].double().cpu().numpy() for v in (y.source,y.target,y.primitive))
        ref = dominance(a,b,h,c,d,l,np.argsort(h),np.argsort(np.argsort(l)))
        if abs(ref - float(changed[i,j])) > 2e-6:
            raise ValueError("Full law and compact dominance sum disagree")
    natural = paired_natural_names(split,x,y,literal,changed,placebo,paths) if split.name in SKETCHY else None
    replay = PrefixReplay(kd,split.photo_labels,split.name)
    raw = {arm:[] for arm in ARMS}
    matched = {arm:[] for arm in ARMS[1:]}
    top_ids, change_sizes = [], []
    truth, gallery = split.sketch_labels.cuda(), split.photo_labels.cuda()
    for start in range(0,len(qnative),BATCH):
        sl = slice(start,min(start+BATCH,len(qnative)))
        base = qnative[sl] @ pnative
        dot_top = prefix(base,*replay.cutoffs)
        in_base = torch.zeros_like(base,dtype=torch.bool).scatter_(1,dot_top,True)
        _, original_hits = newcomer_hits(dot_top,in_base,gallery,truth[sl],baseline["dot"][sl,1])
        if not torch.equal(original_hits[:,0].cpu(),baseline["dot"][sl,3].bool()):
            raise ValueError("Original Dot rank differs")
        small = literal[sl]
        source_dot = x.source[sl] @ y.source.T
        target_dot = x.target[sl] @ y.target.T
        independent_diagonal = (x.source[sl]*x.target[sl]) @ (y.source*y.target).T
        scores = (small+changed[sl], small+placebo[sl], small+source_dot*target_dot-independent_diagonal,
                  small+cx[sl]@y.target.T, small+x.source[sl]@ry.T,
                  x.stable[sl]@y.stable.T + changed[sl])
        primary = None
        for arm, score in zip(ARMS,scores,strict=True):
            result = score_rows(score,split.sketch_labels[sl],gallery,split.name,replay,pool)
            raw[arm].append(result)
            top = prefix(score,*replay.cutoffs)
            entry = newcomer_hits(top,in_base,gallery,truth[sl],result[:,1])
            if arm == "reciprocal":
                primary = entry
                top_ids.append(torch.stack((dot_top[:,0],top[:,0]),1).cpu())
                change_sizes.append(entry[0].sum(1).cpu())
            else:
                matched[arm].append(matched_counts(primary,entry))
        if start == 0 or sl.stop == len(qnative) or (start//BATCH+1)%10 == 0:
            print(split.name,sl.stop,"/",len(qnative),"queries",flush=True)
    rows = {k:torch.cat(blocks).double() for k,blocks in raw.items()}
    rows["dot"] = baseline["dot"]
    comparisons = {k:torch.cat(blocks) for k,blocks in matched.items()}
    ends = torch.cat(top_ids)
    changes = torch.cat(change_sizes)
    examples = {}
    correct, after = (rows[k][:,3].bool() for k in ("dot","reciprocal"))
    for event,mask in (("repair",~correct & after),("damage",correct & ~after)):
        ids = mask.nonzero().flatten()
        if len(ids):
            i = int(ids[0]); old, new = ends[i].tolist()
            path = Path(paths[0][split.query_ids[i]])
            ex = {"query":str(Path(path.parent.name)/path.name)}
            for name,j in (("dot",old),("reciprocal",new)):
                path = Path(paths[1][split.gallery_ids[j]])
                ex[name] = {"photo":str(Path(path.parent.name)/path.name),**trace_pair(x,y,i,j,words)}
            examples[event] = ex
    save_tensor({"qi":split.query_ids,"pi":split.gallery_ids,"labels":split.sketch_labels,"rows":rows,
                 "matched":comparisons,"prefix_changes":changes},artifact)
    table = {
        "protocol":lock,"queries":len(qnative),"gallery":len(y.source),"classes":len(truth.unique()),
        "cutoffs":list(replay.cutoffs),"metrics":{k:mean_percent(v[:,:2]) for k,v in rows.items()},
        "primary_minus_macro_ap":{k:class_interval(100*(rows["reciprocal"][:,0]-v[:,0]),split.sketch_labels,seed=SEED)
                                 for k,v in rows.items() if k!="reciprocal"},
        "matched_promotions":{k:summarize(v,split.sketch_labels) for k,v in comparisons.items()},
        "fixed_and_damaged_top1":[int((~correct&after).sum()),int((correct&~after).sum())],
        "mean_new_photos_at_cutoff":changes.double().mean().item(),"examples":examples,"natural_name_switches":natural,
        "mean_probability_of_naming_change":[float(t.sum(1).mean()) for t in (rx,ry)],
        "joint_law_checks":check,"max_upper_bound_violation":violation,"exhaustive_matrix_seconds":matrix_seconds,
        "per_query_sha256":checksum(artifact),
    }
    atomic_json(target,table)
    print(split.name,"verified",table["metrics"],flush=True)


@torch.inference_mode()
def run(dataset):
    seed_run()
    w = Path("/content/sbir")
    root = w/"results"/FAMILY
    config = json.loads((DOCS/VOCABULARY_STUDY/"freeze.json").read_bytes())
    vh = config.pop("sha256")
    words = config["subsets"]["boxable"]["words"]
    if digest(config) != vh or len(words) != 601:
        raise ValueError("Unexpected public vocabulary")
    kd,_ = load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("Original KD metric changed")
    lock = save_lock(root/"freeze.json",{
        "commit":revision(),"evaluator_sha256":KD_FUNCTION_SHA,"clip_sha256":CHECKPOINT,"vocabulary_sha256":vh,
        "seed":SEED,"prompts":list(DEFAULT_PROMPTS),"gpu":torch.cuda.get_device_name(),"torch":str(torch.__version__),
        "arms":list(ARMS),"selection":"no tuned coefficients; protocol fixed before running any split",
        "primary":"correct class among equal numbers of newcomers versus same marginal, stable and directional control",
        "natural_switch":"same verified source against a photo with identical dominant native word and nearest literal score",
        "simulation_check":fixture(torch.device("cuda")),
    })
    names = SKETCHY if dataset=="sketchy" else ("tuberlin",)
    workers = min(4,len(os.sched_getaffinity(0)))
    cfg = runtime(w,dataset,workers=workers)
    clip = load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = {s:encode_prompt(clip,tokenizer,words,s) for s in DEFAULT_PROMPTS}
    data = image_inputs(cfg,names,clip)
    paths = [merge_inputs(cfg.root,names,mode)[0] for mode in ("sketch","photo")]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name in names:
            score_methods(root,Split.from_inputs(data,name),words,text,paths,kd,pool,lock)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset",choices=("sketchy","tuberlin"))
    run(parser.parse_args().dataset)
