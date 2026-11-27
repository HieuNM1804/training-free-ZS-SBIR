"""Read the fixed diagnostic cohort and measure cross-depiction identity and category choices."""

import argparse
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from .concept_benchmark import atomic_json, checksum
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .kaggle_unseen_features import DEFAULT_CACHE, index_file, load_split
from .paired_patch_interaction_probe import DST, PARTS, PER_CLASS, SEED, read
from .profiled_study import true_sources
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_disputed_word_risk import query_posteriors, word_bank


@dataclass
class Choices:
    between: torch.Tensor
    within: torch.Tensor


def win(margin):
    return ((margin.sign() + 1.) / 2).float()


def choices(query, photo, pair):
    """Constant-invariant assignments between classes across subjects and within a class across identities."""
    c, n, d = query.shape
    sim = (query.reshape(c*n,d).double() @ photo.reshape(c*n,d).double().T).reshape(c,n,c,n).permute(0,2,1,3)
    ids = torch.arange(c,device=query.device)
    native = sim[ids,ids]
    diagonal = native.diagonal(dim1=-2,dim2=-1)
    within = diagonal[:,:,None] + diagonal[:,None,:] - native - native.transpose(-1,-2)
    index = torch.triu_indices(n,n,offset=1,device=query.device)
    # An independent source of the same class supplies the positive endpoint for category choices.
    other = sim.roll(-n//2, dims=-1)
    main = other[ids,ids].diagonal(dim1=-2,dim2=-1)
    between = main[:,None,:,None] + main[None,:,None,:] - other - other.permute(1,0,3,2)
    return Choices(win(between[pair[0],pair[1]]), win(within[:,index[0],index[1]]))


def verify_choices():
    rng = torch.Generator().manual_seed(SEED)
    a,b = (torch.randn(3,4,8,generator=rng,dtype=torch.float64) for _ in range(2))
    edges = torch.triu_indices(3,3,1)
    cm = choices(a,b,edges)
    shifted = choices(a+13.,b-5.,edges)
    assert torch.equal(cm.between,shifted.between) and torch.equal(cm.within,shifted.within)
    for k,(u,v) in enumerate(edges.T.tolist()):
        for i in range(4):
            for j in range(4):
                margin = (a[u,i]-a[v,j]) @ (b[u,(i+2)%4]-b[v,(j+2)%4])
                assert cm.between[k,i,j].item() == win(margin).item()
    for c in range(3):
        for k,(i,j) in enumerate(torch.triu_indices(4,4,1).T.tolist()):
            assert cm.within[c,k].item() == win((a[c,i]-a[c,j])@(b[c,i]-b[c,j])).item()


def diagnostic_states(root, name):
    by_class = {}
    plan = None
    for part in range(PARTS):
        path = root/f"{name}-{part}.pt"
        record = read(path.with_suffix(".json"))
        if checksum(path) != record["sha256"]:
            raise ValueError("Diagnostic shard hash mismatch")
        if plan is not None and plan != record["plan"]:
            raise ValueError("The two workers used different participants or references")
        plan = record["plan"]
        shard = torch.load(path,weights_only=True)
        paths = [v for cls in shard["classes"] for row in plan["classes"][cls]["records"] for v in row]
        if paths != shard["ordered_paths"]:
            raise ValueError("Cohort order mismatch")
        for i,cls in enumerate(shard["classes"]):
            sl = slice(i*PER_CLASS*3,(i+1)*PER_CLASS*3)
            by_class[cls] = {k:v[sl].reshape(PER_CLASS,3,-1) for k,v in shard["states"].items()}
    ordered = UNSEEN_CLASSES[name]
    if set(by_class) != set(ordered):
        raise ValueError("Missing diagnostic classes")
    states = {k:torch.stack([by_class[cls][k] for cls in ordered]).cuda() for k in by_class[ordered[0]]}
    return plan, states


def cache_controls(name, plan, cache):
    split, text, _ = load_split(cache, name)
    record = read(cache/"manifest.json")["datasets"]["sketchy"]
    idx = torch.load(index_file(cache,"sketchy",record),weights_only=True)
    lut = {m:{idx[m]["paths"][g]:i for i,g in enumerate(ids.tolist())}
           for m,ids in (("sketch",split.query_ids),("photo",split.gallery_ids))}
    subjects = [r for c in UNSEEN_CLASSES[name] for r in plan["classes"][c]["records"]]
    qids = torch.tensor([[lut["sketch"][n] for n in row[1:]] for row in subjects],device="cuda")
    pids = torch.tensor([lut["photo"][row[0]] for row in subjects],device="cuda")
    c = len(UNSEEN_CLASSES[name])
    source_ids = true_sources(split,cache)
    if not torch.equal(source_ids[qids],pids[:,None].expand_as(qids)):
        raise ValueError("Drawings do not refer to the sampled source photo")
    q, mirror = split.sketch[qids.flatten()].float(), split.mirror[qids.flatten()].float()
    p = split.photo[pids].float()
    qs = query_posteriors(q,mirror,text[DEFAULT_PROMPTS[0]],split.scale)
    ps = word_bank(p,text[DEFAULT_PROMPTS[1]],split.scale).T
    native = torch.cat([F.normalize(p,dim=-1)[:,None],F.normalize(q+mirror,dim=-1).reshape(len(p),2,-1)],dim=1)
    semantic = torch.cat([ps[:,None],qs.reshape(len(p),2,-1)],dim=1)
    return split, {"native_clip":native.reshape(c,PER_CLASS,3,-1),"semantic_dot":semantic.reshape(c,PER_CLASS,3,-1)}


def paired_assignments(fields, pairs):
    # Each orientation uses separate reference photos for the query and endpoint.
    first, second = fields
    between, within, photos, redraw = [], [], [], []
    for qbank, pbank in ((first,second),(second,first)):
        draws = [choices(qbank[:,:,v],pbank[:,:,0],pairs) for v in (1,2)]
        between.append(torch.stack([val.between for val in draws]))
        within.append(torch.stack([val.within for val in draws]))
        photos.append(choices(qbank[:,:,0],pbank[:,:,0],pairs).between)
        redraw.append(choices(qbank[:,:,1],pbank[:,:,2],pairs).within)
    return {"between":torch.stack(between), "within":torch.stack(within),
            "photo_between":torch.stack(photos), "redraw_within":torch.stack(redraw)}


def average_by_unit(matrix, bin_dim):
    axes = tuple(i for i in range(matrix.ndim) if i != bin_dim)
    return matrix.mean(dim=axes).double().cpu().numpy()


def class_bootstrap(values, counts, class_edges=None):
    if class_edges is not None:
        a,b = class_edges
        weights = counts[:,a]*counts[:,b]
    else:
        weights = counts
    totals = weights.sum(1)
    at = totals > 0
    distribution = ((weights[at] @ values) / totals[at]) * 100
    return {"difference_points":float(values.mean()*100),
            "bootstrap95":np.quantile(distribution,[.025,.975]).tolist(),
            "units_positive":int((values>0).sum()), "units":len(values)}


def redraw_disagreement(split, reference_dir, cache):
    archive = reference_dir/"paired-L4.zip"
    with zipfile.ZipFile(archive) as z:
        raw = z.read(split.name+".pt")
    if hashlib.sha256(raw).hexdigest() != read(reference_dir/f"{split.name}.json")["per_query_sha256"]:
        raise ValueError("Official semantic reference is not the archived evaluator output")
    data = torch.load(io.BytesIO(raw),weights_only=True)
    if not all(torch.equal(data[k],v) for k,v in
               (("qi",split.query_ids),("pi",split.gallery_ids),("labels",split.sketch_labels))):
        raise ValueError("Official reference IDs differ from the restored cache")
    scores = data["rows"]["dot"][:,0].double().numpy()
    source = true_sources(split,cache).cpu().numpy()
    if (source<0).any():
        raise ValueError("Cannot attribute every query to its source photo")
    label = split.sketch_labels.numpy()
    groups = {}
    by_class = {}
    for i,k in enumerate(source):
        groups.setdefault(int(k),[]).append(i)
        by_class.setdefault(int(label[i]),set()).add(int(k))
    rng = np.random.default_rng(SEED)
    summaries = []
    for cls in sorted(by_class):
        ids = np.array(sorted(by_class[cls]))
        ids = ids[rng.permutation(len(ids))]
        for j,k in enumerate(ids):
            ss = scores[groups[k]]
            if len(ss)<2:
                continue
            u,v = np.triu_indices(len(ss),1)
            inside = np.abs(ss[u]-ss[v])
            other = scores[groups[ids[(j+1)%len(ids)]]]
            outside = np.abs(ss[:,None]-other[None,:])
            summaries.append([inside.mean(),(inside>=.20).mean(),outside.mean(),(outside>=.20).mean()])
    means = np.asarray(summaries).mean(axis=0)
    return {"queries":len(scores),"sources":len(groups),"sources_with_repeat":len(summaries),
            "mean_abs_same_source_ap_difference_points":float(100*means[0]),
            "within_source_fraction_over_20_points":float(means[1]),
            "mean_abs_different_source_same_class_ap_difference_points":float(100*means[2]),
            "different_source_fraction_over_20_points":float(means[3]),
            "weighting":"Equal source weight; all pairs of redraws; another seed-42 source of the same class is the control."}


@torch.inference_mode()
def analyze(args):
    verify_choices()
    labels = UNSEEN_CLASSES[args.split]
    c = len(labels)
    pair = torch.triu_indices(c,c,offset=1,device="cuda")
    edges = pair.cpu().numpy()
    plan, states = diagnostic_states(args.output,args.split)
    split, controls = cache_controls(args.split,plan,args.cache)
    full = states["full"]
    interactions = (states["interaction_0"],states["interaction_1"])
    arms = {"full_linear_readout":(full,full),"interaction":interactions,
            "additive":tuple(full-v for v in interactions), **{k:(v,v) for k,v in controls.items()}}
    outcomes = {k:paired_assignments(pair_fields,pair) for k,pair_fields in arms.items()}
    ids = set(UNSEEN_CLASSES["sketchy_1"]) & set(UNSEEN_CLASSES["sketchy_2"])
    unique = np.array([label not in ids for label in labels])
    unique_pairs = unique[edges[0]] & unique[edges[1]]
    nmask = torch.from_numpy(unique_pairs).cuda()
    rates = {}
    for name,values in outcomes.items():
        rates[name] = {k:float(v.double().mean().item()*100) for k,v in values.items()}
        between = values["between"]
        rates[name]["between_by_panel_orientation"] = (between.double().mean(dim=(1,2,3,4))*100).tolist()
        rates[name]["between_without_shared_classes"] = 100*between[:,:,nmask].double().mean().item()
    random = np.random.default_rng(SEED)
    counts = random.multinomial(c,np.full(c,1/c),size=10000)
    ci = {}
    for reference in ("additive","full_linear_readout","native_clip","semantic_dot"):
        result = {}
        for key in ("between","within"):
            delta = outcomes["interaction"][key] - outcomes[reference][key]
            result[key] = class_bootstrap(average_by_unit(delta,2), counts, edges if key=="between" else None)
        ci[reference] = result
    inter = outcomes["interaction"]["between"]
    base = outcomes["semantic_dot"]["between"]
    changes = inter-base
    total = changes.numel()
    errors = {"gained_fraction":float(changes.clamp_min(0).double().sum().item()/total),
              "lost_fraction":float((-changes).clamp_min(0).double().sum().item()/total),
              "net_points":100*changes.double().mean().item(),
              "note":"Fractions of all category assignments; redraws and reference panels repeat the same subjects."}
    unshared_effect = rates["interaction"]["between_without_shared_classes"] - rates["additive"]["between_without_shared_classes"]
    gate = (ci["additive"]["between"]["bootstrap95"][0]>0 and errors["net_points"]>0 and unshared_effect>0)
    result = {"evaluation_commit":args.revision,"feature_commit":read(args.output/f"{args.split}-0.json")["revision"],
              "seed":SEED,"split":args.split,"classes":list(labels),"shared_classes":sorted(ids),"subjects":c*PER_CLASS,
              "drawings":2*c*PER_CLASS,"class_pairs":len(unique_pairs),"classes_outside_overlap":int(unique.sum()),
              "accuracy_percent":rates,"interaction_minus_control":ci,"changes_vs_semantic":errors,
              "full_semantic_ap_redraw_variation":redraw_disagreement(split,args.reference,args.cache),
              "passes_preregistered_probe_split":bool(gate),
              "shard_sha256":{str(i):checksum(args.output/f"{args.split}-{i}.pt") for i in range(PARTS)}}
    atomic_json(args.output/f"diagnosis-{args.split}.json",result)
    print(args.split, {k:[round(v["between"],2),round(v["within"],2)] for k,v in rates.items()},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",choices=SKETCHY_SPLITS,required=True)
    parser.add_argument("--output",type=Path,default=DST)
    parser.add_argument("--cache",type=Path,default=DEFAULT_CACHE)
    parser.add_argument("--reference",type=Path,
                        default=Path(__file__).resolve().parents[1]/"docs/results/exchangeable-concept-observations")
    parser.add_argument("--revision",required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("highest")
    analyze(args)
