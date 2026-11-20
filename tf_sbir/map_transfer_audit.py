"""Audit mAP locks, paired scores and gallery perturbations on the Colab runtime."""

import argparse
import csv
from importlib import util
from pathlib import Path
import subprocess
import sys

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import SIZES, bare_words
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .foundation import load_words
from .map_diagnosis import by_rank, map_values
from .map_study import Data, score_key
from .retrieval_diagnosis import global_inputs, stratified_sample
from .retrieval_evidence import paired_delta, tensor
from .reverse_prompt import Setting
from .semantic import read_words
from .shared_concept_likelihood import FAMILY, PRIMARY, fit_prior, posterior


def verify_decision(root):
    decision = read(root / "map-study/joint-freeze.json")
    if digest({k:v for k,v in decision.items() if k != "sha256"}) != decision["sha256"]:
        raise ValueError("Corrupt joint file")
    studies = decision["families"]
    if decision["selection_data"] != list(SKETCHY_SPLITS) or decision["primary"] != max(studies, key=lambda n: score_key(studies[n]["scores"])):
        raise ValueError("Method was not selected by Sketchy mAP")
    locks = {}
    for name, entry in studies.items():
        for s, recorded in zip(SKETCHY_SPLITS, entry["scores"], strict=True):
            path = root/name/"sketchy"/f"{s}-{entry['name']}.json"
            row = read(path)
            if checksum(path) != recorded["sha256"] or any(row[k] != recorded[k] for k in ("mAP","precision")):
                raise ValueError("Changed selection scores")
        if entry["historical"]:
            continue
        lock = read(root/name/"freeze.json")
        if (lock["sha256"] != entry["freeze"] or lock["sha256"] != digest({k:v for k,v in lock.items() if k != "sha256"})
                or lock["tu_rows_before_freeze"] != 0 or lock["utc"] > decision["utc"]):
            raise ValueError("Invalid prospective family decision")
        if any(checksum(root/name/"sketchy"/file) != h for file,h in lock["sources"].items()):
            raise ValueError("Changed source of a family lock")
        locks[name] = lock
    return decision, locks


def import_measured(workspace, name, lock):
    directory = workspace/"worktrees"/("map-"+lock["revision"][:12])
    if subprocess.check_output(["git","-C",str(directory),"rev-parse","HEAD"],text=True).strip() != lock["revision"]:
        raise ValueError("Expected measured method worktree")
    stem = name.replace("-", "_")
    spec = util.spec_from_file_location("tf_sbir."+stem, directory/"tf_sbir"/(stem+".py"))
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_result(root, family, split, cid, ql, pl, lock=None):
    location = root/family/("heldout" if split == "tuberlin" else "sketchy")
    file = location/f"{split}-{cid}.json"
    row = read(file)
    details = tensor(location/"per-query"/f"{split}-{cid}.pt")
    scope = (200,200) if split=="sketchy_2" else (0,100)
    if ((row["queries"],row["gallery"],row["classes"]) != SIZES[split] or (row["map_k"],row["p_k"]) != scope
            or (len(ql),len(pl),len(ql.unique())) != SIZES[split] or row["checkpoint_sha256"]!=CHECKPOINT
            or row.get("dataset_sha256",row.get("split_sha256"))!=SPLIT_HASHES[split]
            or not torch.isfinite(details).all() or details.shape != (len(ql),2)
            or max(abs(float(details[:,i].mean())-row[k]) for i,k in enumerate(("mAP","precision")))>1e-6):
        raise ValueError(f"Invalid official or per-query results: {file}")
    if lock is not None:
        if (row["revision"]!=lock["revision"] or (split=="tuberlin" and row["freeze_sha256"]!=lock["sha256"])
                or (cid==lock["winner_id"] and (row["kd_check"]["queries"]!=len(ql)
                    or max(row["kd_check"]["max_per_query_abs_AP_P"])!=0.))):
            raise ValueError(f"Failed independent official replay: {file}")
    return {k:row[k] for k in ("mAP","precision","map_k","p_k","revision")}|{"sha256":checksum(file)}, details


def on_subset(details, mask):
    return {"mAP":float(details[mask,0].mean()),"precision":float(details[mask,1].mean())}


def compare(after, before, labels):
    diff = (after[:,0]-before[:,0])*100
    return paired_delta(after,before,labels)|{"queries_gain_1pp":int((diff>1.).sum()),
        "queries_lose_1pp":int((diff< -1.).sum()),"median_delta_AP_pp":float(diff.median())}


def score_sample(data, module, cfg):
    if module is None:
        priors,_ = fit_prior(data.a)
        return data.query_emission @ posterior(data.a,priors["mle"]).T
    return module.Index(data).score(cfg)


def short(path):
    return "/".join(Path(path).parts[-3:])


def examples(base, score, labels, pl, qpaths, ppaths, split):
    rows, original, old_good, _ = map_values(base,labels,pl,split)
    after, current, new_good, _ = map_values(score,labels,pl,split)
    changes = after[:,0]-rows[:,0]
    records = []
    for event, mask in (("rescued",~old_good[:,0]&new_good[:,0]),("regressed",old_good[:,0]&~new_good[:,0])):
        ids = mask.nonzero().flatten()
        if len(ids):
            j = int(ids[changes[ids].argsort()[len(ids)//2]])
            i, k = int(original[j,0]), int(current[j,0])
            records.append({"case":event,"query":short(qpaths[j]),"uniform_top1":short(ppaths[i]),
                "method_top1":short(ppaths[k]),"method_top1_rank_before":int((original[j]==k).nonzero()[0])+1,
                "uniform_AP":float(rows[j,0]),"method_AP":float(after[j,0]),"cases_in_sample":len(ids)})
    return records


def rank_audit(inputs, split, module, cfg, official, qpaths, ppaths):
    q, p, ql, pl, source, target, scale = inputs
    choice = stratified_sample(ql,48,55429)
    qs, y, z = q[choice].cuda(), ql[choice].cuda(), pl.cuda()
    data = Data(qs,p,source,target,scale)
    primary = score_sample(data,module,cfg)
    uniform = data.base
    native = qs @ p.T
    estimates = {"native":by_rank(native,y,z,split), "uniform":by_rank(uniform,y,z,split),
                 "primary":by_rank(primary,y,z,split)}
    sample_details = map_values(primary,y,z,split)[0].cpu()
    examples_kept = examples(uniform,primary,y,z,[qpaths[j] for j in choice.tolist()],ppaths,split)
    del data
    single = score_sample(Data(qs[:1],p,source,target,scale),module,cfg)
    top_a = single[0].topk(100).indices
    top_b = primary[0].topk(100).indices
    invariance = {"max_absolute_score_difference":float((single[0]-primary[0]).abs().max()),
                  "common_top100":int(torch.isin(top_a,top_b).sum()),"same_top1":bool(top_a[0]==top_b[0])}
    perm = torch.randperm(len(p),generator=torch.Generator().manual_seed(29173))
    halves = []
    for part in perm.tensor_split(2):
        sub = part.cuda()
        partial = Data(qs,p[sub],source,target,scale)
        res = score_sample(partial,module,cfg)
        rows = {name:map_values(v,y,z[sub],split)[0].mean(0).tolist() for name,v in (
            ("reindexed",res),("full_index_restricted",primary[:,sub]),("uniform",partial.base))}
        halves.append({"gallery":len(part),"mAP_and_precision":rows})
        del partial,res
    return {"sample":len(choice),"sample_seed":55429,"rank_bins":estimates,"gallery_halves":halves,
            "max_abs_vectorized_vs_official_sample_AP_P":(sample_details-official[choice]).abs().amax(0).tolist(),
            "single_query_vs_mixed_batch":invariance,"typical_top1_changes_relative_to_uniform":examples_kept}


@torch.inference_mode()
def audit(workspace):
    root = workspace/"results"
    decision, locks = verify_decision(root)
    primary = decision["primary"]
    module = import_measured(workspace,primary,locks[primary]) if primary in locks else None
    cfg = module.Config(**locks[primary]["winner"]) if module else None
    reverse = Setting(**read(root/"reverse-prompt-diffusion/freeze.json")["winner"]).name
    ids = {"native":("conditional-semantic-index","native"),"uniform":(FAMILY,"uniform"),"MLE":(FAMILY,PRIMARY),
           "old_diffusion":("reverse-prompt-diffusion",reverse)}
    ids.update({name:(name,lock["winner_id"]) for name,lock in locks.items()})
    selected = primary if primary in locks else "MLE"
    output = {"joint":decision,"audit_revision":revision(),"splits":{}}
    table = []
    for dataset in ("sketchy","tuberlin"):
        args,qp,queries,pp,galleries,raw,rgb = global_inputs(workspace,dataset)
        words = read_words(args.vocabulary)
        source,target = (v.cuda() for v in load_words(args,words))
        _,scale = bare_words(args,words)
        for split,(qi,ql) in queries.items():
            pi,pl = galleries[split]
            location = "heldout" if split=="tuberlin" else "sketchy"
            rows,pts,controls,contrasts = {},{},{},{}
            for label,(family,cid) in ids.items():
                rows[label],pts[label] = load_result(root,family,split,cid,ql,pl,locks.get(family))
            for n,lock in locks.items():
                saved = tensor(root/n/location/f"{split}-labels.pt")
                if not torch.equal(ql,saved["query"]) or not torch.equal(pl,saved["gallery"]):
                    raise ValueError("Query or gallery order differs from the loader")
                controls[n] = {}
                for key,value in lock["controls"].items():
                    row,points = load_result(root,n,split,digest(value)[:16],ql,pl,lock)
                    controls[n][key] = row|{"parameters":value}
                    if n==primary:
                        contrasts["control/"+key] = compare(pts[selected],points,ql)
            for label,points in pts.items():
                contrasts[label] = compare(pts[selected],points,ql)
                table.append({"split":split,"method":label,"mAP":rows[label]["mAP"],"precision":rows[label]["precision"],
                              "map_k":rows[label]["map_k"],"p_k":rows[label]["p_k"],"queries":len(ql),"gallery":len(pl)})
            per_class = {klass:{"queries":int((ql==i).sum()),"metrics":{k:on_subset(v,ql==i) for k,v in pts.items()}}
                         for i,klass in enumerate(UNSEEN_CLASSES[split])}
            scores = {"metrics":rows,"contrasts":contrasts,"controls":controls,"classes":per_class,
                      "queries":len(ql),"gallery":len(pl)}
            if split=="tuberlin":
                shared = set(UNSEEN_CLASSES["sketchy_1"])|set(UNSEEN_CLASSES["sketchy_2"])
                mask = torch.tensor([k not in shared for k in UNSEEN_CLASSES[split]])[ql]
                scores["names_disjoint_from_Sketchy"] = {"queries":int(mask.sum()),"classes":len(ql[mask].unique()),
                    "metrics":{label:on_subset(points,mask) for label,points in pts.items()}}
            inputs = (raw[qi],rgb[pi].cuda(),ql,pl,source,target,scale)
            scores["diagnosis"] = rank_audit(inputs,split,module,cfg,pts[selected],[qp[i] for i in qi.tolist()],[pp[i] for i in pi.tolist()])
            output["splits"][split] = scores
            print(split,primary,rows[selected],flush=True)
    out = root/"map-study"
    atomic_json(out/"transfer-audit.json",output)
    with (out/"full-metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    audit(p.parse_args().workspace)
