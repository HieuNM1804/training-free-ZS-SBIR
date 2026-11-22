"""Select the gallery-panel estimator on the complete Sketchy splits, then score TU-Berlin."""

import argparse
import hashlib
import inspect
from pathlib import Path
import time

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .gallery_depiction_study import views
from .gallery_region_panel_study import audited_rows, evaluate, key, parse, select
from .gallery_region_panels import FAMILY, compile_all
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import load_kd, paired_delta
from .semantic import read_words


def shortlist(pilot):
    values = pilot["means"]
    rows = set(audited_rows(pilot_selection(pilot)))
    for mode in ("cross", "cross_diff", "same", "average", "uniform", "reverse_diff", "wrong", "wrong_diff"):
        ranked = sorted((k for k in values if k != "regional" and parse(k)[2] == mode),
                        key=values.get, reverse=True)
        rows.update(ranked[:3])
    for row in list(rows):
        if row == "regional":
            continue
        kind, size, mode, temp, weight = parse(row)
        if mode == "cross":
            rows.update(key(kind, size, m, temp, weight) for m in ("same","average","wrong"))
            rows.add(key(kind, size, "uniform", 0, weight))
        if mode == "cross_diff":
            rows.update(key(kind, size, m, temp, weight) for m in ("wrong_diff","reverse_diff"))
    return sorted(rows)


def pilot_selection(pilot):
    return {k:v for k,v in pilot.items() if k == "nominee" or k == "matched" or k.startswith("best_")}


def store_or_check(path, info):
    record = info | {"sha256": digest(info)}
    if path.exists() and read(path) != record:
        raise ValueError("Changed stored protocol at "+str(path))
    atomic_json(path, record)
    return record["sha256"]


def read_valid(path, sig):
    if not path.exists():
        return None
    record = read(path)
    if record["signature"] != sig or checksum(path.with_suffix(".pt")) != record["per_query_sha256"]:
        raise ValueError("Invalid split checkpoint "+str(path))
    return record


def load_split_inputs(w, ds):
    cfg, paths, queries, _, galleries, _, _ = global_inputs(w, ds)
    names = read_words(cfg.vocabulary)
    if len(names) != 20931:
        raise ValueError("Changed public vocabulary")
    src = load_words(cfg, names)[0].cuda()
    _, scale = bare_words(cfg, names)
    view = views(w, ds, paths)
    torch.backends.cuda.matmul.allow_tf32 = True
    for s in queries:
        if digest(UNSEEN_CLASSES[s]) != SPLIT_HASHES[s]:
            raise ValueError("Changed unseen classes")
    return queries, galleries, src, scale, view


def split_result(w, s, inputs, rows, path, sig, kd, workers, batch, primary=None, scan=None):
    finished = read_valid(path, sig)
    if finished is not None:
        return finished
    qs, gs, src, scale, view = inputs
    labels = qs[s][1]
    ids = torch.arange(len(labels))
    before = time.monotonic()
    met, top, evidence, err = evaluate(w, s, qs[s], gs[s], src, scale, view, ids, rows, kd, workers, batch)
    if any(v.shape != (len(ids), 2) for v in met.values()):
        raise ValueError("Incomplete split")
    replay_diff = None
    if scan is not None:
        previous = torch.load(scan, weights_only=True, mmap=True)["metrics"]
        replay_diff = max(float((old-previous[k]).abs().max()) for k,old in met.items())
        if replay_diff > 2e-6:
            raise ValueError(f"Official replay disagrees with scan {replay_diff}")
    save_tensor({"query_ids":ids, "query_labels":labels, "metrics":met, "top1_correct":top,
                 "panel_diagnostics":evidence}, path.with_suffix(".pt"))
    values = {k:v.mean(0).tolist() for k,v in met.items()}
    result = {"signature":sig, "queries":len(labels), "gallery":len(gs[s][1]), "classes":len(labels.unique()),
              "values":values, "scores":{k:dict(zip(("mAP","precision"),v)) for k,v in values.items()},
              "per_query_sha256":checksum(path.with_suffix(".pt")), "GPU":torch.cuda.get_device_name(),
              "seconds":time.monotonic()-before, "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20}
    if primary:
        ap = met[primary][:,0]-met["regional"][:,0]
        result.update(primary=primary, original_kd_queries_per_condition=len(labels), kd_max_abs_error=err,
                      vs_controls={k:paired_delta(met[primary],v,labels) for k,v in met.items()},
                      reference_error=replay_diff, top1_changes={"fixed":int((~top["regional"] & top[primary]).sum()),
                      "broken":int((top["regional"] & ~top[primary]).sum())},
                      AP_queries_up=int((ap>1e-6).sum()), AP_queries_down=int((ap< -1e-6).sum()))
    atomic_json(path, result)
    print("stored", s, "KD" if kd else "scan", len(ids), values.get(primary), flush=True)
    return result


@torch.inference_mode()
def run(w, workers, batch):
    root = w/"results"/FAMILY
    pilot = read(root/"pilot.json")
    rows = shortlist(pilot)
    kd, _ = load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("Changed original evaluator")
    protocol = {"revision":revision(), "checkpoint":CHECKPOINT, "KD":KD_FUNCTION_SHA, "pilot":checksum(root/"pilot.json"),
                "rule":"top three per condition on balanced pilots plus controls at matched parameters; select on mean complete Sketchy mAP then mean precision",
                "rows":rows, "batch":batch}
    study = store_or_check(root/"full-protocol.json", protocol)
    store_or_check(root/"shortlist.json", {"full_protocol":study, "rows":rows, "pilot_main":pilot["nominee"]})
    sketches = load_split_inputs(w, "sketchy")
    tables = {}
    for split in sketches[0]:
        p = root/("scan-"+split+".json")
        tables[split] = split_result(w, split, sketches, rows, p, study, None, workers, batch)["values"]
    means = {k:[sum(x[k][j] for x in tables.values())/len(tables) for j in (0,1)] for k in rows}
    choice = select(means)
    audit = audited_rows(choice)
    lock = store_or_check(root/"freeze.json", choice | {"means":means,"full_protocol":study,"official_rows":audit,
                    "complete_Sketchy":{s:checksum(root/("scan-"+s+".json")) for s in tables}})
    print("frozen", choice["nominee"], means[choice["nominee"]], "baseline", means["regional"], flush=True)
    for s in sketches[0]:
        split_result(w,s,sketches,audit,root/(s+".json"),lock,kd,workers,batch,choice["nominee"],
                     root/("scan-"+s+".pt"))
    del sketches
    torch.cuda.empty_cache()
    compile_all(w,"tuberlin")
    tu = load_split_inputs(w, "tuberlin")
    split_result(w,"tuberlin",tu,audit,root/"tuberlin.json",lock,kd,workers,batch,choice["nominee"])
    print("complete", lock, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    cli.add_argument("--workers", type=int, default=8)
    cli.add_argument("--batch", type=int, default=128)
    args = cli.parse_args()
    run(args.workspace, args.workers, args.batch)
