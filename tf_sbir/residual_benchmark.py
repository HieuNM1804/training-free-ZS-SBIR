"""Two Sketchy screening rounds, complete KD scoring, then prospective TU transfer."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path

import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json, checksum, read
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT, FrameViews
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .mta import load_view_features
from .retrieval_diagnosis import global_inputs, stratified_sample
from .reverse_benchmark import pilot_score, record_config
from .semantic import read_words


SIZES = {"sketchy_1": (14179, 17100, 25), "sketchy_2": (12694, 12552, 21), "tuberlin": (2130, 27262, 30)}
INCUMBENT = "reverse-prompt-diffusion"


def ordered(records):
    by_split = [{r["method"]: r for r in rows} for rows in records]
    if by_split[0].keys() != by_split[1].keys():
        raise ValueError("The two Sketchy splits have different configurations")
    def score(name):
        rows = [table[name] for table in by_split]
        return (-sum(r["precision"] for r in rows), -sum(r.get("mAP", 0.) for r in rows), name)
    return [by_split[0][n]["parameters"] for n in sorted(by_split[0], key=score)]


def screened(method, folder, stage):
    def leaders(previous):
        paths = [folder / f"{previous}-{s}.json" for s in SKETCHY_SPLITS]
        return [method.Config(**c) for c in ordered([read(path)["scores"] for path in paths])]
    if stage == "pilot":
        return method.candidates()
    if stage == "refine":
        initial = leaders("pilot")
        return list(dict.fromkeys([*initial[:6], *method.refine(initial[0])]))
    return leaders("refine")[:6]


def winner(method, folder):
    candidates = screened(method, folder, "full")
    rows = [[read(folder / "sketchy" / f"{s}-{c.name}.json") for c in candidates] for s in SKETCHY_SPLITS]
    return method.Config(**ordered(rows)[0])


def freeze(method, folder, rev):
    path = folder / "freeze.json"
    if path.is_file():
        load_lock(folder, rev)
        return
    if any((folder / "heldout").glob("tuberlin-*.json")):
        raise ValueError("TU scores predate lock")
    c = winner(method, folder)
    controls = method.controls(c)
    configs = set(screened(method, folder, "full")) | set(controls.values())
    hashes = {}
    for s in SKETCHY_SPLITS:
        for choice in configs:
            file = folder / "sketchy" / f"{s}-{choice.name}.json"
            row = read(file)
            if row["revision"] != rev or row["split_sha256"] != SPLIT_HASHES[s]:
                raise ValueError("Unexpected source revision or split")
            hashes[file.name] = checksum(file)
    lock = dict(utc=datetime.now(timezone.utc).isoformat(), revision=rev, winner=asdict(c),
                method=c.name, controls={k:asdict(v) for k,v in controls.items()},
                rule="equal mean Sketchy precision, then mAP", sketchy_rows=hashes, tu_score_count=0)
    atomic_json(path, lock | {"freeze_sha256": digest(lock)})


def load_lock(folder, rev=None):
    lock = read(folder / "freeze.json")
    signature = lock.pop("freeze_sha256")
    if digest(lock) != signature or lock["tu_score_count"] or (rev and lock["revision"] != rev):
        raise ValueError("Invalid Sketchy lock")
    return lock | {"freeze_sha256": signature}


def joint(root, studies):
    joint_path = root / "residual-study/joint-freeze.json"
    if joint_path.exists():
        file = read(joint_path)
        signature = file.pop("sha256")
        if digest(file) != signature or set(file["families"]) != set(studies) | {INCUMBENT}:
            raise ValueError("Inconsistent joint freeze")
        return file["primary"]
    old = read(root / "inquiry-joint-freeze.json")
    if old["primary"] != INCUMBENT:
        raise ValueError("Invalid historical incumbent")
    old_entry = old["families"][INCUMBENT]
    families = {INCUMBENT: old_entry | {"historical": True}}
    for study in studies:
        folder = root / study
        lock = load_lock(folder)
        if any((folder / "heldout").glob("tuberlin-*.json")):
            raise ValueError("Holdout already scored in the new study")
        rows = [read(folder / "sketchy" / f"{s}-{lock['method']}.json") for s in SKETCHY_SPLITS]
        families[study] = {"historical": False, "tu_rows_prelock": 0, "revision": lock["revision"],
            "freeze_sha256": lock["freeze_sha256"],
            "scores": [{k:r[k] for k in ("split","method","precision","mAP")} for r in rows]}
    def criterion(key):
        return tuple(sum(x[k] for x in families[key]["scores"])/2 for k in ("precision","mAP"))
    primary = max(families, key=criterion)
    record = {"utc": datetime.now(timezone.utc).isoformat(), "families": families, "primary": primary,
              "selection_splits": list(SKETCHY_SPLITS), "rule": "equal mean Sketchy precision, then mAP"}
    atomic_json(joint_path, record | {"sha256": digest(record)})
    return primary


def get_views(args, paths, mode):
    args.crop_workers, args.crop_batch_size = 8, 768
    data = load_view_features(args, FrameViews(paths), 5, mode, "neutral-frame-reflect-center180-144-v1")
    return data["features"]


@torch.inference_mode()
def run(workspace, study, stage):
    method = import_module("tf_sbir." + study.replace("-", "_"))
    folder = workspace / "results" / study
    rev = revision()
    if stage == "freeze":
        return freeze(method, folder, rev)
    held = stage == "heldout"
    lock = load_lock(folder, rev) if held else {}
    if held:
        join = read(workspace / "results/residual-study/joint-freeze.json")
        signature = join.pop("sha256")
        if digest(join) != signature or join["families"][study]["freeze_sha256"] != lock["freeze_sha256"]:
            raise ValueError("Missing pre-holdout joint decision")
        configs = [method.Config(**v) for v in lock["controls"].values()]
    elif stage == "controls":
        configs = list(method.controls(winner(method, folder)).values())
    else:
        configs = screened(method, folder, stage)
    args, qp, qs, pp, gs, raw_q, raw_p = global_inputs(workspace, "tuberlin" if held else "sketchy")
    texts = load_words(args, read_words(args.vocabulary))
    if not all(len(t) == 20931 for t in texts):
        raise ValueError("Wrong public vocabulary")
    source, target = (t.cuda() for t in texts)
    loader = getattr(method, "get_views", get_views)
    multi_q = loader(args, qp, "sketch") if method.USE_VIEWS else None
    multi_p = loader(args, pp, "photo") if method.USE_VIEWS else None
    out = folder / ("heldout" if held else "sketchy")
    for s,(qi,ql) in qs.items():
        pi,pl = gs[s]
        if (len(qi),len(pi),len(ql.unique())) != SIZES[s] or len(pl.unique()) != SIZES[s][2] or digest(UNSEEN_CLASSES[s]) != SPLIT_HASHES[s]:
            raise ValueError("Unexpected split")
        pilot = stage in ("pilot", "refine")
        count, seed = (28, 9901) if stage == "pilot" else (52, 32123)
        if pilot:
            chosen = stratified_sample(ql,count,seed)
            qi, ql = qi[chosen], ql[chosen]
        q, p = raw_q[qi].cuda(), raw_p[pi].cuda()
        if multi_q is not None:
            q = torch.cat((q[:,None], F.normalize(multi_q[qi].cuda().float(),dim=-1)),1)
            p = torch.cat((p[:,None], F.normalize(multi_p[pi].cuda().float(),dim=-1)),1)
        idx = method.Index(p, source, target)
        metadata = dict(revision=rev,backbone="OpenAI CLIP ViT-B/32",checkpoint_sha256=CHECKPOINT,
                        gpu=torch.cuda.get_device_name(),split_sha256=SPLIT_HASHES[s],classes=SIZES[s][2],
                        study=study,exploratory_unseen=not held,freeze_sha256=lock.get("freeze_sha256"))
        scores = []
        if not pilot:
            save_tensor({"query":ql,"gallery":pl},out/f"{s}-labels.pt")
        for c in dict.fromkeys(configs):
            if pilot:
                result = pilot_score(idx,q,c,ql.cuda(),pl.cuda(),200 if s=="sketchy_2" else 100)
                scores.append({"method":c.name,"parameters":asdict(c),**result})
            else:
                args.metric_workers = 5 if s!="sketchy_2" else 3
                record_config(args,out,s,c,idx,q,ql,pl,metadata | {"external_vocabulary":method.vocab_size(c)})
        if pilot:
            atomic_json(folder/f"{stage}-{s}.json",metadata | {"seed":seed,"queries":len(q),"scores":scores})
            print(s, sorted(((r["precision"],r["method"]) for r in scores),reverse=True)[:3],flush=True)
        del idx, q, p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--study",nargs="+",required=True)
    parser.add_argument("--stage",choices=("pilot","refine","full","controls","freeze","joint","heldout"),required=True)
    args = parser.parse_args()
    if args.stage == "joint":
        print(joint(args.workspace/"results",args.study))
    else:
        if len(args.study) != 1:
            parser.error("Exactly one study is required for scoring")
        run(args.workspace,args.study[0],args.stage)
