"""Tune concept-space retrieval on Sketchy and lock the method before scoring TU-Berlin."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch

from .concept_likelihood import ConceptIndex, Config, photo_graph, pilot_configs, propagate, ranked_scores
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .metrics import cosine_batches
from .retrieval_diagnosis import global_inputs, stratified_sample
from .semantic import read_words
from .transfer_benchmark import write_result


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    temp.replace(path)


def read(path):
    return json.loads(path.read_text())


def checksum(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def controls(winner):
    return {
        "primary": winner, "no_walk": replace(winner, walk=0.),
        "no_prior": replace(winner, prior_power=0.), "only_likelihood": replace(winner, direct=0.),
        "direct_only": replace(winner, direct=1.), "neutral_prompts": replace(winner, pairing="neutral"),
        "shuffled_pairs": replace(winner, pairing="shuffled"),
    }


def shortlist(folder):
    catalog = {c.name: c for c in pilot_configs()}
    tables = [read(folder / f"pilot-{s}.json")["results"] for s in SKETCHY_SPLITS]
    if any(set(t) != set(catalog) for t in tables):
        raise ValueError("Pilot incomplete")
    def key(c):
        return sum(t[c.name]["precision"] for t in tables) / 2
    eligible = [c for c in catalog.values() if c.channel == "mixture" and c.pairing == "paired"]
    best = sorted(eligible, key=lambda c: (-key(c), c.name))
    candidates = list(dict.fromkeys([*best[:4], *[max((c for c in eligible if c.target_temp == t), key=key)
                                                  for t in sorted({c.target_temp for c in eligible})],
                                     max((c for c in eligible if not c.walk), key=key), Config()]))
    # Cross influential coefficients around the observed pilot winner, always using both splits.
    chosen = candidates[0]
    for k in sorted({chosen.neighbors, 192}):
        for blend in sorted({0., chosen.direct, .7}):
            candidates.append(replace(chosen, neighbors=k, direct=blend))
    return list(dict.fromkeys(candidates))


def base_blocks(query, p, cfg, bank, graph):
    if cfg is None:
        yield from cosine_batches(query, p, 4)
        return
    for block in query.split(96):
        yield propagate(bank.evidence(block, cfg), graph, cfg.seeds, cfg.walk)


def official(args, stage_dir, split, query, p, ql, pl, cfg, bank, graph, common):
    name = "native-clip" if cfg is None else cfg.name
    path = stage_dir / f"{split}-{name}.json"
    details = stage_dir / "per-query" / f"{split}-{name}.pt"
    rank1 = stage_dir / "top1" / f"{split}-{name}.pt"
    if path.is_file() and details.is_file() and rank1.is_file():
        previous = read(path)
        if any(previous[k] != common[k] for k in ("revision", "freeze_sha256", "checkpoint_sha256", "split_sha256")):
            raise ValueError(f"Stale result {path}")
        return
    top = []
    def rows():
        for scores in base_blocks(query, p, cfg, bank, graph):
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite score")
            top.append(scores.argmax(1).cpu().int())
            batch = scores if cfg is None else ranked_scores(scores)
            yield from batch.cpu()
    args.output = stage_dir
    stage_dir.mkdir(parents=True, exist_ok=True)
    params = {"ranking": "native cosine"} if cfg is None else asdict(cfg)
    write_result(args, split, name, rows(), ql, pl,
                 common | {"external_vocabulary": 0 if cfg is None else len(bank.target)}, params)
    save_tensor(torch.cat(top), rank1)


def pilot(query, p, ql, pl, c, bank, graph, k):
    correct, top1 = 0, 0
    for i, s in enumerate(base_blocks(query, p, c, bank, graph)):
        start = i * 96
        ids = s.topk(k, 1).indices
        matches = pl[ids] == ql[start:start + len(s), None]
        correct += int(matches.sum())
        top1 += int(matches[:, 0].sum())
    return {"precision": correct / (k * len(query)), "top1": top1 / len(query)}


def get_winner(folder, candidates):
    def key(c):
        rows = [read(folder / "sketchy" / f"{s}-{c.name}.json") for s in SKETCHY_SPLITS]
        return (sum(r["precision"] for r in rows) / 2, sum(r["mAP"] for r in rows) / 2)
    return max(candidates, key=key)


def freeze(folder):
    if (folder / "freeze.json").is_file():
        return
    if list((folder / "heldout").glob("tuberlin-*.json")):
        raise ValueError("Held-out scores exist before selection")
    candidates = shortlist(folder)
    winner = get_winner(folder, candidates)
    scores = {}
    selections = controls(winner)
    for s in SKETCHY_SPLITS:
        scored = [folder / "sketchy" / f"{s}-{c.name}.json" for c in {*candidates, *selections.values()}]
        scored.append(folder / "sketchy" / f"{s}-native-clip.json")
        for p in scored:
            row = read(p)
            if row["revision"] != revision() or row["split_sha256"] != SPLIT_HASHES[s]:
                raise ValueError(f"Unexpected revision or split for {p}")
            scores[p.name] = checksum(p)
    f = {"utc": datetime.now(timezone.utc).isoformat(), "revision": revision(),
         "selection": "equal Sketchy precision, then equal Sketchy mAP", "winner": asdict(winner),
         "controls": {k: asdict(c) for k, c in selections.items()}, "heldout_rows_at_lock": 0,
         "sketchy_rows": scores, "checkpoint": CHECKPOINT}
    atomic_json(folder / "freeze.json", f | {"freeze_sha256": digest(f)})


@torch.inference_mode()
def run(workspace, stage):
    folder = workspace / "results/calibrated-concept-retrieval"
    if stage == "freeze":
        freeze(folder)
        return
    held = stage == "heldout"
    lock = {}
    if held:
        lock = read(folder / "freeze.json")
        signature = lock.pop("freeze_sha256")
        if digest(lock) != signature or lock["revision"] != revision():
            raise ValueError("Invalid frozen protocol")
        lock["freeze_sha256"] = signature
        configs = [Config(**p) for p in lock["controls"].values()]
    elif stage == "pilot":
        configs = pilot_configs()
    else:
        selected = shortlist(folder)
        configs = list(controls(get_winner(folder, selected)).values()) if stage == "controls" else selected
    args, _, queries, _, gallery, qraw, praw = global_inputs(workspace, "tuberlin" if held else "sketchy")
    words = read_words(args.vocabulary)
    if len(words) != 20931:
        raise ValueError("Public vocabulary differs from the reference")
    texts = tuple(t.cuda() for t in load_words(args, words))
    for split, (qids, ql) in queries.items():
        gp, pl = gallery[split]
        if digest(UNSEEN_CLASSES[split]) != SPLIT_HASHES[split]:
            raise ValueError("Unexpected split")
        expected = {"sketchy_1": (14179, 17100, 25), "sketchy_2": (12694, 12552, 21), "tuberlin": (2130, 27262, 30)}
        if (len(ql), len(pl), pl.unique().numel()) != expected[split] or ql.unique().numel() != expected[split][2]:
            raise ValueError(f"Incomplete data for {split}")
        if stage == "pilot":
            take = stratified_sample(ql, 32, 4451)
            qids, ql = qids[take], ql[take]
        q, p = qraw[qids].cuda(), praw[gp].cuda()
        report = folder / ("heldout" if held else "sketchy")
        args.metric_workers = 4 if split != "sketchy_2" else 2
        common = {"gpu": torch.cuda.get_device_name(), "revision": revision(), "checkpoint_sha256": CHECKPOINT,
                  "backbone": "OpenAI CLIP ViT-B/32", "classes": pl.unique().numel(), "split_sha256": digest(UNSEEN_CLASSES[split]),
                  "exploratory_unseen": not held, "freeze_sha256": lock.get("freeze_sha256"),
                  "study": "calibrated-concept-retrieval"}
        if stage != "pilot":
            official(args, report, split, q, p, ql, pl, None, None, None, common)
            save_tensor({"query": ql, "gallery": pl}, report / f"{split}-labels.pt")
        graphs, result = {}, {}
        pilot_path = folder / f"pilot-{split}.json"
        for temperature in sorted({c.target_temp for c in configs}):
            bank = ConceptIndex(p, *texts, temperature)
            for c in dict.fromkeys(v for v in configs if v.target_temp == temperature):
                if c.walk and c.neighbors not in graphs:
                    graphs[c.neighbors] = photo_graph(p, c.neighbors)
                graph = graphs.get(c.neighbors)
                if stage == "pilot":
                    result[c.name] = pilot(q, p, ql.cuda(), pl.cuda(), c, bank, graph, 200 if split == "sketchy_2" else 100)
                else:
                    official(args, report, split, q, p, ql, pl, c, bank, graph, common)
            del bank
        if stage == "pilot":
            atomic_json(pilot_path, common | {"results": result, "queries": len(q), "gallery": len(p), "seed": 4451})
            print(split, sorted(result.items(), key=lambda pair: -pair[1]["precision"])[:3], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--stage", choices=["pilot", "sketchy", "controls", "freeze", "heldout"], required=True)
    opts = parser.parse_args()
    run(opts.workspace, opts.stage)
