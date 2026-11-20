"""Run the reverse and forward gallery projections, freeze on Sketchy, score locked TU controls."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, official, read
from .concept_likelihood import ranked_scores
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .retrieval_diagnosis import global_inputs, stratified_sample
from .reverse_prompt import PhotoIndex, Setting, candidates, matched_controls
from .semantic import read_words
from .transfer_benchmark import write_result


def pilot_score(index, queries, config, labels, photo_labels, k):
    total, hits = 0, 0
    for start, scores in enumerate(index.scores(queries, config, block_size=128)):
        begin = start * 128
        top = scores.topk(k).indices
        correct = photo_labels[top] == labels[begin:begin + len(scores), None]
        total += int(correct.sum())
        hits += int(correct[:, 0].sum())
    return {"precision": total / (len(labels) * k), "top1": hits / len(labels)}


def shortlist(folder):
    table = [read(folder / f"pilot-{s}.json")["results"] for s in SKETCHY_SPLITS]
    choices = candidates()
    if any(set(t) != {c.name for c in choices} for t in table):
        raise ValueError("Incomplete pilot")
    ranked = sorted(choices, key=lambda c: (-sum(t[c.name]["precision"] for t in table), c.name))
    selected = ranked[:4] + [next(c for c in ranked if c.direction == d)
                             for d in ("reverse", "forward", "average")]
    winner = ranked[0]
    # Combine the leading direction with individual parameter changes before making the final choice.
    for terms in (8, 32, 64):
        selected.append(replace(winner, terms=terms))
    for k in (48, 192, 384):
        selected.append(replace(winner, support=k))
    for value in (.35, .85):
        selected.append(replace(winner, expansion=value))
    selected.append(replace(Setting(), diffusion=0., expansion=0.))
    return list(dict.fromkeys(selected))


def winner(folder):
    def key(c):
        rows = [read(folder / "sketchy" / f"{s}-{c.name}.json") for s in SKETCHY_SPLITS]
        return sum(r["precision"] for r in rows), sum(r["mAP"] for r in rows)
    return max(shortlist(folder), key=key)


def freeze(folder, rev):
    file = folder / "freeze.json"
    if file.exists():
        read_lock(folder, rev)
        return
    if list((folder / "heldout").glob("tuberlin-*.json")):
        raise ValueError("Previous TU scores in this study directory")
    selected = winner(folder)
    controls = matched_controls(selected)
    checks = {}
    for s in SKETCHY_SPLITS:
        names = {c.name for c in [*shortlist(folder), *controls.values()]} | {"native-clip"}
        for n in names:
            p = folder / "sketchy" / f"{s}-{n}.json"
            row = read(p)
            if row["revision"] != rev or row["split_sha256"] != SPLIT_HASHES[s]:
                raise ValueError("Invalid source for freeze")
            checks[p.name] = checksum(p)
    record = {"utc": datetime.now(timezone.utc).isoformat(), "revision": rev, "checkpoint_sha256": CHECKPOINT,
              "chosen_on": list(SKETCHY_SPLITS), "criterion": "equal mean precision, then equal mean mAP",
              "winner": asdict(selected), "controls": {k: asdict(c) for k,c in controls.items()},
              "sketchy_scores_sha256": checks, "tu_score_count": 0}
    atomic_json(file, record | {"freeze_sha256": digest(record)})


def read_lock(folder, rev):
    lock = read(folder / "freeze.json")
    signature = lock.pop("freeze_sha256")
    if digest(lock) != signature or lock["revision"] != rev or lock["tu_score_count"] != 0:
        raise ValueError("Invalid lock for TU")
    return lock | {"freeze_sha256": signature}


def record_config(args, out, s, c, idx, queries, ql, pl, meta):
    row = out / f"{s}-{c.name}.json"
    top_path = out / "top10" / f"{s}-{c.name}.pt"
    if row.is_file() and top_path.is_file() and (out / "per-query" / (row.stem + ".pt")).is_file():
        saved = read(row)
        if any(saved[k] != meta[k] for k in ("revision", "freeze_sha256", "split_sha256")):
            raise ValueError("Stale result")
        return
    top = []
    def score_rows():
        for scores in idx.scores(queries, c):
            if not torch.isfinite(scores).all():
                raise ValueError("Invalid retrieval score")
            top.append(scores.topk(10).indices.int().cpu())
            yield from ranked_scores(scores).cpu()
    args.output = out
    out.mkdir(parents=True, exist_ok=True)
    write_result(args, s, c.name, score_rows(), ql, pl, meta, asdict(c))
    save_tensor(torch.cat(top), top_path)


@torch.inference_mode()
def run(workspace, stage):
    folder = workspace / "results/reverse-prompt-diffusion"
    rev = revision()
    if stage == "freeze":
        freeze(folder, rev)
        return
    held = stage == "heldout"
    lock = read_lock(folder, rev) if held else {}
    if held:
        cfgs = [Setting(**v) for v in lock["controls"].values()]
    elif stage == "pilot":
        cfgs = candidates()
    elif stage == "sketchy":
        cfgs = shortlist(folder)
    else:
        cfgs = list(matched_controls(winner(folder)).values())
    args, _, queries, _, photos, raw, rgb = global_inputs(workspace, "tuberlin" if held else "sketchy")
    words = read_words(args.vocabulary)
    if len(words) != 20931:
        raise ValueError("Unexpected public vocabulary")
    source, target = (v.cuda() for v in load_words(args, words))
    out = folder / ("heldout" if held else "sketchy")
    for s, (qi, ql) in queries.items():
        pi, pl = photos[s]
        expected = {"sketchy_1": (14179, 17100, 25), "sketchy_2": (12694, 12552, 21), "tuberlin": (2130, 27262, 30)}
        if ((len(ql), len(pl), pl.unique().numel()) != expected[s] or ql.unique().numel() != expected[s][2]
                or digest(UNSEEN_CLASSES[s]) != SPLIT_HASHES[s]):
            raise ValueError(f"Unexpected test protocol {s}")
        if stage == "pilot":
            selection = stratified_sample(ql, 32, 4451)
            qi, ql = qi[selection], ql[selection]
        q, p = raw[qi].cuda(), rgb[pi].cuda()
        index = PhotoIndex(p, source, target)
        meta = {"revision": rev, "backbone": "OpenAI CLIP ViT-B/32", "checkpoint_sha256": CHECKPOINT,
                "gpu": torch.cuda.get_device_name(), "split_sha256": SPLIT_HASHES[s], "classes": expected[s][2],
                "study": "reverse-prompt-diffusion", "external_vocabulary": len(source),
                "freeze_sha256": lock.get("freeze_sha256"), "exploratory_unseen": not held}
        if stage == "pilot":
            results = {c.name: pilot_score(index, q, c, ql.cuda(), pl.cuda(), 200 if s == "sketchy_2" else 100)
                       for c in cfgs}
            atomic_json(folder / f"pilot-{s}.json", meta | {"results": results, "seed": 4451, "queries": len(q)})
            print(s, sorted(results.items(), key=lambda pair: -pair[1]["precision"])[:4], flush=True)
        else:
            args.metric_workers = 4 if s != "sketchy_2" else 2
            official(args, out, s, q, p, ql, pl, None, None, None, meta)
            save_tensor({"query": ql, "gallery": pl}, out / f"{s}-labels.pt")
            for c in dict.fromkeys(cfgs):
                if c.direction == "native":
                    record_config(args, out, s, c, index, q, ql, pl, meta | {"external_vocabulary": 0})
                else:
                    record_config(args, out, s, c, index, q, ql, pl, meta)
        del index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--stage", choices=("pilot", "sketchy", "controls", "freeze", "heldout"), required=True)
    opts = parser.parse_args()
    run(opts.workspace, opts.stage)
