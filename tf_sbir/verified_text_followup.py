"""Check a pilot-nominated soft token gate on all queries and two locked regional readouts."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
import json
import math
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES, bare_words
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT
from .cross_view_region_cache import FAMILY as VIEWS
from .cross_view_regions import CandidateRegions
from .data import UNSEEN_CLASSES
from .encoding import load_frozen_clip, save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .regional_intervention_study import index as compact_index
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import load_kd, paired_delta
from .semantic import read_words
from .verified_text_span_study import FAMILY
from .verified_text_spans import ObjectSpans, prepare

PILOT_REV = "eaa8136"
SKETCHY = ("sketchy_1", "sketchy_2")


def key(setting):
    return json.dumps(setting, separators=(",", ":"))


def nominate(results, gate, count):
    eligible = []
    for candidate, scores in results.items():
        if candidate == "regional":
            continue
        names, view, method, temp, readout, strength = json.loads(candidate)
        if view == "zoom" and method == gate and readout == "increment":
            eligible.append((scores, (names, gate, temp, strength, "native")))
    return [cell for _, cell in sorted(eligible, reverse=True)[:count]]


def choose(table, gate):
    return json.loads(max((k for k in table if json.loads(k)[1] == gate), key=table.get))


def regional_scores(photo_names, query_native, query_reverse, raw, mirror, src, heads, compact, history, scale):
    # These comparators use the native full-image score. Supply an empty second
    # index since neither comparator uses the optional dual-pose global score.
    empty_index = src.new_empty((0, len(src)))
    regions = CandidateRegions(raw, mirror, query_native, query_reverse, src, heads, compact, empty_index, history)
    regions.prepare(scale)
    native = .4 * regions.read("native", 0., "raw") + .6 * regions.first
    out = {"native": native}
    if "candidate" in photo_names:
        out["candidate"] = .35 * regions.read("self", 1., "gallery") + .65 * regions.first
    return out


def evaluate_split(w, s, qi, y, photo, g, qraw, views, pp_signature, src, tgt, spans, scale, cells, kd, workers, primary):
    heads, compact = compact_index(w, s, photo, tgt, scale, pp_signature)
    history = torch.load(w / "cache/pragmatic-gallery-competition" / (s + "-named-rgb16.pt"), mmap=True, weights_only=True)
    if history.shape != (len(y), len(g)):
        raise ValueError("Historical base does not align with this gallery and these queries")
    banks = {n: prepare(photo, tgt, src, spans, scale, n) for n in sorted({c[0] for c in cells})}
    diagnostics = {n: [panel.diagnostics for panel in panels] for n, (_, panels) in banks.items()}
    results, errors, top = defaultdict(list), {}, []
    gallery_labels = g.cuda()
    replay = Replay(kd, g, s) if kd else None
    n_queries = len(qi)
    anchors = {c[4] for c in cells}
    with (ThreadPoolExecutor(max_workers=workers) if replay else nullcontext()) as pool:
        for start in range(0, n_queries, 64):
            rows = slice(start, min(start + 64, n_queries))
            at = qi[rows]
            native = views["original_regions"][at].cuda().float()
            reverse = views["regions"][at].cuda().float()
            base = regional_scores(anchors, native, reverse, qraw[at].cuda(), views["global"][at].cuda(), src,
                                   heads, compact, history[rows].cuda(), scale)
            labels = y[rows]
            for count, (members, panels) in banks.items():
                observed = [panel.responses(native, src, scale) for panel in panels]
                here = [c for c in cells if c[0] == count]
                modes = {(c[1], c[2]) for c in here} | {("hard", 1.)}
                by_gate = {}
                for method, temp in modes:
                    panel_scores = [p.groups(x, method, temp) for p, x in zip(panels, observed)]
                    by_gate[method, temp] = torch.logaddexp(*panel_scores) - math.log(2)
                for c in here:
                    _, gate, t, strength, anchor = c
                    score = base[anchor]
                    if strength:
                        increment = by_gate[gate, t] - by_gate["hard", 1.]
                        score = score + strength * increment[:, members]
                    fast, order, _, _ = map_values(score, labels.cuda(), gallery_labels, s)
                    k = key(c)
                    if replay:
                        true = replay.score(order, labels, pool)
                        error = float((true - fast.cpu()).abs().max())
                        if error > 2e-6:
                            raise ValueError("Original KD evaluator differs from the fast scores")
                        errors[k] = max(errors.get(k, 0.), error)
                        if k == primary:
                            top.append(order[:, :25].int().cpu())
                    else:
                        true = fast.cpu()
                    results[k].append(true)
    return {k: torch.cat(v) for k, v in results.items()}, errors, torch.cat(top) if top else None, diagnostics


@torch.inference_mode()
def run(w, phase, workers):
    root = w / "results" / FAMILY
    pilot = read(root / "protocol.json")
    if not pilot["revision"].startswith(PILOT_REV):
        raise ValueError("A different pilot was supplied")
    pilot_scores = read(root / "pilot.json")["means"]
    spec = {"revision": revision(), "pilot_sha256": checksum(root / "pilot.json"),
            "primary": "local", "base": "native", "transfer_comparator": "candidate",
            "nominees": nominate(pilot_scores, "local", 3) + nominate(pilot_scores, "verified", 2)}
    signature = digest(spec)
    spec_path = root / "followup-protocol.json"
    if spec_path.exists() and read(spec_path)["sha256"] != signature:
        raise ValueError("Follow-up protocol has changed")
    atomic_json(spec_path, spec | {"sha256": signature})
    reference, _ = load_kd(w)
    if hashlib.sha256(inspect.getsource(reference).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("Unexpected KD-SBIR source")

    stages = {"lock": ("shortlist",), "audit": ("final",), "all": ("shortlist", "final")}[phase]
    for stage in stages:
        if stage == "shortlist":
            cells = [tuple(c) for c in spec["nominees"]]
            primary = None
        else:
            freeze = read(root / "freeze.json")
            if freeze["signature"] != signature or freeze["shortlist_sha256"] != checksum(root / "shortlist.json"):
                raise ValueError("Freeze does not match the full Sketchy selection")
            locked = tuple(freeze["primary"])
            n, _, t, a, _ = locked
            cells = sorted({locked, tuple(freeze["original_primary"]),
                            (n, "hard", 1., 0., "native"), (n, "hard", 1., 0., "candidate"),
                            (n, "local", t, a, "candidate"), (n, "local_output", t, a, "candidate"),
                            *[(n, gate, t, a, "native") for gate in (
                                "shared", "always", "local_reverse", "local_output", "output_own_gate", "verified")]})
            primary = key(locked)
        means = {}
        for ds in (("sketchy", "tuberlin") if stage == "final" else ("sketchy",)):
            cfg, _, qs, pp, galleries, qraw, praw = global_inputs(w, ds)
            words = read_words(cfg.vocabulary)
            if len(words) != 20931:
                raise ValueError("Public vocabulary has changed")
            if ds == "sketchy":
                model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
                spans = ObjectSpans(model, words)
            src, tgt = (x.cuda() for x in load_words(cfg, words))
            _, scale = bare_words(cfg, words)
            viewset = torch.load(w / "cache" / VIEWS / ds / "views.pt", mmap=True, weights_only=True)
            for s, (qi, y) in qs.items():
                pi, g = galleries[s]
                if (len(qi), len(pi), len(g.unique())) != SIZES[s] or digest(UNSEEN_CLASSES[s]) != SPLIT_HASHES[s]:
                    raise ValueError("Unexpected split")
                file = root / ((s if stage == "final" else "shortlist-" + s) + ".json")
                if file.exists():
                    record = read(file)
                    if record["signature"] != signature or (stage == "final" and record["per_query_sha256"] != checksum(file.with_suffix(".pt"))):
                        raise ValueError("Stale split record")
                    means[s] = record["values"]
                    continue
                photo = praw[pi].cuda()
                ident = digest({"study": pilot["revision"], "gallery": digest([str(pp[i]) for i in pi.tolist()])})[:16]
                rows, err, top, checks = evaluate_split(w, s, qi, y, photo, g, qraw, viewset, ident, src, tgt, spans,
                                                        scale, cells, reference if stage == "final" else None, workers, primary)
                vals = {k: x.mean(0).tolist() for k, x in rows.items()}
                record = {"signature": signature, "queries": len(qi), "gallery": len(pi), "values": vals, "verification": checks}
                if stage == "final":
                    tensor = file.with_suffix(".pt")
                    save_tensor({"metrics": rows, "query_labels": y, "top25": top}, tensor)
                    record.update({"scores": {k: dict(zip(("mAP", "precision"), val)) for k, val in vals.items()},
                                   "protocol": {"split": s, "queries": len(qi), "gallery": len(pi), "classes": len(g.unique()),
                                                "revision": spec["revision"], "checkpoint": CHECKPOINT, "kd_sha256": KD_FUNCTION_SHA,
                                                "gpu": torch.cuda.get_device_name(), "vocabulary_sha256": checksum(cfg.vocabulary),
                                                "map_k": 200 if s == "sketchy_2" else 0, "p_k": 200 if s == "sketchy_2" else 100},
                                   "primary": primary, "best_contrast": key((n, "local_output", t, a, "native")),
                                   "matched_base": key((n, "hard", 1., 0., "native")),
                                   "baseline": key((n, "hard", 1., 0., "candidate")), "original_primary": key(freeze["original_primary"]),
                                   "candidate_gate": key((n, "local", t, a, "candidate")),
                                   "vs_controls": {k: paired_delta(rows[primary], v, y) for k, v in rows.items()},
                                   "transfer_deltas": paired_delta(rows[key((n, "local", t, a, "candidate"))], rows[key((n, "hard", 1., 0., "candidate"))], y),
                                   "kd_max_abs_error": err, "original_kd_queries_per_condition": len(y), "per_query_sha256": checksum(tensor)})
                atomic_json(file, record)
                means[s] = vals
                print(stage, s, max(vals, key=vals.get), vals[max(vals, key=vals.get)], flush=True)
        if stage == "shortlist":
            mean = {k: [sum(means[s][k][i] for s in SKETCHY) / 2 for i in (0, 1)] for k in means[SKETCHY[0]]}
            atomic_json(root / "shortlist.json", {"means": mean, "splits": means})
            freeze = {"signature": signature, "revision": spec["revision"], "primary": choose(mean, "local"),
                      "original_primary": choose(mean, "verified"), "shortlist_sha256": checksum(root / "shortlist.json")}
            atomic_json(root / "freeze.json", freeze | {"sha256": digest(freeze)})
            print("locked", freeze, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--phase", choices=("lock", "audit", "all"), default="all")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    run(args.workspace, args.phase, args.workers)
