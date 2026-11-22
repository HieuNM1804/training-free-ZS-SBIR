"""Audit one-zoom retrieval, with matched scale and per-action original KD metrics."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES, bare_words
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT
from .cross_view_region_cache import FAMILY as VIEWS
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .gallery_region_acquisition_study import FAMILY, single_region_scores
from .gallery_region_actions import photo_channels, utilities, group_prior, within_scale_mismatch, concordance
from .map_diagnosis import map_values
from .partial_object_parts import vocabulary
from .quickdraw_locked import KD_FUNCTION_SHA
from .regional_intervention_study import index as compact_index
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import load_kd, paired_delta
from .semantic import read_words

PILOT = "1745791"
SPLITS = ("sketchy_1", "sketchy_2")


def best(mapping, policy, count):
    subset = [k for k in mapping if k.startswith(policy + "/")]
    return sorted(subset, key=mapping.get, reverse=True)[:count]


def unpack(k):
    policy, temperature, rivals, reader, mix = k.split("/")
    return policy, float(temperature), int(rivals), reader, float(mix)


def replace_policy(key, name):
    return name + "/" + key.split("/", 1)[1]


def actions(full, assignment, channels, fixed, settings):
    rows = {}
    for temp, n in {(temp, n) for _, temp, n, _, _ in settings}:
        prior, groups = group_prior(full, assignment, temp, n)
        values = utilities(prior, groups, channels)
        heldout = values["heldout"].argmax(1)
        large = heldout >= 9
        rows[temp, n] = {**{name: score.argmax(1) for name, score in values.items()},
                         "fixed": fixed[0].expand_as(heldout),
                         "fixed_scale": torch.where(large, fixed[2], fixed[1]),
                         "misplaced": within_scale_mismatch(heldout),
                         "predicted": values["heldout"], "large": large}
    return rows


def evaluate(w, cfg, split, qrows, y, photos, g, photo_paths, gallery_rows, native, zooms,
             src, tgt, scale, word_path, keys, primary, kd, workers):
    names = [unpack(k) for k in keys]
    readers = sorted({(reader, mix) for _, _, _, reader, mix in names})
    primary_name = unpack(primary) if primary else None
    ident = digest({"gallery": digest([str(photo_paths[j]) for j in gallery_rows.tolist()]), "revision": read(word_path)["revision"]})[:16]
    heads, compact = compact_index(w, split, photos, tgt, scale, ident)
    terms = vocabulary(zooms, tgt, scale, 128)
    labels, channels = photo_channels(photos, zooms, tgt[terms], scale)
    count = channels.size(1)
    equal = photos.new_ones((1, count))
    fixed_risk = utilities(equal, torch.arange(count, device=photos.device)[None], channels)["heldout"][0]
    fixed = torch.stack((fixed_risk.argmax(), fixed_risk[:9].argmax(), 9 + fixed_risk[9:].argmax()))
    hist = torch.load(w / "cache/pragmatic-gallery-competition" / (split + "-named-rgb16.pt"), mmap=True, weights_only=True)
    if hist.shape != (len(qrows), len(g)):
        raise ValueError("Global scores do not match the current gallery")
    replay = Replay(kd, g, split) if kd else None
    gallery_labels = g.cuda()
    weights = photos.new_tensor([1 / 18] * 9 + [1 / 8] * 4)
    result, details, errors = defaultdict(list), defaultdict(list), {}
    with (ThreadPoolExecutor(max_workers=workers) if replay else nullcontext()) as pool:
        for start in range(0, len(y), 48):
            loc = slice(start, start + 48)
            qlabels = y[loc]
            gpu_y = qlabels.cuda()
            full = hist[loc].cuda().clamp_min(1e-30).log()
            take = torch.arange(len(full))
            chosen = actions(full, labels, channels, fixed, names)
            # In a real single-action query only the selected crop is read.
            # The evaluator opens all crops after deciding, to measure every counterfactual.
            local = single_region_scores(native[qrows[loc]].cuda().float(), src, heads, compact, terms, channels, labels, scale)

            def measure(score, code):
                fast, order, rel, _ = map_values(score, gpu_y, gallery_labels, split)
                if replay is None:
                    return fast.cpu(), None, None
                actual = replay.score(order, qlabels, pool)
                error = float((fast.cpu() - actual).abs().max())
                if error > 2e-6:
                    raise ValueError("KD-SBIR per-query mismatch")
                errors[code] = max(errors.get(code, 0.), error)
                return actual, order[:, :25].int().cpu(), rel[:, 0].cpu()

            if replay:
                base, _, initial = measure(full, "global")
                result["global"].append(base)
                details["global_top1"].append(initial)
            for reader, mix in readers:
                relevant = [k for k in keys if unpack(k)[3:] == (reader, mix)]
                primary_reader = primary is not None and primary_name[3:] == (reader, mix)
                action_metrics, action_top, top_correct = [], [], []
                for region in range(13):
                    measured, top, correct = measure((1 - mix) * full + mix * local[reader][:, region], f"{reader}/{mix:g}/{region}")
                    action_metrics.append(measured)
                    if replay and primary_reader:
                        action_top.append(top)
                        top_correct.append(correct)
                metrics = torch.stack(action_metrics, 1)
                for k in relevant:
                    method, temp, rivals, _, _ = unpack(k)
                    plan = chosen[temp, rivals]
                    which = plan[method].cpu()
                    result[k].append(metrics[take, which])
                if replay and primary_reader:
                    _, t, n, _, _ = primary_name
                    plan = chosen[t, n]
                    ix = plan["heldout"].cpu()
                    expectation = torch.where(plan["large"].cpu()[:, None], metrics[:, 9:].mean(1), metrics[:, :9].mean(1))
                    result["random_at_selected_scale"].append(expectation)
                    result["uniform_random_action"].append((metrics * weights.cpu()[None, :, None]).sum(1))
                    result["oracle_one_action"].append(metrics[take, metrics[:, :, 0].argmax(1)])
                    fixed_score = torch.logsumexp(local["photo"] + weights.log()[None, :, None], 1)
                    pooled, _, _ = measure((1 - mix) * full + mix * fixed_score, "all13")
                    result["all13"].append(pooled)
                    details["per_action_metrics"].append(metrics)
                    details["actions"].append(ix)
                    details["top25"].append(torch.stack(action_top, 1)[take, ix])
                    details["new_top1"].append(torch.stack(top_correct, 1)[take, ix])
                    agree, n_pairs = concordance(plan["predicted"].cpu(), metrics[:, :, 0])
                    details["concordance"].append(torch.stack((agree, n_pairs.float()), 1))
    return {k: torch.cat(v) for k, v in result.items()}, {k: torch.cat(v) for k, v in details.items()}, errors, fixed.tolist()


@torch.inference_mode()
def run(w, phase, workers):
    root = w / "results" / FAMILY
    pilot = read(root / "protocol.json")
    if not pilot["revision"].startswith(PILOT):
        raise ValueError("Unexpected pilot source")
    pilot_means = read(root / "pilot.json")["means"]
    shortlist = sorted(set(best(pilot_means, "heldout", 3) + best(pilot_means, "information", 2)
                           + best(pilot_means, "within", 2) + best(pilot_means, "fixed", 1)))
    spec = {"revision": revision(), "pilot_sha256": checksum(root / "pilot.json"), "settings": shortlist,
            "primary_family": "heldout", "controls": ["within", "information", "fixed", "fixed_scale", "misplaced", "scale-matched random"]}
    signature = digest(spec)
    record = root / "full-protocol.json"
    if record.exists() and read(record)["sha256"] != signature:
        raise ValueError("Follow-up protocol has changed")
    atomic_json(record, spec | {"sha256": signature})
    kd, _ = load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("Unknown KD-SBIR evaluator")
    stages = {"lock": ("shortlist",), "audit": ("final",)}[phase]
    for stage in stages:
        if stage == "shortlist":
            keys = shortlist
            primary = None
        else:
            lock = read(root / "freeze.json")
            if lock["signature"] != signature or lock["full_sketchy_sha256"] != checksum(root / "shortlist.json"):
                raise ValueError("Full Sketchy selection changed")
            primary = lock["primary"]
            keys = sorted({primary, lock["within"], lock["information"], lock["fixed"],
                           *[replace_policy(primary, k) for k in ("within", "information", "fixed", "fixed_scale", "misplaced")]})
        table = {}
        for ds in (("sketchy", "tuberlin") if stage == "final" else ("sketchy",)):
            cfg, _, qs, pp, galleries, _, praw = global_inputs(w, ds)
            words = read_words(cfg.vocabulary)
            if len(words) != 20931:
                raise ValueError("Wrong public vocabulary")
            src, tgt = (v.cuda() for v in load_words(cfg, words))
            _, scale = bare_words(cfg, words)
            native = torch.load(w / "cache" / VIEWS / ds / "views.pt", weights_only=True, mmap=True)["original_regions"]
            zooms = torch.load(w / "cache/nested-scale-word-evidence" / ds / "photo.pt", weights_only=True, mmap=True)
            for split, (qi, y) in qs.items():
                pi, g = galleries[split]
                if (len(qi), len(pi), len(g.unique())) != SIZES[split] or digest(UNSEEN_CLASSES[split]) != SPLIT_HASHES[split]:
                    raise ValueError("Unexpected split")
                file = root / ((split if stage == "final" else "shortlist-" + split) + ".json")
                if file.exists():
                    old = read(file)
                    if old["signature"] != signature or (stage == "final" and old["per_query_sha256"] != checksum(file.with_suffix(".pt"))):
                        raise ValueError("Stale split result")
                    table[split] = old["values"]
                    continue
                rows, details, errors, fixed = evaluate(w, cfg, split, qi, y, praw[pi].cuda(), g, pp, pi, native, zooms[pi], src, tgt, scale,
                                                      root / "protocol.json", keys, primary, kd if stage == "final" else None, workers)
                values = {k: v.mean(0).tolist() for k, v in rows.items()}
                info = {"signature": signature, "queries": len(qi), "gallery": len(pi), "values": values, "fixed_windows": fixed}
                if stage == "final":
                    target = file.with_suffix(".pt")
                    save_tensor({"metrics": rows, "query_labels": y} | details, target)
                    concord = details["concordance"]
                    ok, new = details["global_top1"], details["new_top1"]
                    info.update({"protocol": {"split": split, "queries": len(qi), "gallery": len(pi), "classes": len(g.unique()),
                                               "checkpoint": CHECKPOINT, "revision": spec["revision"], "vocabulary_sha256": checksum(cfg.vocabulary),
                                               "kd_sha256": KD_FUNCTION_SHA, "gpu": torch.cuda.get_device_name(),
                                               "map_k": 200 if split == "sketchy_2" else 0, "p_k": 200 if split == "sketchy_2" else 100},
                                 "primary": primary, "best_contrast": lock["fixed"], "matched_base": replace_policy(primary, "fixed_scale"),
                                 "baseline": "global", "scores": {k: dict(zip(("mAP", "precision"), v)) for k, v in values.items()},
                                 "vs_controls": {k: paired_delta(rows[primary], v, y) for k, v in rows.items()},
                                 "original_kd_queries_per_condition": len(qi), "kd_max_abs_error": errors, "per_query_sha256": checksum(target),
                                 "top1_changes": {"fixed_misses": int((~ok & new).sum()), "broken_hits": int((ok & ~new).sum()),
                                                  "global_misses": int((~ok).sum()), "global_hits": int(ok.sum())},
                                 "concordance": float(concord[concord[:, 1] > 0, 0].mean()), "large_zoom_share": float((details["actions"] >= 9).float().mean())})
                atomic_json(file, info)
                table[split] = values
                print(stage, split, primary or "best primary", values[primary or best(values, "heldout", 1)[0]], flush=True)
        if stage == "shortlist":
            mean = {k: [sum(table[s][k][i] for s in SPLITS) / len(SPLITS) for i in (0, 1)] for k in keys}
            atomic_json(root / "shortlist.json", {"splits": table, "means": mean})
            lock = {name: best(mean, name, 1)[0] for name in ("heldout", "information", "within", "fixed")}
            lock["primary"] = lock.pop("heldout")
            lock.update({"revision": spec["revision"], "signature": signature, "full_sketchy_sha256": checksum(root / "shortlist.json")})
            atomic_json(root / "freeze.json", lock | {"sha256": digest(lock)})
            print("locked", lock, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--phase", required=True, choices=("lock", "audit"))
    parser.add_argument("--workers", type=int, default=10)
    params = parser.parse_args()
    run(params.workspace, params.phase, params.workers)
