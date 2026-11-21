"""Run the fixed concept-agreement experiment on Colab and replay KD-SBIR ranks."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import os
from pathlib import Path
import time

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_concept_agreement import ConceptAgreement
from .conditional_semantic_index import SIZES, bare_words
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import load_kd, paired_delta
from .semantic import read_words
from .shared_concept_likelihood import emission, target_inputs


FAMILY = "conditional-concept-agreement"
CANDIDATES = ("conditional", "relative")
SPLITS = ("sketchy_1", "sketchy_2")


class Replay:
    def __init__(self, kd, labels, split):
        self.kd, self.labels, self.split = kd, labels, split
        self.cutoffs = (200, 200) if split == "sketchy_2" else (0, 100)
        cosines = torch.linspace(.75, -.75, len(labels))
        self.x = torch.tensor([[1., 0.]])
        self.y = torch.stack((cosines, (1 - cosines.square()).sqrt()), 1)
        realized = (torch.nn.functional.cosine_similarity(self.x, self.y) + 1) / 2
        if not (realized[:-1] > realized[1:]).all():
            raise RuntimeError("The KD-SBIR replay encoding must preserve every rank")

    def chunk(self, args):
        order, labels = args
        rows = []
        for perm, y in zip(order, labels, strict=True):
            ap, precision, mk, pk = self.kd(self.x, self.y, y.view(1), self.labels[perm], self.split)
            if (mk, pk) != self.cutoffs:
                raise ValueError("KD-SBIR returned different cutoffs")
            rows.append(torch.stack((ap, precision)))
        return torch.stack(rows)

    def score(self, order, labels, pool):
        pieces = zip(order.to(device="cpu", dtype=torch.int32).split(16), labels.split(16), strict=True)
        return torch.cat(list(pool.map(self.chunk, pieces)))


def diagnose(scores, agreement, ql, pl, k):
    """Pair diagnostics use labels only after all scoring and ranking."""
    row = torch.arange(len(ql), device=ql.device)
    order = scores["diffusion"].argsort(dim=1, descending=True, stable=True)[:, :k]
    positive = pl[order] == ql[:, None]
    different = positive[:, :-1] != positive[:, 1:]
    first = different.int().argmax(1)
    first_is_positive = positive[row, first].long()
    right = order[row, first + 1 - first_is_positive]
    wrong = order[row, first + first_is_positive]
    seed_top = scores["direct"].argmax(1)
    base_top = order[:, 0]
    harmed = (pl[seed_top] == ql) & (pl[base_top] != ql)
    helped = (pl[seed_top] != ql) & (pl[base_top] == ql)
    delta = agreement[row, right] - agreement[row, wrong]
    gap = agreement[row, seed_top] - agreement[row, base_top]
    return torch.stack((different.any(1), delta, harmed, helped, gap, seed_top, base_top,
                        scores["conditional"].argmax(1), scores["relative"].argmax(1)), 1).cpu()


def diagnosis_report(rows, ql, pl, qpaths, ppaths):
    eligible, delta, harmed, helped, gap, seed, base, candidate, relative = rows.T
    eligible, harmed, helped = eligible.bool(), harmed.bool(), helped.bool()
    summary = {"adjacent_opposite_pairs_in_top_k": int(eligible.sum()),
               "higher_agreement_for_relevant": int((eligible & (delta > 0)).sum()),
               "mean_agreement_difference": float(delta[eligible].mean()),
               "correct_top1_lost_in_diffusion": int(harmed.sum()),
               "wrong_top1_corrected_in_diffusion": int(helped.sum()),
               "lost_with_higher_agreement_at_original_photo": int((harmed & (gap > 0)).sum()),
               "helped_with_higher_agreement_at_new_photo": int((helped & (gap < 0)).sum())}
    for name, top in (("conditional", candidate), ("relative", relative)):
        correct = pl[top.long()] == ql
        summary[name] = {"recovered_from_lost": int((harmed & correct).sum()),
                         "lost_from_helped": int((helped & ~correct).sum()),
                         "rescues_vs_diffusion": int(((pl[base.long()] != ql) & correct).sum()),
                         "losses_vs_diffusion": int(((pl[base.long()] == ql) & ~correct).sum())}
    examples = []
    for group in (harmed & (pl[candidate.long()] == ql), helped & (pl[candidate.long()] != ql)):
        for ix in torch.where(group)[0][:3].tolist():
            examples.append({"sketch": str(qpaths[ix]), "seed": str(ppaths[int(seed[ix])]),
                             "diffusion": str(ppaths[int(base[ix])]), "conditional": str(ppaths[int(candidate[ix])]),
                             "agreement_seed_minus_diffusion": float(gap[ix])})
    summary["examples"] = examples
    return summary


@torch.inference_mode()
def measure(folder, meta, query, photos, source, target, scale, ql, pl, qpaths, ppaths, kd, workers):
    split = meta["split"]
    file = folder / (split + ".json")
    point_file = folder / (split + ".pt")
    if file.exists() and point_file.exists() and read(file)["protocol"] == meta:
        saved = read(file)
        if saved["per_query_sha256"] == checksum(point_file):
            return saved
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    probabilities = target_inputs(photos, target, scale)
    probabilities /= probabilities.sum(1, keepdim=True)
    ix = ConceptAgreement(photos, probabilities, permutation_control=True)
    torch.cuda.synchronize()
    index_seconds = time.perf_counter() - start
    replay = Replay(kd, pl, split)
    print(split, "index built", round(index_seconds, 1), "s", flush=True)
    rows, errors, diagnoses = {}, {}, []
    gl = pl.cuda()
    scoring_secs = 0.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(0, len(ql), 192):
            y = ql[i:i+192]
            before = time.perf_counter()
            scores, agreement = ix.scores(emission(query[i:i+192], source, scale))
            torch.cuda.synchronize()
            scoring_secs += time.perf_counter() - before
            if not bool(torch.isfinite(agreement).all()) or float(agreement.max()) > 1.005:
                raise ValueError("Conditional agreement violates its probability bound")
            diagnoses.append(diagnose(scores, agreement, y.cuda(), gl, replay.cutoffs[1]))
            for name, matrix in scores.items():
                if not torch.isfinite(matrix).all():
                    raise ValueError("Nonfinite " + name)
                calculated, order, _, _ = map_values(matrix, y.cuda(), gl, split)
                official = replay.score(order, y, pool)
                err = float((official - calculated.cpu()).abs().max())
                if err > 2e-6:
                    raise ValueError(f"Replay differs on {split} {name}: {err}")
                errors[name] = max(errors.get(name, 0), err)
                rows.setdefault(name, []).append(official)
            if i % 1920 == 0:
                print(split, i + len(y), "/", len(ql), flush=True)
    values = {name: torch.cat(parts) for name, parts in rows.items()}
    if any(v.shape != (len(ql), 2) for v in values.values()):
        raise RuntimeError("Incomplete evaluation")
    diagnostics = torch.cat(diagnoses)
    save_tensor({"metrics": values, "labels": ql, "diagnostics": diagnostics}, point_file)
    result = {"protocol": meta, "map_k": replay.cutoffs[0], "p_k": replay.cutoffs[1],
              "scores": {name: dict(zip(("mAP","precision"), x.mean(0).tolist())) for name,x in values.items()},
              "vs_diffusion": {name: paired_delta(v, values["diffusion"], ql) for name, v in values.items()},
              "vs_controls": {name: {c: paired_delta(values[name], values[c], ql)
                  for c in ("word_root", "scalar_root", "flat_bias", "permuted_"+name)} for name in CANDIDATES},
              "diagnosis": diagnosis_report(diagnostics, ql, pl, qpaths, ppaths),
              "kd_queries_per_configuration": len(ql), "kd_max_abs_error": errors,
              "index_seconds": index_seconds, "all_scores_seconds": scoring_secs,
              "elapsed_seconds": time.perf_counter()-start, "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
              "per_query_sha256": checksum(point_file)}
    atomic_json(file, result)
    print(split, result["scores"], flush=True)
    return result


def freeze(folder):
    records = [read(folder / (s + ".json")) for s in SPLITS]
    if any(r["protocol"]["revision"] != revision() for r in records):
        raise ValueError("Sketchy results from different revisions")
    means = {k: [sum(r["scores"][k][m] for r in records)/2 for m in ("mAP", "precision")]
             for k in (*CANDIDATES, "diffusion")}
    choice = max(CANDIDATES, key=means.get)
    lock = {"revision": revision(), "choice": choice, "means": means,
            "selector": "equal Sketchy mAP, then precision, among the two prospective candidates",
            "source_sha256": {s: checksum(folder / (s + ".json")) for s in SPLITS}}
    signature = digest(lock)
    p = folder / "freeze.json"
    if p.exists() and read(p).get("sha256") != signature:
        raise ValueError("Existing decision refers to another experiment")
    atomic_json(p, lock | {"sha256": signature})
    print("Sketchy selection", choice, means, flush=True)
    return signature


def run(w, workers):
    f = w / "results" / FAMILY
    kd, _ = load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("KD-SBIR function checksum mismatch")
    for dataset in ("sketchy", "tuberlin"):
        selection = freeze(f) if dataset == "tuberlin" else None
        cfg, qpaths, queries, ppaths, galleries, qraw, praw = global_inputs(w, dataset)
        words = read_words(cfg.vocabulary)
        if len(words) != 20931:
            raise ValueError("Open Images vocabulary mismatch")
        _, scale = bare_words(cfg, words)
        src, tgt = (v.cuda() for v in load_words(cfg, words))
        for split, (qi, ql) in queries.items():
            pi, pl = galleries[split]
            if (len(qi), len(pi), len(pl.unique())) != SIZES[split] or digest(UNSEEN_CLASSES[split]) != SPLIT_HASHES[split]:
                raise ValueError("Incomplete evaluation split " + split)
            meta = {"family": FAMILY, "split": split, "revision": revision(), "queries": len(ql), "gallery":len(pl),
                    "class_list_sha256": SPLIT_HASHES[split], "checkpoint": CHECKPOINT, "scale":scale,
                    "vocabulary_sha256":checksum(cfg.vocabulary), "kd_sha256":KD_FUNCTION_SHA,
                    "gpu":torch.cuda.get_device_name(), "selection_sha256":selection}
            measure(f, meta, qraw[qi].cuda(), praw[pi].cuda(), src, tgt, scale, ql, pl,
                    [qpaths[i] for i in qi.tolist()], [ppaths[i] for i in pi.tolist()], kd, workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--workers", type=int, default=min(10, len(os.sched_getaffinity(0))))
    with torch.inference_mode():
        args = parser.parse_args()
        run(args.workspace, args.workers)
