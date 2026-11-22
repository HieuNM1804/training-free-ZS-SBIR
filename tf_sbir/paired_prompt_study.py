"""Test whether invariance to a noun's depiction prompt repairs frozen retrieval."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import inspect
from pathlib import Path
import time

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .paired_prompt_inputs import prepare
from .paired_prompt_intervention import (
    center, fit_panels, holdout_measure, intervention, preserve_spread, reference_bases, text_reference,
)
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs, stratified_sample
from .retrieval_evidence import load_kd, paired_delta
from .semantic import read_words

FAMILY = "paired-prompt-head-intervention"
WIDTHS = (64, 192)
RIDGES = (.1, 1.)
STRENGTHS = (.3, 3., 12.)
FRACTIONS = (.35, 1.)
TYPES = ("heads", "rotate", "text", "exchange", "unshrunk", "opposite")
MIDPOINTS = (.1, .3, .6, 1.)
COLUMNS = ("unseen_word_gap_variance_removed", "unseen_word_content_correlation",
           "unseen_word_content_relative_change", "readout_l2_change", "mean_crossfit_multiplier")


def cell(kind, width, ridge, strength, fraction):
    return f"{kind}/{width}/{ridge:g}/{strength:g}/{fraction:g}"


def log_readout(logits, bank):
    evidence = (logits - logits.amax(1, keepdim=True)).exp()
    return (evidence @ bank.T).clamp_min_(1e-30).log_()


def label_diagnostic(public_names, categories):
    def normalized(name):
        return " ".join(name.casefold().replace("_", " ").replace("-", " ").split())
    public = defaultdict(list)
    for i, word in enumerate(public_names):
        public[normalized(word)].append(i)
    return torch.tensor([public[normalized(k)][0] if len(public[normalized(k)]) == 1 else -1
                         for k in categories], device="cuda")


def literal_rank(logits, ids):
    safe = ids.clamp_min(0)
    score = logits.gather(1, safe[:, None])
    pos = (logits > score).sum(1) + 1
    return torch.where(ids >= 0, pos, 0)


class Study:
    def __init__(self, w):
        self.w = w
        cfg, *inputs = global_inputs(w, "sketchy")
        queries = inputs[1]
        samples = {s: stratified_sample(labels, 36, 63241) for s, (_, labels) in queries.items()}
        names = read_words(cfg.vocabulary)
        self.public_names = names
        if len(names) != 20931:
            raise ValueError("The public name bank changed")
        self.vocabulary = checksum(cfg.vocabulary)
        self.source, self.target = [v.cuda() for v in load_words(cfg, names)]
        _, self.scale = bare_words(cfg, names)
        self.midpoint = (self.source + self.target) * .5
        self.axes = text_reference(self.source, self.target, 48)
        rng = torch.Generator().manual_seed(8129)
        self.perm = torch.randperm(self.source.size(1), generator=rng).cuda()
        self.sign = (2 * torch.randint(2, (self.source.size(1),), generator=rng) - 1).float().cuda()
        self.data, self.positions, self.part_record = prepare(
            w, FAMILY, "sketchy", cfg, inputs, self.source, self.target, self.scale, samples)

    def embeddings(self, q, heads, keep):
        nominee = (q @ self.midpoint.T).topk(max(WIDTHS) + 128).indices
        held = nominee[:, max(WIDTHS):]
        m_test = self.midpoint[held]
        d_test = self.source[held] - self.target[held]
        bases = reference_bases(heads, self.axes, self.perm, self.sign)
        for width in WIDTHS:
            names = nominee[:, :width]
            midpoint = self.midpoint[names]
            for base_kind, basis in bases.items():
                kinds = ("heads", "unshrunk", "opposite") if base_kind == "heads" else (base_kind,)
                candidates = [key for key in keep if key.split("/")[0] in kinds
                              and len(key.split("/")) == 5 and key.split("/")[1] == str(width)] if keep else None
                if keep and not candidates:
                    continue
                panels = fit_panels(q, basis, self.source[names], self.target[names], base_kind == "exchange")
                for ridge in RIDGES:
                    for strength in STRENGTHS:
                        plan = [(kind, f, cell(kind, width, ridge, strength, f))
                                for kind in kinds for f in FRACTIONS]
                        if keep:
                            plan = [part for part in plan if part[2] in keep]
                        if not plan:
                            continue
                        step, raw, shrink = intervention(basis, panels, ridge, strength)
                        for kind, frac, key in plan:
                            delta = raw if kind == "unshrunk" else (-step if kind == "opposite" else step)
                            changed = preserve_spread(q, frac * delta, midpoint)
                            measured = holdout_measure(q, changed, m_test, d_test)
                            multiplier = torch.ones_like(shrink) if kind == "unshrunk" else shrink
                            yield key, changed, torch.cat((measured, multiplier[:, None]), 1)

    def batches(self, q, heads, base, bank, keep):
        source = q @ self.source.T * self.scale
        raw = log_readout(source, bank)
        yield "regional", base, None, source
        for amount in MIDPOINTS:
            name = f"mix/{amount:g}"
            if keep and name not in keep:
                continue
            shifted = torch.lerp(source, q @ self.midpoint.T * self.scale, amount)
            yield name, base + .65 * (log_readout(shifted, bank) - raw), None, shifted
        vectors, keys, measures = [], [], []
        def flush():
            # Keep GPU matrix dimensions identical in the subsequent selective KD replay.
            padded = vectors + [vectors[-1]] * (8 - len(vectors))
            logits = torch.cat(padded) @ self.source.T * self.scale
            log_scores = log_readout(logits, bank).split(len(q))[:len(keys)]
            logits = logits.split(len(q))[:len(keys)]
            for label, values, details, word_scores in zip(keys, log_scores, measures, logits, strict=True):
                yield label, base + .65 * (values - raw), details, word_scores
            vectors.clear()
            keys.clear()
            measures.clear()
        for name, vec, obs in self.embeddings(q, heads, keep):
            vectors.append(vec)
            keys.append(name)
            measures.append(obs)
            if len(keys) == 8:
                yield from flush()
        if keys:
            yield from flush()

    def evaluate(self, split, ids, root, sig, keep=None, kd=None, pool=None):
        data, y, g = self.data[split]
        bank = torch.load(self.w / "cache/pose-conditional-layout" / (split+"-gallery-uniform.pt"),
                          weights_only=True, mmap=True).cuda()
        old = torch.load(self.w / "cache/pragmatic-gallery-competition" / (split+"-named-rgb16.pt"),
                         weights_only=True, mmap=True)
        gr = g.cuda()
        replayer = Replay(kd, g, split) if kd else None
        metrics, diagnostics, correct = defaultdict(list), defaultdict(list), defaultdict(list)
        literal = defaultdict(list)
        class_indices = label_diagnostic(self.public_names, UNSEEN_CLASSES[split])
        errors, checks = {}, {}
        for offset, block in enumerate(ids.split(32)):
            q = data["query_native"][block].cuda().float()
            slot = self.positions[split][block]
            if (slot < 0).any():
                raise ValueError("This sketch was not included in the encoded sample")
            base = data["regional_score"][slot].cuda().float()
            heads = data["selected_heads"][slot].cuda().float()
            if offset == 0:
                columns = torch.linspace(0, len(gr)-1, 137, device="cuda").long()
                prior = old[block][:, columns.cpu()].cuda().clamp_min(1e-30).log()
                direct = log_readout(q @ self.source.T * self.scale, bank)[:, columns]
                err = (center(prior)-center(direct)).abs().max().item()
                midpoint = self.midpoint[:20].expand(3, -1, -1)
                # Identical prompts imply zero centered domain response and exactly zero intervention.
                panels = fit_panels(q[:3], heads[:3], midpoint, midpoint)
                zero_step = intervention(heads[:3], panels, .1, 3.)[0].abs().max().item()
                checks = {"max_centered_historical_log_error": err, "zero_difference_edit": zero_step}
                if err > .008 or zero_step > 1e-6:
                    raise ValueError(f"The frozen lexical operator or null check diverged: {checks}")
            truth = y[block].cuda()
            for name, score, obs, word_scores in self.batches(q, heads, base, bank, keep):
                value, order, target, _ = map_values(score, truth, gr, split)
                value = value.cpu()
                if replayer:
                    actual = replayer.score(order, y[block], pool)
                    error = (value - actual).abs().max().item()
                    if error > 2e-6:
                        raise ValueError(f"KD metric mismatch {split} {name} {error}")
                    errors[name] = max(errors.get(name, 0.), error)
                    value = actual
                metrics[name].append(value)
                correct[name].append(target[:, 0].cpu())
                if not replayer:
                    literal[name].append(literal_rank(word_scores, class_indices[truth]).cpu())
                if obs is not None:
                    diagnostics[name].append(obs.cpu())
            if offset % 8 == 0:
                print(split, "KD" if kd else "search", offset*32, len(ids), flush=True)
        scores = {k: torch.cat(v) for k, v in metrics.items()}
        evidence = {k: torch.cat(v) for k, v in diagnostics.items()}
        top = {k: torch.cat(v) for k, v in correct.items()}
        if replayer:
            return scores, errors
        values = {k: v.mean(0).tolist() for k, v in scores.items()}
        word_ranks = {k: torch.cat(v) for k,v in literal.items()}
        missing_top = (~top["regional"]) & (word_ranks["regional"] > 0)
        report = {}
        for k, diag in evidence.items():
            delta = scores[k][:, 0] - scores["regional"][:, 0]
            valid = diag[:, 0].std() > 1e-7 and delta.std() > 1e-7
            report[k] = dict(zip(COLUMNS, diag.mean(0).tolist(), strict=True))
            report[k].update({"gap_reduction_AP_correlation": float(torch.corrcoef(torch.stack((diag[:, 0], delta)))[0, 1]) if valid else None,
                              "top1_fixed": int((top[k] & ~top["regional"]).sum()),
                              "top1_broken": int((~top[k] & top["regional"]).sum()),
                              "base_misses_with_exact_public_name": int(missing_top.sum()),
                              "literal_name_up_on_misses": int((missing_top & (word_ranks[k] < word_ranks["regional"])).sum()),
                              "literal_name_down_on_misses": int((missing_top & (word_ranks[k] > word_ranks["regional"])).sum())})
        path = root / ("pilot-"+split+".pt")
        save_tensor({"metrics": scores, "query_ids": ids, "query_labels": y[ids],
                     "measurements": evidence, "top1_correct": top, "posthoc_literal_ranks": word_ranks}, path)
        atomic_json(path.with_suffix(".json"), {"signature": sig, "queries": len(ids), "gallery": len(g),
                    "values": values, "checks": checks, "heldout": report, "per_query_sha256": checksum(path)})
        return values


@torch.inference_mode()
def run(w):
    started = time.monotonic()
    study = Study(w)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    kd, _ = load_kd(w)
    if sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError("Unexpected evaluator source")
    root = w / "results" / FAMILY
    spec = {"revision": revision(), "checkpoint": CHECKPOINT, "vocabulary": study.vocabulary,
            "parts": study.part_record["signature"], "widths": WIDTHS, "ridges": RIDGES,
            "strengths": STRENGTHS, "fractions": FRACTIONS, "midpoints": MIDPOINTS, "types": TYPES,
            "seed": 63241, "per_class": 36, "heldout_word_ranks": [193, 320], "evaluator": KD_FUNCTION_SHA}
    sig = digest(spec)
    if (root/"protocol.json").is_file() and read(root/"protocol.json")["sha256"] != sig:
        raise ValueError("The stored protocol belongs to a different study")
    atomic_json(root/"protocol.json", spec | {"sha256": sig})
    samples = {s: stratified_sample(y, 36, 63241) for s, (_, y, _) in study.data.items()}
    tables = {}
    for split, ids in samples.items():
        path = root/("pilot-"+split+".json")
        if path.is_file():
            record = read(path)
            if record["signature"] != sig or record["per_query_sha256"] != checksum(path.with_suffix(".pt")):
                raise ValueError("Incomplete pilot checkpoint")
            tables[split] = record["values"]
        else:
            tables[split] = study.evaluate(split, ids, root, sig)
    avg = {k: [sum(table[k][i] for table in tables.values())/len(tables) for i in (0, 1)]
           for k in tables["sketchy_1"]}
    best = {kind: max((k for k in avg if k.startswith(kind+"/")), key=avg.get) for kind in (*TYPES, "mix")}
    primary = best["heads"]
    suffix = primary[primary.index("/"):]
    matched = ["regional"] + [kind+suffix for kind in TYPES if kind != "heads"] + [best["mix"]]
    retained = sorted(set([primary, *matched, *best.values()]))
    checked, differences, audited = {}, {}, {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for split, ids in samples.items():
            actual, errors = study.evaluate(split, ids, root, sig, set(retained), kd, pool)
            stored = torch.load(root/("pilot-"+split+".pt"), weights_only=True)
            max_error = max((actual[k] - stored["metrics"][k]).abs().max().item() for k in retained)
            if max_error > 2e-6:
                raise ValueError(f"Replayed pilot differs {max_error}")
            checked[split] = {"original_KD_queries_per_condition": len(ids), "max_errors": errors, "replay_error": max_error}
            audited[split] = {k: v.mean(0).tolist() for k, v in actual.items()}
            differences[split] = {k: paired_delta(actual[primary], actual[k], stored["query_labels"]) for k in retained}
    atomic_json(root/"pilot.json", {"signature": sig, "nominee": primary, "best_by_family": best,
                "means": avg, "matched": matched, "KD": checked, "audited": audited, "paired": differences,
                "device": torch.cuda.get_device_name(), "elapsed_seconds_including_input_preparation": time.monotonic()-started,
                "peak_allocated_scoring_GiB": torch.cuda.max_memory_allocated()/2**30})
    print("audited", primary, avg[primary], "regional", avg["regional"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
