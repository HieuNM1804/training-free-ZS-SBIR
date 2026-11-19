"""Validate local full-gallery results and compute paired query and category bootstrap intervals."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torchmetrics.functional.retrieval import retrieval_average_precision, retrieval_precision

from .benchmark import merge_inputs
from .capture_query_metrics import evaluate_with_details
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES, collect_paths
from .encoding import global_cache_name
from .metrics import cosine_rows
from .transfer_validation import include_candidate, load_query_results, run as validate_classes


def verify_reference(root, cache, checkpoint, kd_dir, *, full=False):
    config_text = (kd_dir / "src/data_config.py").read_text(encoding="utf-8-sig")
    node = next(n for n in ast.parse(config_text).body if isinstance(n, ast.Assign) and n.targets[0].id == "UNSEEN_CLASSES")
    classes = ast.literal_eval(node.value)
    model_file = kd_dir / "src/model.py"
    definition = next(n for n in ast.parse(model_file.read_text(encoding="utf-8-sig")).body
                      if isinstance(n, ast.FunctionDef) and n.name == "_retrieval_metrics")
    ns = {"torch": torch, "F": F, "retrieval_average_precision": retrieval_average_precision,
          "retrieval_precision": retrieval_precision}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(model_file), "exec"), ns)
    banks = []
    for mode in ("sketch", "photo"):
        paths, index = merge_inputs(root, SKETCHY_SPLITS, mode)
        file = cache / global_cache_name(root, paths, mode, "rgb", "fp16", str(checkpoint))
        banks.append((torch.load(file, map_location="cpu", weights_only=True, mmap=True), index))
    data = {}
    for split in SKETCHY_SPLITS:
        if list(UNSEEN_CLASSES[split]) != classes[split]:
            raise ValueError(f"Different class order for {split}")
        path_hashes = {}
        for mode in ("sketch", "photo"):
            expected = [str(p) for c in classes[split] for p in sorted((root / mode / c).glob("*"))]
            actual, _ = collect_paths(root, UNSEEN_CLASSES[split], mode)
            if actual != expected:
                raise ValueError(f"Gallery or sketch paths differ for {split} {mode}")
            names = (Path(p).relative_to(root).as_posix() for p in actual)
            path_hashes[mode] = hashlib.sha256("\n".join(names).encode()).hexdigest()
        sketches, q_index = banks[0]
        photos, p_index = banks[1]
        ids, labels = q_index[split]
        p_ids, p_labels = p_index[split]
        if full:
            selected = torch.arange(len(ids))
        else:
            selected = []
            for c in sorted(labels.unique().tolist()):
                group = (labels == c).nonzero().flatten()
                selected.extend((int(group[0]), int(group[len(group) // 2]), int(group[-1])))
        q, p, labels = sketches[ids[selected]].cuda(), photos[p_ids].cuda(), labels[selected]
        recorded = {}
        def recording_metric(fn):
            arr = []
            recorded[fn.__name__] = arr
            def wrapped(*args, **kwargs):
                value = fn(*args, **kwargs)
                arr.append(value.detach())
                return value
            return wrapped
        for fn in (retrieval_average_precision, retrieval_precision):
            ns[fn.__name__] = recording_metric(fn)
        original = ns["_retrieval_metrics"](q, p, labels, p_labels, split)
        expected_rows = torch.stack([torch.stack(recorded[fn.__name__]) for fn in (retrieval_average_precision, retrieval_precision)], dim=1)
        current, actual_rows = evaluate_with_details(cosine_rows(q, p, 1), labels, p_labels, split, f"{split} KD parity", 4)
        if (not torch.equal(actual_rows, expected_rows) or
                not all(torch.equal(a, b) for a, b in zip(current[:2], original[:2])) or current[2:] != original[2:]):
            raise ValueError(f"Reference metric parity failed for {split}")
        data[split] = {"classes": len(classes[split]), "queries_checked": len(q), "all_queries": len(q) == len(ids),
                       "full_gallery": len(p), "bitwise_equal": True, "per_query_values_checked": actual_rows.numel(),
                       "mAP": float(original[0]), "precision": float(original[1]), "ordered_paths_sha256": path_hashes}
    return {"reference_file_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(), "splits": data}


def rankings(rows):
    return max((n for n in rows if include_candidate(n)), key=lambda n: (rows[n]["precision"], rows[n]["mAP"]))


def intervals(rows, labels, pairs, generator):
    changes = torch.stack([rows[a] - rows[b] for a, b in pairs], dim=1).cuda()
    labels = labels.cuda()
    groups = len(labels.unique())
    totals = changes.new_zeros((groups, *changes.shape[1:])).index_add_(0, labels, changes)
    sizes = torch.bincount(labels).float()
    selected = torch.randint(groups, (5000, groups), device="cuda", generator=generator)
    category = totals[selected].sum(dim=1) / sizes[selected].sum(dim=1)[:, None, None]
    samples = []
    for i in range(0, 5000, 16):
        chosen = torch.randint(len(labels), (min(16, 5000-i), len(labels)), device="cuda", generator=generator)
        samples.append(changes[chosen].mean(dim=1))
    quantiles = changes.new_tensor([.025, .975])
    q_limits = torch.quantile(torch.cat(samples), quantiles, dim=0).permute(1, 2, 0).tolist()
    c_limits = torch.quantile(category, quantiles, dim=0).permute(1, 2, 0).tolist()
    means = changes.mean(dim=0).tolist()
    return [{"method": a, "reference": b, "delta_precision": means[i][1], "delta_map": means[i][0],
             "query_ci_precision": q_limits[i][1], "class_ci_precision": c_limits[i][1],
             "query_ci_map": q_limits[i][0], "class_ci_map": c_limits[i][0]} for i, (a, b) in enumerate(pairs)]


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    results = {s: {} for s in SKETCHY_SPLITS}
    per_query = {s: {} for s in SKETCHY_SPLITS}
    evidence = {"families": {}, "draws": 5000, "seed": 907}
    for study in args.studies:
        values = load_query_results(study)
        if set(values[SKETCHY_SPLITS[0]]) != set(values[SKETCHY_SPLITS[1]]):
            raise ValueError(f"Incomplete paired splits in {study}")
        groups = {}
        for split in SKETCHY_SPLITS:
            group = {}
            for name, details in values[split].items():
                row = json.loads((study / f"{split}-{name}.json").read_text(encoding="utf-8-sig"))
                means = details.mean(dim=0)
                expected = (14179, 17100, 100, 0) if split == "sketchy_1" else (12694, 12552, 200, 200)
                actual = (row["queries"], row["gallery"], row["p_k"], row["map_k"])
                if actual != expected or details.shape != (expected[0], 2) or row["backbone"] != "OpenAI CLIP ViT-B/32":
                    raise ValueError(f"Invalid protocol for {split} {name}")
                visual_only = name.startswith("transfer-image-") or name == "ground-cosine-native"
                if not row["exploratory_unseen"] or row["external_vocabulary"] != (0 if visual_only else 20931):
                    raise ValueError(f"Invalid vocabulary or tuning origin for {name}")
                if row["checkpoint_sha256"] != "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af":
                    raise ValueError(f"Different checkpoint for {name}")
                if not torch.allclose(means, torch.tensor([row["mAP"], row["precision"]]), atol=5e-7, rtol=0):
                    raise ValueError(f"Per-query values differ from the official result for {name}")
                group[name] = row
            if group.keys() & results[split].keys():
                raise ValueError(f"Ambiguous method names in {study.name}")
            groups[split] = group
            results[split].update(group)
            per_query[split].update(values[split])
        by_method = {n: {k: sum(groups[s][n][k] for s in SKETCHY_SPLITS) / 2 for k in ("precision", "mAP")}
                     for n in groups["sketchy_1"]}
        common = rankings(by_method)
        evidence["families"][study.name] = {"variants_per_split": len(by_method), "common": common,
                                            "best_by_split": {s: rankings(groups[s]) for s in SKETCHY_SPLITS},
                                            "common_rows": {s: groups[s][common] for s in SKETCHY_SPLITS}}
    validation_file = args.output.with_name("class-validation.json")
    validate_classes(args.root, args.studies, validation_file)
    validation = json.loads(validation_file.read_text())
    common = validation["all_candidates"]["common"]
    evidence["common"] = common
    evidence["reference_audit"] = verify_reference(args.root, args.cache, args.checkpoint, args.kd, full=args.full_reference)
    generator = torch.Generator(device="cuda").manual_seed(907)
    for split, scores in results.items():
        best = rankings(scores)
        _, labels = collect_paths(args.root, UNSEEN_CLASSES[split], "sketch")
        base = "transfer-two-k64-g1-r1"
        pairs = [(common, base), (best, base), ("transfer-image-plain", "transfer-image-identity"),
                 ("transfer-image-plain", "transfer-image-unpaired"), ("transfer-image-k64-g1-r1", "transfer-image-k64-g0-r1"),
                 ("transfer-risk-only-k64-g1-r1", "transfer-fixed-k64-g1-r1"),
                 ("transfer-risk4-k64-g1-r1", "transfer-cohesion-k64-g1-r1"),
                 ("transfer-risk4-k64-g1-r1", "transfer-shuffled-k64-g1-r1"),
                 ("transfer-channel-p4-l0.5-k64-g1-r1", "transfer-channel-no-risk-k64-g1-r1"),
                 ("transfer-channel-p4-l0.5-k64-g1-r1", "transfer-channel-shuffled-k64-g1-r1"),
                 ("predictable-adaptive-k16-s0.75", "predictable-mean-k16-s0.75"),
                 ("predictable-adaptive-k16-s0.75", "predictable-shuffled-reliability-k16-s0.75"),
                 ("ground-r8-k24-g1", "ground-r8-k24-g0"), ("ground-cosine-paired", "ground-cosine-unpaired"),
                 ("reflect-pairs-intersect025", "reflect-pairs-shuffled"),
                 ("reflect-pairs-agree1", "reflect-pairs-sharp"), ("reflect-pairs-balanced", "reflect-flat-risk2"),
                 ("reflect-pairs-visual-risk-l035", "reflect-pairs-visual-shuffled-l035"),
                 ("reflect-pairs-visual-risk-l035", "reflect-flat-visual-risk-l035"),
                 ("reflect-control-reference", "reflect-flat-visual-risk-l035"),
                 ("reflect-control-reference", "reflect-control-none"),
                 ("reflect-control-reference", "reflect-control-source"),
                 ("reflect-control-reference", "reflect-control-target"),
                 ("reflect-control-reference", "reflect-control-shuffled"),
                 ("reflect-control-sharp", "reflect-control-sharp-source"),
                 ("reflect-control-sharp", "reflect-control-sharp-target"),
                 ("reflect-control-sharp", "reflect-control-sharp-shuffled"),
                 ("reciprocal-predicted-b0.5-p0", "reciprocal-shuffled"),
                 ("reciprocal-predicted-b0.5-p0", "reciprocal-observed-p0"),
                 ("reciprocal-predicted-b0.5-p4", "reciprocal-forward-p4")]
        if "channels" in evidence["families"]:
            channel = evidence["families"]["channels"]["common"]
            pairs.extend([(channel, "transfer-risk-only-k64-g1-r1"), (channel, base)])
        if "grounded-semantic-anchors" in evidence["families"]:
            pairs.append((evidence["families"]["grounded-semantic-anchors"]["common"], "ground-graph-reference"))
        if "reflection" in evidence["families"]:
            winner = evidence["families"]["reflection"]["common"]
            pairs.extend([(winner, "reflect-five-risk-only"), (winner, "reflect-five-risk4")])
        if "reciprocal" in evidence["families"]:
            winner = evidence["families"]["reciprocal"]["common"]
            pairs.extend([(winner, "reciprocal-forward-p0"), (winner, "reciprocal-forward-p4")])
        pairs = list(dict.fromkeys((a, b) for a, b in pairs if a in scores and b in scores and a != b))
        class_scores = {c: per_query[split][common][labels.eq(i)].mean(dim=0).tolist()
                        for i, c in enumerate(UNSEEN_CLASSES[split])}
        evidence[split] = {"common": scores[common], "best": scores[best], "class_scores_ap_precision": class_scores,
                           "paired": intervals(per_query[split], labels, pairs, generator)}
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("root", "cache", "checkpoint", "kd", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--studies", type=Path, nargs="+", required=True)
    parser.add_argument("--full-reference", action="store_true", help="Compare every real sketch's AP and precision to unmodified KD-SBIR.")
    run(parser.parse_args())
