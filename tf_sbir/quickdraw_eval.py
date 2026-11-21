"""Shared full-gallery QuickDraw scoring with the original KD-SBIR evaluator."""

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import inspect
from pathlib import Path
import zipfile

import torch
import torch.nn.functional as F

from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import load_global_inputs, save_tensor
from .foundation import load_words
from .quickdraw_inputs import paths
from .retrieval_evidence import load_kd
from .semantic import read_words
from .shared_concept_likelihood import target_inputs


SPLIT_SHA = "de129ecf410271ecb69ad614a57b4a8ab567bb448876d022e2338f1b2185a89c"
KD_FUNCTION_SHA = "9310e11f47f6fafb54c64060709df8a302165e54e2a8812c97bdb85193c15139"


class OfficialMetrics:
    def __init__(self, function, photo_labels):
        self.function = function
        self.labels = photo_labels
        self.query = torch.tensor([[1.0, 0.0]])
        fixed = torch.linspace(0.75, -0.75, len(photo_labels))
        self.gallery = torch.stack((fixed, (1.0 - fixed.square()).sqrt()), dim=1)
        scores = (F.cosine_similarity(self.query, self.gallery) + 1.0) * 0.5
        if not bool((scores[:-1] > scores[1:]).all()):
            raise ValueError("The metric adapter must preserve every strict rank")

    def __call__(self, orders, labels):
        values = torch.empty(len(labels), 2)
        for i, (order, label) in enumerate(zip(orders, labels, strict=True)):
            ap, precision, map_k, p_k = self.function(
                self.query, self.gallery, label.view(1), self.labels[order], "quickdraw"
            )
            if (map_k, p_k) != (0, 200):
                raise ValueError("KD-SBIR returned unexpected QuickDraw metric cutoffs")
            values[i] = torch.tensor((ap, precision))
        return values


def prepare(workspace, batch_size=448, workers=10):
    if not torch.cuda.is_available():
        raise RuntimeError("Run QuickDraw inference on the Colab GPU")
    classes = UNSEEN_CLASSES["quickdraw"]
    if digest(classes) != SPLIT_SHA:
        raise ValueError("QuickDraw class list differs from the frozen split")

    args, query_paths, query_labels, photo_paths, photo_labels = paths(workspace, batch_size, workers)
    query_features, photo_features, _ = load_global_inputs(args, query_paths, photo_paths, False)
    query_features = query_features.cuda()
    photo_features = photo_features.cuda()
    if (len(query_features), len(photo_features), len(query_labels.unique()), len(photo_labels.unique())) != (
        91491, 56407, 30, 30
    ):
        raise ValueError("Expected the complete 30-class QuickDraw split")

    words = read_words(args.vocabulary)
    if len(words) != 20931:
        raise ValueError("The public vocabulary differs from the frozen experiments")
    _, scale = bare_words(args, words)
    source, target = (value.cuda() for value in load_words(args, words))
    gallery_concepts = target_inputs(photo_features, target, scale)
    gallery_concepts /= gallery_concepts.sum(dim=1, keepdim=True).clamp_min(1e-38)

    function, kd_file = load_kd(workspace)
    kd_sha = hashlib.sha256(inspect.getsource(function).strip().encode()).hexdigest()
    if kd_sha != KD_FUNCTION_SHA:
        raise ValueError("The original KD-SBIR retrieval metric has changed")

    metadata = read(workspace / "quickdraw-data.json")
    return {
        "args": args,
        "queries": query_features,
        "photos": photo_features,
        "query_labels": query_labels,
        "photo_labels": photo_labels,
        "source_words": source,
        "target_words": target,
        "scale": scale,
        "gallery_concepts": gallery_concepts,
        "kd": function,
        "kd_file_sha256": checksum(kd_file),
        "metadata": metadata,
        "vocabulary_sha256": checksum(args.vocabulary),
    }


def validate_lock(method_root, family, expected_revision):
    path = method_root / "docs" / "results" / family / "freeze.json"
    lock = read(path)
    signature = lock.pop("sha256")
    if digest(lock) != signature or lock["revision"] != expected_revision:
        raise ValueError(f"Invalid frozen parameters for {family}")
    return lock, signature


def evaluate_batch(scores, query_labels, photo_labels, metrics, pool):
    if not torch.isfinite(scores).all():
        raise ValueError("Retrieval scores contain non-finite values")
    order = scores.argsort(dim=1, descending=True, stable=True)
    ordered_labels = photo_labels[order]
    matches = ordered_labels == query_labels.cuda()[:, None]
    rank = torch.arange(1, len(photo_labels) + 1, device=scores.device)
    cumulative = matches.cumsum(dim=1) / rank
    ap = (matches * cumulative).sum(dim=1) / matches.sum(dim=1).clamp_min(1)
    p200 = matches[:, :200].float().mean(dim=1)
    calculated = torch.stack((ap, p200), dim=1).cpu()

    host_order = order.to(device="cpu", dtype=torch.int32)
    jobs = [pool.submit(metrics, rank_block, label_block)
            for rank_block, label_block in zip(host_order.split(24), query_labels.split(24), strict=True)]
    official = torch.cat([job.result() for job in jobs])
    error = (official - calculated).abs().amax(dim=0)
    if float(error.max()) > 2e-6:
        raise ValueError("Vectorized scores differ from the original KD-SBIR evaluator")
    return official, error


def score_all(workspace, family, expected_revision, freeze_sha, configuration, inputs, scorer,
              method_source, batch_queries=192, metric_workers=10):
    folder = workspace / "results" / "quickdraw-top5" / family
    folder.mkdir(parents=True, exist_ok=True)
    metadata = inputs["metadata"]
    protocol = {
        "dataset": "quickdraw",
        "family": family,
        "configuration": configuration,
        "method_revision": expected_revision,
        "runner_revision": revision(),
        "method_source_sha256": checksum(method_source),
        "freeze_sha256": freeze_sha,
        "class_list_sha256": SPLIT_SHA,
        "image_manifest_sha256": metadata["image_manifest_sha256"],
        "archive_sha256": metadata["archive_sha256"],
        "checkpoint_sha256": CHECKPOINT,
        "vocabulary_sha256": inputs["vocabulary_sha256"],
        "kd_file_sha256": inputs["kd_file_sha256"],
        "kd_function_sha256": KD_FUNCTION_SHA,
        "metric_cutoffs": {"map_k": 0, "p_k": 200},
        "queries": len(inputs["queries"]),
        "gallery": len(inputs["photos"]),
    }
    bound = digest(protocol)
    values = []
    errors = []
    torch.cuda.reset_peak_memory_stats()
    metrics = OfficialMetrics(inputs["kd"], inputs["photo_labels"])
    gallery_labels_gpu = inputs["photo_labels"].cuda()
    with ThreadPoolExecutor(max_workers=metric_workers) as pool:
        for begin in range(0, len(inputs["queries"]), batch_queries):
            end = min(begin + batch_queries, len(inputs["queries"]))
            shard = folder / "shards" / f"{begin:06d}-{end:06d}.pt"
            if shard.is_file():
                saved = torch.load(shard, map_location="cpu", weights_only=True)
                if saved["protocol"] != bound or saved["official"].shape != (end - begin, 2):
                    raise ValueError("Found a QuickDraw shard from a different protocol")
            else:
                scores = scorer(inputs["queries"][begin:end])
                official, error = evaluate_batch(
                    scores,
                    inputs["query_labels"][begin:end],
                    gallery_labels_gpu,
                    metrics,
                    pool,
                )
                saved = {"protocol": bound, "official": official, "max_AP_P_error": error}
                save_tensor(saved, shard)
                del scores
            values.append(saved["official"])
            errors.append(saved["max_AP_P_error"])
            if end == len(inputs["queries"]) or end % (batch_queries * 10) == 0:
                print(f"{family} QuickDraw {end}/{len(inputs['queries'])} queries", flush=True)

    observed = torch.cat(values)
    if observed.shape != (len(inputs["query_labels"]), 2) or not torch.isfinite(observed).all():
        raise ValueError("QuickDraw result is incomplete or non-finite")
    per_query = folder / "per-query.pt"
    save_tensor({"AP": observed[:, 0], "P200": observed[:, 1],
                 "query_labels": inputs["query_labels"], "gallery_labels": inputs["photo_labels"]}, per_query)
    per_class = folder / "per-class.csv"
    with per_class.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("class", "queries", "gallery", "mAP_all", "P200"))
        for class_id, name in enumerate(UNSEEN_CLASSES["quickdraw"]):
            mask = inputs["query_labels"] == class_id
            means = observed[mask].mean(dim=0).tolist()
            writer.writerow((name, int(mask.sum()), int((inputs["photo_labels"] == class_id).sum()), *means))

    torch.cuda.synchronize()
    result = protocol | {
        "utc": datetime.now(timezone.utc).isoformat(),
        "mAP": float(observed[:, 0].mean()),
        "precision": float(observed[:, 1].mean()),
        "verified_queries": len(observed),
        "verification_max_abs_AP_P": torch.stack(errors).amax(dim=0).tolist(),
        "gpu": torch.cuda.get_device_name(),
        "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
        "per_query_sha256": checksum(per_query),
        "per_class_sha256": checksum(per_class),
    }
    atomic_json(folder / "quickdraw.json", result)
    with zipfile.ZipFile(folder / "quickdraw-evidence.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (folder / "quickdraw.json", per_class, per_query):
            archive.write(path, path.name)
    print(f"{family} QuickDraw mAP@all {result['mAP']:.6f} P@200 {result['precision']:.6f}", flush=True)
    return result
