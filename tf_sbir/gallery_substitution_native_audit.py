"""Ask whether the substitution operator recovers a photo from its own alternate text readout."""

import json
import os
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F

from .concept_benchmark import atomic_json
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT, runtime
from .encoding import load_frozen_clip
from .gallery_convex_support import SEED
from .gallery_substitution_alignment import GallerySubstitution
from .gallery_substitution_study import FAMILY
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import BATCH, SKETCHY, Split
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt
from .semantic_boxable_sparse import image_inputs
from .semantic_disputed_word_risk import word_bank


ARMS = ("substitution", "uncalibrated", "fixed_map", "fixed_substitution", "hellinger")


def exact_example(model, i, j):
    """Rebuild the leave-one-out matrix and explicitly maximize each constrained objective."""
    a, b = model.source.double(), model.target.double()
    matrix = a.T @ b / len(a)
    minus = matrix - a[j, :, None] * b[j, None, :] / len(a)
    target = F.normalize(b[j], dim=0)
    right = minus - (minus @ target)[:, None] * target

    def maximum(source):
        source = F.normalize(source, dim=0)
        compressed = right - source[:, None] * (source @ right)
        return source @ minus @ target + torch.linalg.svdvals(compressed, driver="gesvd").sum()

    return float((maximum(a[i]) - maximum(a[j])) / model.unit)


def audit(split, text):
    heads = [word_bank(split.photo, text[prompt], split.scale).T.contiguous() for prompt in DEFAULT_PROMPTS]
    model = GallerySubstitution(*heads, seed=SEED)
    n = model.n
    pieces = {arm: [] for arm in ARMS}
    largest = (-float("inf"), -1, -1)
    ids = torch.arange(n, device=model.a.device)
    for i in range(0, n, BATCH):
        rows = ids[i:i + BATCH]
        local = torch.arange(len(rows), device=ids.device)
        scores = model.scores(model.a[i:i + BATCH])
        for name in ARMS:
            values = scores[name]
            if not torch.isfinite(values).all():
                raise ValueError(f"Nonfinite native score in {name}")
            self_score = values[local, rows].clone()
            # Equal scores break by gallery position, including for top1 and reciprocal rank.
            equal = values == self_score[:, None]
            rank = 1 + (values > self_score[:, None]).sum(1) + (equal & (ids < rows[:, None])).sum(1)
            best = values.argmax(1)
            values[local, rows] = -torch.inf
            alternative, idx = values.max(1)
            if name == "substitution":
                strongest = int(alternative.argmax())
                candidate = (float(alternative[strongest]), int(rows[strongest]), int(idx[strongest]))
                if candidate[0] > largest[0]:
                    largest = candidate
            # Labels are read only after all rank and own-match decisions.
            pieces[name].append(torch.stack((rank, best, self_score, alternative), 1).cpu())
        if i == 0 or i + BATCH >= n or (i // BATCH + 1) % 20 == 0:
            print(split.name, min(i + BATCH, n), "/", n, flush=True)
    summary = {}
    classes = split.photo_labels.cpu()
    records = {}
    for name, parts in pieces.items():
        rank, best, self_score, alternative = torch.cat(parts).T
        best = best.long()
        own = rank == 1
        same = classes[best] == classes
        summary[name] = {
            "own_top1": int(own.sum()), "own_mrr": float(rank.double().reciprocal().mean()),
            "own_median_rank": float(rank.median()), "other_top1_same_class": int((~own & same).sum()),
            "other_top1_different_class": int((~own & ~same).sum()),
            "strictly_better_alternative": int((alternative > self_score).sum()),
            "tied_best_alternative": int((alternative == self_score).sum()),
        }
        if name == "substitution":
            records = {
                "computed_own_max_abs": float(self_score.abs().max()),
                "strict_positive_rival_above_own": int(((alternative > 0) & (alternative > self_score)).sum()),
                "best_rival_above_threshold": {str(t): int((alternative > t).sum()) for t in (0., 2e-4, 1e-3)},
                "margin_above_computed_self_and_zero": {
                    str(t): int(((alternative > t) & (alternative - self_score > t)).sum())
                    for t in (2e-4, 1e-3)
                },
            }
    value, query, other = largest
    svd = exact_example(model, query, other)
    records["strongest_example"] = {
        "query_gallery_id": int(split.gallery_ids[query]), "rival_gallery_id": int(split.gallery_ids[other]),
        "query_class": int(classes[query]), "rival_class": int(classes[other]),
        "same_class": bool(classes[query] == classes[other]), "approximate_substitution": value,
        "direct_double_deleted_matrix_svd": svd, "absolute_discrepancy": abs(value - svd),
    }
    if abs(svd - value) > 2e-4:
        raise ValueError(f"Explicit SVD disagrees with strongest reversal in {split.name}")
    return {"gallery_as_queries": n, "arms": summary, "native_substitution": records}


@torch.inference_mode()
def run():
    workspace = Path("/content/sbir")
    root = workspace / "results" / FAMILY
    lock = json.loads((root / "freeze.json").read_bytes())
    vocab = json.loads((DOCS / VOCABULARY_STUDY / "freeze.json").read_bytes())
    vocab_sha = vocab.pop("sha256")
    words = vocab["subsets"]["boxable"]["words"]
    cfg = lock["config"]
    if (digest(cfg) != lock["sha256"] or digest(vocab) != vocab_sha or len(words) != 601
            or cfg["vocabulary_sha256"] != vocab_sha or cfg["clip_sha256"] != CHECKPOINT
            or tuple(cfg["prompts"]) != DEFAULT_PROMPTS or cfg["seed"] != SEED):
        raise ValueError("Archive does not match the primary protocol")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    workers = min(4, len(os.sched_getaffinity(0)))
    config = runtime(workspace, "sketchy", batch=BATCH, workers=workers)
    clip = load_frozen_clip(config.clip_checkpoint, torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = {p: encode_prompt(clip, tokenizer, words, p) for p in DEFAULT_PROMPTS}
    results = {}
    for dataset, names in (("sketchy", SKETCHY), ("tuberlin", ("tuberlin",))):
        config = runtime(workspace, dataset, batch=BATCH, workers=workers)
        data = image_inputs(config, names, clip)
        for name in names:
            results[name] = audit(Split.from_inputs(data, name), text)
        del data
    atomic_json(root / "native-match-audit.json", {
        "primary_commit": cfg["commit"], "protocol": lock["sha256"], "audit_commit": revision(),
        "gpu": torch.cuda.get_device_name(), "seed": SEED,
        "task": "RGB photo via drawing text head to same RGB gallery via photo text head",
        "tie_break": "stable gallery row ascending", "own_theoretical_substitution_score": 0,
        "results": results,
    })


if __name__ == "__main__":
    run()
