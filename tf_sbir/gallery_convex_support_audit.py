"""Compare the relevance of equal numbers of newly retrieved photos."""

import json
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F

from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import revision
from .crossdomain_inputs import runtime
from .depiction_path_diagnostic import class_interval
from .encoding import load_frozen_clip
from .gallery_convex_support import (
    FAMILY, NEIGHBORS, SEED, direction_masks, nearest_gallery, pooled_order,
    shuffle_directions, support_floor, tangent_coordinates,
)
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import BATCH, SKETCHY, Split, posteriors
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt
from .semantic_boxable_sparse import image_inputs


def prefix(scores, map_k, precision_k):
    return scores.topk(map_k or scores.size(1)).indices[:, :precision_k]


def newcomer_hits(ordered, in_baseline, gallery_labels, truth, official_precision):
    hits = gallery_labels[ordered] == truth[:, None]
    if not torch.allclose(hits.float().mean(1).cpu(), official_precision.float(), atol=2e-6, rtol=0):
        raise ValueError("Diagnostic did not reproduce the original evaluator's prefix")
    return ~in_baseline.gather(1, ordered), hits


def matched_counts(first, control):
    new, truth = first
    old, control_truth = control
    n = new.sum(dim=1).minimum(old.sum(dim=1))
    first_hits = (truth & new & (new.cumsum(dim=1) <= n[:, None])).sum(dim=1)
    control_hits = (control_truth & old & (old.cumsum(dim=1) <= n[:, None])).sum(dim=1)
    return torch.stack((n, first_hits, control_hits, new.sum(1), old.sum(1)), dim=1).cpu()


def summarize(rows, truth):
    n, first, control, first_all, control_all = rows.T
    valid = n > 0
    pairs = int(n.sum())
    return {
        "queries_with_two_nonempty_newcomer_sets": int(valid.sum()),
        "pairs": pairs, "concept_hits": int(first.sum()), "control_hits": int(control.sum()),
        "all_newcomers_concept": int(first_all.sum()), "all_newcomers_control": int(control_all.sum()),
        "purity_points": [100 * int(hits.sum()) / pairs for hits in (first, control)] if pairs else None,
        "paired_macro_purity_change": class_interval(
            (100 * (first[valid] - control[valid]) / n[valid]).double(), truth[valid], seed=SEED,
        ),
    }


def lexical_quartiles(uncertainty, rows, truth):
    groups = [[] for _ in range(4)]
    for cls in truth.unique():
        members = torch.where(truth == cls)[0]
        order = uncertainty[members].argsort(stable=True)
        for group, ids in zip(groups, members[order].tensor_split(4), strict=True):
            group.append(ids)
    contrasts = ("pooled_hull", "shuffled_directions", "dot")
    return [{
        "queries": len(ids),
        "median_one_minus_q_max": uncertainty[ids].median().item(),
        "macro_ap_changes": {
            control: class_interval(
                100 * (rows["concept_hull"][ids, 0] - rows[control][ids, 0]), truth[ids], seed=SEED,
            )
            for control in contrasts
        },
    } for ids in map(torch.cat, groups)]


def audit(root, split, text, signature):
    row_file = root / f"{split.name}.pt"
    report = json.loads(row_file.with_suffix(".json").read_bytes())
    if report["protocol"] != signature or report["per_query_sha256"] != checksum(row_file):
        raise ValueError("Audit requires the verified full runs")
    saved = torch.load(row_file, weights_only=True)
    if any(not torch.equal(saved[k], v) for k, v in (
        ("qi", split.query_ids), ("pi", split.gallery_ids), ("labels", split.sketch_labels),
    )):
        raise ValueError("Gallery or query order changed")

    q, p = posteriors(split, text, [DEFAULT_PROMPTS])
    q, p = q[DEFAULT_PROMPTS[0]], p[DEFAULT_PROMPTS[1]]
    photo = F.normalize(split.photo.float(), dim=-1)
    index, _ = nearest_gallery(photo, NEIGHBORS)
    index = index.int().contiguous()
    coordinates, _, _ = tangent_coordinates(photo, index)
    masks = direction_masks(coordinates)
    real = p.maximum(support_floor(p, index, masks))
    shuffled = p.maximum(support_floor(p, index, shuffle_directions(masks)))
    labels, truth = split.photo_labels.cuda(), split.sketch_labels.cuda()
    map_k, precision_k = report["cutoffs"]
    contrasts = {name: [] for name in ("shuffled_directions", "pooled_hull")}
    for start in range(0, len(q), BATCH):
        qs = q[start:start + BATCH]
        base = qs @ p
        sl = slice(start, start + len(qs))
        ordered = prefix(base, map_k, precision_k)
        in_baseline = torch.zeros_like(base, dtype=torch.bool).scatter_(1, ordered, True)
        newcomer_hits(ordered, in_baseline, labels, truth[sl], saved["rows"]["dot"][sl, 1])
        first = newcomer_hits(
            prefix(qs @ real, map_k, precision_k), in_baseline, labels, truth[sl],
            saved["rows"]["concept_hull"][sl, 1],
        )
        controls = (
            qs @ shuffled,
            pooled_order(base, support_floor(base, index, masks)),
        )
        for name, scores in zip(contrasts, controls, strict=True):
            second = newcomer_hits(
                prefix(scores, map_k, precision_k), in_baseline, labels, truth[sl], saved["rows"][name][sl, 1],
            )
            contrasts[name].append(matched_counts(first, second))
    return {
        "matched_promotions": {name: summarize(torch.cat(parts), split.sketch_labels)
                                for name, parts in contrasts.items()},
        "lexical_quartiles": lexical_quartiles(1 - q.amax(1).cpu(), saved["rows"], split.sketch_labels),
    }


@torch.inference_mode()
def run():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    workspace = Path("/content/sbir")
    root = workspace / "results" / FAMILY
    freeze = json.loads((root / "freeze.json").read_bytes())
    cfg = runtime(workspace, "sketchy")
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    config = json.loads((DOCS / VOCABULARY_STUDY / "freeze.json").read_bytes())
    if config["sha256"] != freeze["config"]["vocabulary_sha256"]:
        raise ValueError("Different vocabulary")
    words = config["subsets"]["boxable"]["words"]
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = {prompt: encode_prompt(model, tokenizer, words, prompt) for prompt in DEFAULT_PROMPTS}
    data = image_inputs(cfg, SKETCHY, model)
    output = {name: audit(root, Split.from_inputs(data, name), text, freeze["sha256"]) for name in SKETCHY}
    del data
    cfg = runtime(workspace, "tuberlin")
    split = Split.from_inputs(image_inputs(cfg, ("tuberlin",), model), "tuberlin")
    output[split.name] = audit(root, split, text, freeze["sha256"])
    atomic_json(root / "mechanism-audit.json", {
        "source_run": freeze["config"]["commit"], "audit_commit": revision(), "protocol": freeze["sha256"],
        "seed": SEED, "results": output,
    })


if __name__ == "__main__":
    run()
