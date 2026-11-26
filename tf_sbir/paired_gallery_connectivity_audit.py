"""Measure the ranking errors that cannot change inside the two-view score intervals."""

import json
from pathlib import Path

import open_clip
import torch

from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import revision
from .crossdomain_inputs import runtime
from .encoding import load_frozen_clip
from .gallery_convex_support_audit import prefix
from .paired_gallery_connectivity import FAMILY, SEED
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_boxable_prompt_search import BATCH, SKETCHY, Split
from .semantic_boxable_single_prompt import DOCS, VOCABULARY_STUDY, encode_prompt
from .semantic_boxable_sparse import image_inputs
from .semantic_disputed_word_risk import word_bank


def inspect_intervals(root, split, text, expected_protocol):
    result_path = root / f"{split.name}.pt"
    report = json.loads(result_path.with_suffix(".json").read_bytes())
    if report["protocol"] != expected_protocol or report["per_query_sha256"] != checksum(result_path):
        raise ValueError("Missing or invalid full paired-connectivity results")
    evidence = torch.load(result_path, weights_only=True)
    expected_order = (("qi", split.query_ids), ("pi", split.gallery_ids), ("labels", split.sketch_labels))
    if any(not torch.equal(evidence[name], value) for name, value in expected_order):
        raise ValueError("Audit used a different ordering")
    map_k, precision_k = report["cutoffs"]
    truth, labels = split.sketch_labels.cuda(), split.photo_labels.cuda()
    bank = word_bank(split.photo, text[DEFAULT_PROMPTS[1]], split.scale)
    blocks = []
    for start in range(0, len(truth), BATCH):
        sl = slice(start, min(start + BATCH, len(truth)))
        views = torch.stack((split.sketch[sl], split.mirror[sl].float()), dim=1)
        posterior = (split.scale * (views @ text[DEFAULT_PROMPTS[0]].T)).softmax(2)
        base = posterior.mean(1) @ bank
        a, b = (view.contiguous() @ bank for view in posterior.unbind(1))
        upper, lower = a.maximum(b), a.minimum(b)
        if torch.any((base > upper + 2e-6) | (base < lower - 2e-6)):
            raise ValueError("Baseline is outside its certified score interval")

        relevant = labels[None, :] == truth[sl, None]
        base_hit = relevant.gather(1, prefix(base, map_k, precision_k))
        observed = evidence["rows"]["dot"][sl]
        if not torch.equal(base_hit[:, 0].cpu(), observed[:, 3].bool()) or not torch.allclose(
            base_hit.float().mean(1).cpu(), observed[:, 1].float(), atol=2e-6, rtol=0,
        ):
            raise ValueError("Oracle audit does not replay the baseline")
        maximum_positive = upper.masked_fill(~relevant, -torch.inf).amax(1)
        minimum_maximum_negative = lower.masked_fill(relevant, -torch.inf).amax(1)
        feasible = maximum_positive >= minimum_maximum_negative
        best_case = torch.where(relevant, upper, lower)
        # A tie is broken optimistically for this mathematical upper bound.
        oracle_rank = (best_case.view(torch.int32).long() << 1) | relevant.long()
        oracle_hits = relevant.gather(1, oracle_rank.topk(precision_k).indices).sum(1)
        cap_hits = relevant.gather(1, prefix(upper, map_k, precision_k)).sum(1)
        blocks.append(torch.stack((base_hit[:, 0], feasible, base_hit.sum(1), oracle_hits, cap_hits), 1).cpu())
    stats = torch.cat(blocks)
    correct = stats[:, 0].bool()
    feasible = stats[:, 1].bool()
    if torch.any(correct & ~feasible):
        raise ValueError("Optimistic oracle cannot lose a correct first result")
    method = evidence["rows"]["connected_views"][:, 3].bool()
    return {
        "baseline_wrong_top1": int((~correct).sum()),
        "wrong_top1_impossible_to_repair": int((~correct & ~feasible).sum()),
        "wrong_top1_repairable_in_oracle": int((~correct & feasible).sum()),
        "wrong_top1_repaired_by_connected_views": int((~correct & method).sum()),
        "query_count": len(truth), "precision_cutoff": precision_k,
        "precision": dict(zip(("dot", "oracle_upper_bound", "pointwise_maximum"),
                              (100 * stats[:, 2:].double().mean(0) / precision_k).tolist(), strict=True)),
    }


@torch.inference_mode()
def run():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    workspace = Path("/content/sbir")
    root = workspace / "results" / FAMILY
    freeze = json.loads((root / "freeze.json").read_bytes())
    vocabulary = json.loads((DOCS / VOCABULARY_STUDY / "freeze.json").read_bytes())
    if vocabulary["sha256"] != freeze["config"]["vocabulary_sha256"]:
        raise ValueError("Different vocabulary from the official run")
    words = vocabulary["subsets"]["boxable"]["words"]
    cfg = runtime(workspace, "sketchy")
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = {template: encode_prompt(model, tokenizer, words, template) for template in DEFAULT_PROMPTS}
    data = image_inputs(cfg, SKETCHY, model)
    output = {name: inspect_intervals(root, Split.from_inputs(data, name), text, freeze["sha256"]) for name in SKETCHY}
    del data
    cfg = runtime(workspace, "tuberlin")
    split = Split.from_inputs(image_inputs(cfg, ("tuberlin",), model), "tuberlin")
    output[split.name] = inspect_intervals(root, split, text, freeze["sha256"])
    atomic_json(root / "interval-audit.json", {
        "source_run": freeze["config"]["commit"], "audit_commit": revision(), "protocol": freeze["sha256"],
        "results": output,
    })


if __name__ == "__main__":
    run()
