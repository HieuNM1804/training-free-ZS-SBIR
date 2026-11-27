"""Do two other drawings repeat a semantic error that the actual RGB source can resolve?"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .concept_benchmark import atomic_json, checksum
from .data import SKETCHY_SPLITS
from .encoding import save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE, load_split
from .native_block_probe import DST, PAIRED, SEED, deterministic
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_disputed_word_risk import query_posteriors, word_bank


@torch.inference_mode()
def measure(args):
    name = args.split
    spec = torch.load(PAIRED / f"real-errors-{name}.pt", weights_only=True)
    split, text, _ = load_split(DEFAULT_CACHE, name)
    if not torch.equal(split.query_ids, spec["query_ids"]) or not torch.equal(split.gallery_ids, spec["gallery_ids"]):
        raise ValueError("Archived dataset identifiers changed")
    named = spec["measurements"]["semantic"]
    all_source = spec["source"].cuda()
    labels = spec["labels"].cuda()
    q = torch.cat([query_posteriors(a, b, text[DEFAULT_PROMPTS[0]], split.scale)
                   for a, b in zip(split.sketch.split(128), split.mirror.split(128), strict=True)])
    photo = word_bank(split.photo, text[DEFAULT_PROMPTS[1]], split.scale).T.contiguous()
    winners = named["endpoints"].cuda()
    archived = named["own_selected_margin"].cuda()
    own = torch.einsum("ij,ij->i", q, photo[winners[:, 0]] - photo[winners[:, 1]])
    error = (own-archived).abs().max().item()
    if error > 2e-5:
        raise ValueError(f"Semantic margin replay error {error}")
    if not torch.equal(labels, split.photo_labels.cuda()[all_source]):
        raise ValueError("Source photo labels do not match drawings")

    collected = []
    for cls in labels.unique().tolist()[args.part::2]:
        class_rows = (labels == cls).nonzero().flatten()
        subjects, by_source = all_source[class_rows].unique(return_inverse=True)
        source_indicator = torch.nn.functional.one_hot(by_source, len(subjects)).double().T.contiguous()
        source_count = source_indicator.sum(1)
        for ids in torch.arange(len(class_rows), device="cuda").split(96):
            rows = class_rows[ids]
            contrast = photo[winners[rows, 0]] - photo[winners[rows, 1]]
            margins = q[class_rows] @ contrast.T
            source_correct = (photo[all_source[rows]] * contrast).sum(1) > 0
            active = (archived[rows] < 0) & source_correct & (source_count[by_source[ids]] >= 3)
            if not active.any():
                continue
            own_subject = by_source[ids[active]]
            cols = torch.arange(len(own_subject), device="cuda")
            errors = margins[:, active] < 0
            wrong_count = source_indicator @ errors.double()
            # Exclude the selected drawing itself when sampling actual siblings.
            e = wrong_count[own_subject, cols] - errors[ids[active], cols].double()
            n = source_count[own_subject] - 1
            real_two = e * (e - 1) / (n * (n - 1))
            real_one = e / n
            candidate = (source_count[:, None] >= 2) & (torch.arange(len(subjects), device="cuda")[:, None] != own_subject)
            ctrl_two = wrong_count * (wrong_count - 1) / (source_count * (source_count - 1)).clamp_min(1)[:, None]
            ctrl_one = wrong_count / source_count[:, None]
            ctrl_two = (ctrl_two * candidate).sum(0) / candidate.sum(0)
            ctrl_one = (ctrl_one * candidate).sum(0) / candidate.sum(0)
            record = torch.stack((rows[active], labels[rows[active]], all_source[rows[active]], n,
                                  real_one, ctrl_one, real_two, ctrl_two), dim=1)
            collected.append(record.cpu())
    data = torch.cat(collected)
    path = args.output / f"{name}-{args.part}-redraw.pt"
    save_tensor(data, path)
    atomic_json(path.with_suffix(".json"), {"revision": args.revision, "seed": SEED, "split": name,
               "part": args.part, "eligible": len(data), "replay_max_abs": error, "sha256": checksum(path)})
    print(name, args.part, len(data), flush=True)


def report(args):
    result = {}
    for name in SKETCHY_SPLITS:
        groups = []
        for part in (0, 1):
            path = args.output / f"{name}-{part}-redraw.pt"
            if checksum(path) != json.loads(path.with_suffix(".json").read_bytes())["sha256"]:
                raise ValueError("Invalid audit input")
            groups.append(torch.load(path, weights_only=True))
        data = torch.cat(groups).numpy()
        if len(np.unique(data[:, 0])) != len(data) or not np.isfinite(data).all():
            raise ValueError("Repeated rows or nonfinite result")
        values = []
        for c in np.unique(data[:, 1]):
            chunk = data[data[:, 1] == c]
            # Equal weight for source photographs within class, then equal weight for classes.
            summaries = [chunk[chunk[:, 2] == u, 4:].mean(0) for u in np.unique(chunk[:, 2])]
            values.append(np.mean(summaries, axis=0))
        values = np.array(values)
        positions = np.random.default_rng(SEED).integers(0, len(values), size=(10000, len(values)))
        diff = values[:, [0, 2]] - values[:, [1, 3]]
        result[name] = {"queries":len(data), "subjects":len(np.unique(data[:, 2])), "classes":len(values),
                        "mean_other_drawings":float(data[:, 3].mean()),
                        "variables":["one_same_source", "one_other_source", "two_same_source", "two_other_source"],
                        "mean_percent":(100*values.mean(0)).tolist(), "differences_points":(100*diff.mean(0)).tolist(),
                        "class_ci95_points":(100*np.quantile(diff[positions].mean(1), [.025, .975],axis=0).T).tolist()}
    atomic_json(args.output / "redraw-error-specificity.json", {"revision":args.revision,"seed":SEED,"results":result})
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SKETCHY_SPLITS)
    parser.add_argument("--part", type=int, choices=(0, 1))
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, default=DST)
    args = parser.parse_args()
    deterministic()
    if args.report:
        report(args)
    elif args.split is not None and args.part is not None:
        measure(args)
    else:
        parser.error("Specify a split and part, or --report.")
