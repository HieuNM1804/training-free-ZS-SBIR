"""Test predeclared linguistic predictions of two observed visual domain changes."""

import argparse
import json

import numpy as np
import torch
from torch.nn.functional import normalize

from .concept_benchmark import atomic_json, checksum
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .kaggle_unseen_features import DEFAULT_CACHE, load_visual_split
from .native_block_probe import SEED, deterministic
from .native_choice_observations import DOMAINS, OUT
from .paired_patch_interaction_probe import DST as ENDPOINTS


def load_observations(name):
    parts = []
    first = OUT / f"observations-{name}-0.json"
    for i in range(json.loads(first.read_bytes())["parts"]):
        path = OUT / f"observations-{name}-{i}.pt"
        meta = json.loads(path.with_suffix(".json").read_bytes())
        if checksum(path) != meta["sha256"]:
            raise ValueError("Observation shard checksum mismatch")
        parts.append(torch.load(path, weights_only=True))
    a = parts[0]
    if any(p["revision"] != a["revision"] or p["classes"] != a["classes"] for p in parts):
        raise ValueError("Input shards use different configurations")
    order = torch.cat([p["positions"] for p in parts]).sort()
    if not torch.equal(order.values, torch.arange(len(order.values))):
        raise ValueError("Incomplete photo gallery")
    ids = torch.cat([p["gallery_ids"] for p in parts])[order.indices]
    if ids.unique().numel() != len(ids):
        raise ValueError("Repeated photo ID")
    counts = sum(p["class_counts"] for p in parts)
    if counts.min() < 1 or counts.sum() != len(ids):
        raise ValueError("Invalid class photo counts")
    y = {"drawing_to_photo": a["human_change"].numpy(),
         "grayscale_to_rgb": (sum(p["sum_color_changes"] for p in parts) / counts[:, None]).numpy()}
    x = {op: {k: v.numpy() for k, v in sensors.items()} for op, sensors in a["text"].items()}
    info = {"classes": a["classes"], "photos": len(ids), "source": a["human_sampling"],
            "min_original_replay_cosine": min(p["min_baseline_replay_cosine"] for p in parts),
            "mean_RGB_gray_distance": sum(p["sum_movement"] for p in parts) / len(ids), "encoder_revision": a["revision"]}
    return y, x, info


def leave_one_out(y, *features):
    n = len(y)
    mean_other = (y.sum(0) - y) / (n - 1)
    if not features:
        return mean_other, np.empty((n, 0))
    x = np.stack(features, axis=1)
    xc = x - x.mean(0)
    yc = y - y.mean(0)
    gram = xc @ xc.transpose(0, 2, 1)
    cross = (xc * yc[:, None, :]).sum(2)
    ratio = n / (n - 1)
    # Delete a complete class and re-center ALL fitting rows at their remaining mean.
    coefs = np.linalg.solve(gram.sum(0) - ratio * gram, (cross.sum(0) - ratio * cross)[:, :, None])[:, :, 0]
    return mean_other + ratio * (coefs[:, :, None] * xc).sum(1), coefs


def frozen_prediction(y, sources, targets):
    x = np.stack(sources, axis=1)
    x_bar = x.mean(0)
    design = (x - x_bar).transpose(0, 2, 1).reshape(-1, len(sources))
    beta, _, rank, _ = np.linalg.lstsq(design, (y - y.mean(0)).ravel(), rcond=None)
    if rank != len(sources):
        raise ValueError("Source controls not identifiable")
    test = np.stack(targets, axis=1)
    return y.mean(0) + ((test - x_bar) * beta[None, :, None]).sum(1), beta


def loss(y, estimate):
    return ((y - estimate) ** 2).sum(1)


def relative_loss(comparator, proposed):
    return float(100 * (1 - proposed.sum() / comparator.sum()))


def gain(comparator, proposed):
    n = len(comparator)
    rng = np.random.default_rng(SEED)
    resample = rng.integers(n, size=(10000, n))
    change = 100 * (1 - proposed[resample].sum(1) / comparator[resample].sum(1))
    return {"percent_squared_error_reduction": relative_loss(comparator, proposed),
            "class_bootstrap95": np.quantile(change, [.025, .975]).tolist()}


def compare(y, x):
    mu, _ = leave_one_out(y)
    meaning, _ = leave_one_out(y, x["meaning"])
    direct, beta = leave_one_out(y, x["meaning"], x["depiction"])
    article, _ = leave_one_out(y, x["meaning"], x["article"])
    literal, _ = frozen_prediction(y[1:], [x["meaning"][1:], x["depiction"][1:]],
                                   [x["meaning"][:1], x["depiction"][:1]])
    check = float(np.abs(literal[0] - direct[0]).max())
    if check > 1e-9:
        raise ValueError("Deleted-class closed form differs from an explicit refit")
    base, model = loss(y, meaning), loss(y, direct)
    observed = relative_loss(base, model)
    permutations = []
    rng = np.random.default_rng(SEED)
    for _ in range(999):
        shuffled, _ = leave_one_out(y, x["meaning"], x["depiction"][rng.permutation(len(y))])
        permutations.append(relative_loss(base, loss(y, shuffled)))
    null = np.asarray(permutations)
    out = {"classes": len(y), "meaning_over_class_mean": gain(loss(y, mu), base),
           "depiction_over_meaning": gain(base, model), "article_over_meaning": gain(base, loss(y, article)),
           "name_permutation_gain95": float(np.quantile(null, .95)),
           "name_permutation_one_sided_p": float((1 + (null >= observed).sum()) / (1 + len(null))),
           "median_depiction_coefficient": float(np.median(beta[:, 1])),
           "positive_coefficient_folds": int((beta[:, 1] > 0).sum()),
           "direct_refit_max_error": check, "class_squared_errors_meaning": base.tolist(),
           "class_squared_errors_depiction": model.tolist()}
    return out, meaning, direct


def transfer(left, right, classes):
    y_src, x_src, _ = left
    y_target, x_target, _ = right
    result = {}
    for op in DOMAINS:
        sm, tm = x_src[op], x_target[op]
        ref, _ = frozen_prediction(y_src[op], [sm["meaning"]], [tm["meaning"][classes]])
        full, coefficient = frozen_prediction(y_src[op], [sm["meaning"], sm["depiction"]],
                                             [tm["meaning"][classes], tm["depiction"][classes]])
        y = y_target[op][classes]
        result[op] = {**gain(loss(y, ref), loss(y, full)), "source_coefficients": coefficient.tolist()}
    return result


@torch.inference_mode()
def actual_margins(name, labels_mask, meaning, depiction):
    split = load_visual_split(DEFAULT_CACHE, name)
    archive = torch.load(ENDPOINTS / f"real-errors-{name}.pt", weights_only=True, mmap=True)
    if not torch.equal(archive["query_ids"], split.query_ids) or not torch.equal(archive["gallery_ids"], split.gallery_ids):
        raise ValueError("Native endpoints shifted")
    pair = archive["measurements"]["native"]
    query = normalize(normalize(split.sketch.float(), dim=1) + normalize(split.mirror.float(), dim=1), dim=1).double()
    photo = normalize(split.photo.float(), dim=1).double()
    correct, rival = pair["endpoints"].cuda().T
    contrast = photo[correct] - photo[rival]
    ids = archive["source"].cuda()
    labels = split.sketch_labels.cuda()
    dx = ((photo[ids] - query) * contrast).sum(1)
    proposed = []
    label_ids = np.flatnonzero(labels_mask)
    for pred in (meaning, depiction):
        bank = torch.zeros(len(UNSEEN_CLASSES[name]), 512, device="cuda", dtype=torch.float64)
        bank[label_ids] = torch.from_numpy(pred).cuda()
        proposed.append((bank[labels] * contrast).sum(1))
    bad = pair["own_selected_margin"].cuda() < 0
    reports = {}
    for key, event in (("all", torch.ones_like(bad)), ("native_wrong", bad)):
        squared = []
        allowed = torch.tensor(labels_mask, device=labels.device)
        good = event & allowed[labels] & allowed[split.photo_labels.cuda()[rival]]
        for c in label_ids:
            keep = good & (labels == int(c))
            if not keep.any():
                continue
            _, group = ids[keep].unique(return_inverse=True)
            weight = group.bincount().double()[group].reciprocal()
            squared.append([(((dx[keep]-pred[keep]).square()*weight).sum()/weight.sum()).item() for pred in proposed])
        left, right = np.array(squared).T
        reports[key] = {"queries": int(good.sum()), "classes": len(left), **gain(left, right)}
    return reports


def report(names, revision):
    data = {s: load_observations(s) for s in names}
    results = {"revision": revision, "seed": SEED, "datasets": {}, "transfer_from_ext1": {}}
    for name in names:
        y, x, info = data[name]
        labels = info["classes"]
        out = {"info": info, "class_sets": {}}
        other = SKETCHY_SPLITS[1 - SKETCHY_SPLITS.index(name)] if name in SKETCHY_SPLITS else None
        masks = {"all": np.ones(len(labels), dtype=bool)}
        if other is not None:
            masks["disjoint"] = np.array([label not in UNSEEN_CLASSES[other] for label in labels])
        for selection, mask in masks.items():
            rows = {}
            for op in DOMAINS:
                xx = {key: v[mask] for key, v in x[op].items()}
                row, mean, full = compare(y[op][mask], xx)
                if name in SKETCHY_SPLITS and op == "drawing_to_photo":
                    row["actual_native_pair_margin_change"] = actual_margins(name, mask, mean, full)
                rows[op] = row
            out["class_sets"][selection] = rows
        results["datasets"][name] = out
    if len(names) > 1:
        left = data["sketchy_1"]
        for name in names[1:]:
            allow = np.ones(len(data[name][2]["classes"]), dtype=bool)
            if name == "sketchy_2":
                allow = np.array([s not in left[2]["classes"] for s in data[name][2]["classes"]])
            results["transfer_from_ext1"][name] = {"classes": int(allow.sum()),
                                                  "results": transfer(left, data[name], allow)}
    dest = OUT / ("language-locked-transfer.json" if "tuberlin" in names else "language-sketchy.json")
    atomic_json(dest, results)
    print(dest.name, "complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--heldout", action="store_true")
    args = parser.parse_args()
    deterministic()
    report(["sketchy_1", "tuberlin"] if args.heldout else list(SKETCHY_SPLITS), args.revision)
