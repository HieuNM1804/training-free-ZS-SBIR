"""Choose saliency-coordinate transfer settings on Sketchy and score the frozen choices on all TU classes."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from .benchmark import merge_inputs
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT, runtime
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .encoding import load_global_inputs
from .foundation import load_words
from .saliency_transfer import FAMILIES, Setting, evaluate_split, grid, refined, report_parameters, setting_id
from .saliency_views import bank
from .semantic import read_words
from dataclasses import asdict, replace


EXPECTED = {
    "sketchy_1": (14179, 17100, 100, 0),
    "sketchy_2": (12694, 12552, 200, 200),
    "tuberlin": (2130, 27262, 100, 0),
}
LOCK = "saliency-sketchy-to-tuberlin-lock.json"
COARSE = "saliency-sketchy-seed.json"


def folder(workspace, held):
    return workspace / "results" / ("saliency-tuberlin-frozen" if held else "saliency-sketchy")


def checked_row(directory, split, name, config, mode, signature=None):
    row = json.loads((directory / (split + "-" + name + ".json")).read_text())
    tensor = torch.load(
        directory / "per-query" / (split + "-" + name + ".pt"), map_location="cpu", weights_only=True
    )
    count = {"full": 1, "core": 2, "all": 7}[config.views]
    if (
        row["method"] != name
        or row["split"] != split
        or row["parameters"] != report_parameters(config, mode, count)
        or row["revision"] != revision()
        or row["checkpoint_sha256"] != CHECKPOINT
        or row["freeze_sha256"] != signature
        or row["exploratory_unseen"] != (split != "tuberlin")
        or row["external_vocabulary"] != (20931 if config.family == "public" else 0)
        or tuple(row[k] for k in ("queries", "gallery", "p_k", "map_k")) != EXPECTED[split]
        or tensor.shape != (EXPECTED[split][0], 2)
        or not torch.allclose(tensor.mean(0), torch.tensor([row["mAP"], row["precision"]]), atol=5e-7, rtol=0)
    ):
        raise ValueError(f"Invalid result for {split} {name}")
    return row


def verify_and_choose(workspace, catalog):
    directory = folder(workspace, False)
    ledger = {
        n: {
            split: {k: r[k] for k in ("precision", "mAP", "parameters")}
            for split in SKETCHY_SPLITS
            for r in [checked_row(directory, split, n, *value)]
        }
        for n, value in catalog.items()
    }

    def score(n):
        return tuple(sum(ledger[n][s][k] for s in SKETCHY_SPLITS) / 2 for k in ("precision", "mAP"))

    winners = {
        family: max((n for n, (c, _) in catalog.items() if c.paired and c.family == family), key=score)
        for family in FAMILIES
    }
    return winners, ledger


def locked_file(path, payload):
    if path.is_file():
        data = json.loads(path.read_text())
        unsigned = data.copy()
        key = unsigned.pop("freeze_sha256")
        if digest(unsigned) != key or {k: v for k, v in unsigned.items() if k != "utc"} != payload:
            raise ValueError("An incompatible Sketchy selection already exists")
        return data
    saved = payload | {"utc": datetime.now(timezone.utc).isoformat()}
    saved["freeze_sha256"] = digest(saved)
    tmp = path.with_suffix(".pending")
    tmp.write_text(json.dumps(saved, indent=2) + "\n")
    tmp.replace(path)
    return saved


def select_seed(workspace):
    catalog = grid()
    winners, ledger = verify_and_choose(workspace, catalog)
    data = {
        "revision": revision(),
        "checkpoint_sha256": CHECKPOINT,
        "selection_splits": list(SKETCHY_SPLITS),
        "ledger_sha256": digest(ledger),
        "winners": winners,
        "count": len(catalog),
    }
    return locked_file(workspace / "results" / COARSE, data)


def choices(workspace):
    seed = json.loads((workspace / "results" / COARSE).read_text())
    tmp = seed.copy()
    sig = tmp.pop("freeze_sha256")
    if seed["revision"] != revision() or digest(tmp) != sig:
        raise ValueError("Invalid coarse Sketchy selection")
    catalog = grid()
    return refined({f: catalog[n] for f, n in seed["winners"].items()})


def freeze(workspace):
    catalog = grid() | choices(workspace)
    winners, ledger = verify_and_choose(workspace, catalog)
    held = {n: catalog[n] for n in winners.values()}
    controls = {}
    for family, name in winners.items():
        c, mode = catalog[name]
        frame = replace(c, views="full", layout=0.0, region=0.0)
        key = setting_id(frame, mode)
        held[key] = (frame, mode)
        controls[family + "_frame"] = key
        if family == "public":
            shuffled = replace(c, paired=False)
            key = setting_id(shuffled, mode)
            held[key] = (shuffled, mode)
            controls["public_shuffle"] = key
    path = workspace / "results" / LOCK
    if not path.is_file() and list(folder(workspace, True).glob("tuberlin-*.json")):
        raise ValueError("This study already scored TU without its Sketchy lock")
    p = {
        "revision": revision(),
        "checkpoint_sha256": CHECKPOINT,
        "selection_splits": list(SKETCHY_SPLITS),
        "rule": "equal mean Sketchy precision, followed by equal mean Sketchy mAP",
        "tu_split_sha256": SPLIT_HASHES["tuberlin"],
        "tu_classes": 30,
        "tu_rows_before_first_lock": 0,
        "ledger_sha256": digest(ledger),
        "coarse_count": len(grid()),
        "refine_count": len(catalog) - len(grid()),
        "winners": winners,
        "controls": controls,
        "selected_sketchy": {n: ledger[n] for n in winners.values()},
        "heldout": {n: {"config": asdict(c), "mode": mode} for n, (c, mode) in held.items()},
    }
    return locked_file(path, p)


@torch.inference_mode()
def run(workspace, stage, split):
    if stage == "seed":
        print(json.dumps(select_seed(workspace), indent=2))
        return
    if stage == "freeze":
        print(json.dumps(freeze(workspace), indent=2))
        return
    held = stage == "heldout"
    if held:
        lock = json.loads((workspace / "results" / LOCK).read_text())
        copy = lock.copy()
        sig = copy.pop("freeze_sha256")
        if (
            sig != digest(copy)
            or lock["revision"] != revision()
            or lock["selection_splits"] != list(SKETCHY_SPLITS)
            or lock["tu_classes"] != 30
            or lock["tu_split_sha256"] != digest(UNSEEN_CLASSES["tuberlin"])
        ):
            raise ValueError("Unverified TU setting or classes")
        catalog = {n: (Setting(**v["config"]), v["mode"]) for n, v in lock["heldout"].items()}
    else:
        sig = None
        catalog = grid() if stage == "sketchy" else choices(workspace)
    splits = ("tuberlin",) if held else SKETCHY_SPLITS
    if split and split not in splits:
        raise ValueError("Unexpected split for stage")
    args = runtime(workspace, "tuberlin" if held else "sketchy", batch=64, workers=2, crop_batch=192)
    args.score_batch = 192
    args.crop_workers = 6
    args.output = folder(workspace, held)
    args.output.mkdir(exist_ok=True, parents=True)
    qp, qi = merge_inputs(args.root, splits, "sketch")
    pp, pi = merge_inputs(args.root, splits, "photo")
    raw, photo, _ = load_global_inputs(args, qp, pp, False)
    q_bank, p_bank = bank(args, qp, "sketch"), bank(args, pp, "photo")
    words = read_words(args.vocabulary)
    texts = tuple(x.cuda() for x in load_words(args, words))
    for s in splits:
        if split and s != split:
            continue
        if digest(UNSEEN_CLASSES[s]) != SPLIT_HASHES[s]:
            raise ValueError("Unexpected unseen split")
        qids, ql = qi[s]
        pids, pl = pi[s]
        if s == "tuberlin" and (
            len(UNSEEN_CLASSES[s]) != 30 or ql.unique().numel() != 30 or pl.unique().numel() != 30
        ):
            raise ValueError("All 30 TU classes are required in queries and gallery")
        todo = {}
        for n, v in catalog.items():
            if (args.output / (s + "-" + n + ".json")).is_file():
                checked_row(args.output, s, n, *v, sig)
            else:
                todo[n] = v
        if not todo:
            continue
        args.metric_workers = 2 if s == "sketchy_2" else 4
        metadata = {
            "backbone": "OpenAI CLIP ViT-B/32",
            "checkpoint_sha256": CHECKPOINT,
            "gpu": torch.cuda.get_device_name(),
            "torch": str(torch.__version__),
            "revision": revision(),
            "exploratory_unseen": not held,
            "public_terms": len(words),
            "study": "saliency-coordinate-transfer",
            "freeze_sha256": sig,
            "classes": len(UNSEEN_CLASSES[s]),
        }
        evaluate_split(
            args,
            s,
            raw[qids].cuda(),
            photo[pids].cuda(),
            q_bank[qids].cuda(),
            p_bank[pids].cuda(),
            texts,
            todo,
            (ql, pl),
            metadata,
        )
    if held:
        for n, v in catalog.items():
            checked_row(args.output, "tuberlin", n, *v, sig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--stage", required=True, choices=("sketchy", "seed", "refine", "freeze", "heldout"))
    parser.add_argument("--split", choices=(*SKETCHY_SPLITS, "tuberlin"))
    options = parser.parse_args()
    run(options.workspace, options.stage, options.split)
