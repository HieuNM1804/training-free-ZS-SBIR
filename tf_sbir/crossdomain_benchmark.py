"""Tune only on the two Sketchy splits, freeze the selections, and evaluate those choices on TU-Berlin."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from .contour_query_expansion import gallery_word_codes
from .crossdomain_inputs import CHECKPOINT, inputs, projection, runtime
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .domain_prompt_gallery import semantic_rows, settings, visual_rows
from .foundation import DEFAULT_DOMAIN_TEMPLATES, load_words
from .gallery_diffusion import nearest_gallery
from .gallery_routing import EXPERTS, calibrate, rank_banks, routed_tensor
from .global_methods import centered
from .metrics import cosine_rows
from .paired_neighborhood_scale import scale_index
from .reflection_transfer import cache_reflection_features
from .semantic import read_words
from .transfer_benchmark import write_result
from .transfer_reconstruction import anchor_errors, nearest_anchors
from .visual_paired_graph import image_index, score_setting


VISUAL = ("frame-images-one", "frame-images-mean", "frame-images-agreement")
PREVIOUS = ("visual-graph-k64-g1-r1.5", "route-neighbors100-k128-q75")
SPLIT_HASHES = {
    "sketchy_1": "8fafaf3aedf4751e28e2b33f073832e02fdf3ca1b36a9d7fda5ff328f26b0642",
    "sketchy_2": "30baecaac02eeff22cc54e533eb6bc1b757ba3a891a84f61a77ee837f073d8c6",
    "tuberlin": "64fa2e05b5ce359af89caf817c2a90d8d48b268280e2f4896283aaa2d58f8bcd",
}


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def revision():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()


def freeze(workspace, templates):
    study = workspace / "results/domain-sketchy"
    file = workspace / "results/sketchy-to-tuberlin-lock.json"
    groups = {"general_visual": VISUAL, "general_semantic": tuple(n for n in settings() if "unpaired" not in n)}
    catalog = ("native-clip", *VISUAL, *settings())
    ledger, report = {}, {}
    for name in catalog:
        ledger[name] = {}
        for split in SKETCHY_SPLITS:
            row = json.loads((study / f"{split}-{name}.json").read_text())
            details = torch.load(study / "per-query" / f"{split}-{name}.pt", map_location="cpu", weights_only=True)
            expected = (14179, 17100, 100, 0) if split == "sketchy_1" else (12694, 12552, 200, 200)
            if (
                row["revision"] != revision()
                or not row["exploratory_unseen"]
                or row["checkpoint_sha256"] != CHECKPOINT
                or (row["queries"], row["gallery"], row["p_k"], row["map_k"]) != expected
                or details.shape != (expected[0], 2)
                or not torch.allclose(
                    details.mean(0), torch.tensor([row["mAP"], row["precision"]]), atol=5e-7, rtol=0
                )
            ):
                raise ValueError(f"Sketchy evidence does not match {name} on {split}")
            ledger[name][split] = {k: row[k] for k in ("precision", "mAP", "parameters")}
    for kind, names in groups.items():

        def score(name):
            return tuple(sum(ledger[name][s][m] for s in SKETCHY_SPLITS) / 2 for m in ("precision", "mAP"))

        winner = max(names, key=score)
        report[kind] = {
            "name": winner,
            "parameters": ledger[winner]["sketchy_1"]["parameters"],
            "sketchy": ledger[winner],
        }
    lock = {
        "revision": revision(),
        "checkpoint_sha256": CHECKPOINT,
        "tu_split_sha256": SPLIT_HASHES["tuberlin"],
        "data_used_for_selection": list(SKETCHY_SPLITS),
        "selection_metric": "equal mean precision then equal mean mAP",
        "templates": templates,
        "generic_configurations_on_each_sketchy_split": len(catalog),
        "ledger_sha256": digest(ledger),
        "winners": report,
        "previous_sketchy_winners": list(PREVIOUS),
        "previous_sketchy_report": "https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/61cb327/docs/results/sketchy-colab-l4.csv",
    }
    lock["freeze_sha256"] = digest(lock)
    encoded = json.dumps(lock, indent=2, ensure_ascii=False) + "\n"
    if file.exists() and file.read_text() != encoded:
        raise ValueError("The existing TU-Berlin lock refers to a different Sketchy selection")
    if not file.exists():
        temp = file.with_suffix(".pending")
        temp.write_text(encoded)
        temp.replace(file)
    return lock


def fixed_previous(args, split, query_ids, q_labels, p_labels, raw, photos, legacy, transform, texts, meta):
    contours, ink, mirrored = legacy
    proxy = F.normalize(F.normalize(contours @ transform, dim=-1).mean(1), dim=-1)
    center = proxy.mean(0)
    source = centered(proxy, center)
    target = F.normalize(photos @ transform, dim=-1)
    target = centered(target, target.mean(0))
    data = (raw, ink, mirrored)
    with TemporaryDirectory(dir=args.output) as work:
        folder = Path(work)
        errors = anchor_errors(source, target)
        visual_index = image_index(photos, target, 192)
        score_setting(
            args,
            folder,
            split,
            (query_ids, *data),
            (q_labels, p_labels),
            (source, target, errors, center, transform),
            visual_index,
            (64, 1.0, 1.5, False),
            meta | {"external_vocabulary": 0, "method_family": "previous-visual"},
            write_result,
        )
        del visual_index
    words = gallery_word_codes(photos, texts[1]).half().float()
    indexes = [scale_index(photos, target, words, k) for _, k in EXPERTS]
    utility = calibrate(
        source, target, center, transform, photos, contours, texts, indexes, min(args.score_batch, 192)
    )
    difference = utility[:, 1, 0] - utility[:, 0, 0]
    near = nearest_gallery(source, 128)[0]
    cutoff = torch.quantile(difference[near].mean(1), 0.75)
    mask = []
    for ids in query_ids.split(256):
        views = torch.stack(
            (raw[ids].to(photos.device), F.normalize(ink[ids, 0].to(photos.device).float(), dim=-1)), dim=1
        )
        q = centered(F.normalize(views @ transform, dim=-1), center)
        ids_near = nearest_anchors(q.flatten(0, 1), source, 128).unflatten(0, q.shape[:2])
        mask.append(difference[ids_near].mean((1, 2)) > cutoff)
    choice = torch.cat(mask).cpu()
    with TemporaryDirectory(dir=args.output) as work:
        folder = Path(work)
        q = cache_reflection_features(folder, query_ids, data, source, target, center, transform, *texts)
        banks = rank_banks(args, folder, q, indexes, len(q_labels), len(p_labels), photos.device)

        def rows():
            for a, b, c in zip(banks[0].split(128), banks[1].split(128), choice.split(128), strict=True):
                yield from routed_tensor(a, b, c)

        write_result(
            args,
            split,
            PREVIOUS[1],
            rows(),
            q_labels,
            p_labels,
            meta | {"method_family": "previous-router"},
            {"utility": "neighbors100", "neighbors": 128, "offset": "q75"},
        )


@torch.inference_mode()
def run(workspace, stage, templates=DEFAULT_DOMAIN_TEMPLATES):
    if stage == "freeze":
        print(json.dumps(freeze(workspace, templates), indent=2))
        return
    heldout = stage == "heldout"
    dataset = "tuberlin" if heldout else "sketchy"
    lock = {}
    if heldout:
        lock = json.loads((workspace / "results/sketchy-to-tuberlin-lock.json").read_text())
        signed = lock.copy()
        signature = signed.pop("freeze_sha256")
        if (
            signature != digest(signed)
            or lock["revision"] != revision()
            or lock["templates"] != [list(p) for p in templates]
            or lock["data_used_for_selection"] != list(SKETCHY_SPLITS)
            or lock["previous_sketchy_winners"] != list(PREVIOUS)
        ):
            raise ValueError("TU-Berlin requires the original frozen Sketchy choices and source revision")
    args = runtime(workspace, dataset, batch=64, workers=2, crop_batch=192)
    args.crop_workers = 6
    args.output = workspace / "results" / ("domain-tuberlin-frozen" if heldout else "domain-sketchy")
    args.output.mkdir(parents=True, exist_ok=True)
    args.score_batch = 384
    transform = projection(args)
    queries, gallery, raw, photos, frames, legacy = inputs(args, dataset, prepare_sketch_controls=heldout)
    words = read_words(args.vocabulary)
    texts = tuple(x.cuda() for x in load_words(args, words, templates))
    for split in queries:
        if digest(UNSEEN_CLASSES[split]) != SPLIT_HASHES[split]:
            raise ValueError("Unseen class names or their order differ from KD-SBIR")
        args.metric_workers = 2 if split == "sketchy_2" else 4
        ids, q_labels = queries[split]
        pid, p_labels = gallery[split]
        full = raw[ids].cuda()
        seen_frames = torch.cat((full[:, None], frames[ids].cuda().float()), dim=1)
        seen_frames = F.normalize(seen_frames, dim=-1)
        p = photos[pid].cuda()
        meta = {
            "backbone": "OpenAI CLIP ViT-B/32",
            "checkpoint_sha256": CHECKPOINT,
            "gpu": torch.cuda.get_device_name(),
            "torch": str(torch.__version__),
            "revision": revision(),
            "exploratory_unseen": not heldout,
            "configuration_origin": "locked-on-sketchy" if heldout else "tuned-on-sketchy",
            "freeze_sha256": lock.get("freeze_sha256"),
            "external_vocabulary": len(words),
            "study": "cross-domain-prompt-gallery",
        }
        write_result(
            args,
            split,
            "native-clip",
            cosine_rows(full, p, 1),
            q_labels,
            p_labels,
            meta | {"external_vocabulary": 0},
            {"views": 1, "ranking": "raw cosine"},
        )
        imgs = [lock["winners"]["general_visual"]["name"]] if heldout else VISUAL
        labels = (q_labels, p_labels)
        visual_rows(
            args, split, seen_frames, p, transform, imgs, labels, meta | {"external_vocabulary": 0}, write_result
        )
        chosen = [lock["winners"]["general_semantic"]["name"]] if heldout else list(settings())
        semantic_rows(args, split, seen_frames, p, transform, *texts, chosen, labels, meta, write_result)
        if heldout:
            c, ink, mirror = legacy
            fixed_previous(
                args, split, ids, q_labels, p_labels, raw, p, (c[pid].cuda(), ink, mirror), transform, texts, meta
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--stage", choices=("sketchy", "freeze", "heldout"), required=True)
    parser.add_argument("--source-templates", nargs=2, default=DEFAULT_DOMAIN_TEMPLATES[0])
    parser.add_argument("--target-templates", nargs=2, default=DEFAULT_DOMAIN_TEMPLATES[1])
    args = parser.parse_args()
    templates = (tuple(args.source_templates), tuple(args.target_templates))
    if not all(t.count("{}") == 1 for group in templates for t in group):
        parser.error("Each domain template must have exactly one {} placeholder for a public term")
    run(args.workspace, args.stage, templates)
