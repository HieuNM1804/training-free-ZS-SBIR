"""Full-gallery paired-domain transfer with query scores streamed in bounded CUDA batches."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from .benchmark import merge_inputs
from .capture_query_metrics import evaluate_with_details
from .contour_query_expansion import pagerank_tail
from .data import SKETCHY_SPLITS
from .encoding import load_global_inputs, save_tensor
from .foundation import load_words
from .global_methods import centered
from .isoclip import isoclip_maps
from .mta import load_view_features
from .multiview_anchor_transfer import predict_views, transfer_prefixes
from .predictable_gallery import gallery_variants
from .reranking import standardize
from .semantic import read_words
from .sketch_canvas_fusion import canvas_images
from .transfer_reconstruction import anchor_errors, gallery_index, nearest_anchors, transferred_ranking
from .union_density import density_rows
from .view_hypotheses import ink_box_image, view_concepts


VIEW_WEIGHTS = {"fixed": (0., 0.), "cohesion": (4., 0.), "risk1": (4., 1.),
                "risk4": (4., 4.), "shuffled": (4., 4.), "risk-only": (0., 2.)}


class AnchorSketchViews(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            ink = ink_box_image(image.convert("RGB"))
        return torch.stack([pil_to_tensor(view) for view in (ink, *canvas_images(ink))])


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--vocabulary", required=True, type=Path)
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--splits", nargs="+", choices=SKETCHY_SPLITS, default=SKETCHY_SPLITS)
    parser.add_argument("--stage", choices=("references", "views", "prepare", "projector", "gallery"), default="references")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--crop-batch-size", type=int, default=128)
    parser.add_argument("--score-batch", type=int, default=160)
    parser.add_argument("--metric-workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--exploratory-unseen", action="store_true", required=True)
    args = parser.parse_args()
    if min(args.batch_size, args.crop_batch_size, args.score_batch, args.metric_workers) <= 0 or args.workers < 0:
        parser.error("Batch sizes and metric workers must be positive, and image workers must be nonnegative")
    return args


def ranked_rows(mapped, risks, concepts, index, mode, score_batch):
    gallery, vocabulary, banks, graph, walk = index
    for vec, risk, text in zip(mapped.split(score_batch), risks.split(score_batch), concepts.split(score_batch), strict=True):
        lexical = tuple(text[:, i] @ vocabulary.T for i in (0, 1))
        if mode == "two":
            pair = torch.cat((vec[:, 0], vec[:, 1]))
            yield from transferred_ranking(pair, gallery, lexical, banks, graph, walk, score_batch)()
            continue
        appearance = standardize(vec[:, 0] @ gallery.T).lerp(standardize(vec[:, 1] @ gallery.T), .35)
        language = standardize(lexical[0]).lerp(standardize(lexical[1]), .65)
        initial = standardize(appearance) + standardize(language)
        tail = pagerank_tail(appearance, language, initial, banks, *walk, score_batch)
        power, risk_power = VIEW_WEIGHTS[mode]
        prefix = transfer_prefixes(vec, text, risk, gallery, vocabulary, banks, graph, initial,
                                   [(mode, power, risk_power)], score_batch)[mode]
        yield from density_rows(tail, prefix, score_batch, None)


def settings(stage):
    if stage == "references":
        return [(64, 1., 1., ("two",)), (64, 1., 1.5, ("two",)), (64, 0., 1., ("two",)),
                (16, 1., 1., ("two",))]
    return [(64, 1., 1., ("cohesion", "risk1", "risk4", "shuffled", "fixed", "risk-only")),
            (64, 1., 1.5, ("cohesion", "risk1", "risk4")), (16, 1., 1.5, ("cohesion", "risk4"))]


def write_result(args, split, name, rows, q_labels, p_labels, common, parameters):
    path = args.output / f"{split}-{name}.json"
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    (ap, precision, map_k, p_k), details = evaluate_with_details(
        rows, q_labels, p_labels, split, f"{split} {name}", args.metric_workers,
    )
    detail_dir = args.output / "per-query"
    save_tensor(details, detail_dir / f"{split}-{name}.pt")
    result = {**common, "split": split, "method": name, "parameters": parameters, "queries": len(q_labels),
              "gallery": len(p_labels), "mAP": float(ap), "precision": float(precision), "map_k": map_k, "p_k": p_k,
              "evaluation_seconds": round(time.perf_counter() - start, 2),
              "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20)}
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(result), flush=True)


def score_split(args, split, raw, photos, contours, q_labels, p_labels, sketch_words, photo_words, transform, metadata):
    device = photos.device
    proxies = F.normalize(F.normalize(contours @ transform, dim=-1).mean(dim=1), dim=-1)
    source_mean = proxies.mean(dim=0)
    source = centered(proxies, source_mean)
    target = F.normalize(photos @ transform, dim=-1)
    target = centered(target, target.mean(dim=0))
    views = centered(F.normalize(raw @ transform, dim=-1), source_mean)
    concepts = torch.cat([view_concepts(chunk, sketch_words, photo_words) for chunk in raw.split(max(1, args.score_batch // raw.size(1)))])
    support = nearest_anchors(views.flatten(0, 1), source, 64)
    errors = anchor_errors(source, target)
    print(f"Building the complete CUDA gallery index for {split}", flush=True)
    index, smoothing = gallery_index(photos, target, photo_words, return_smoothing=True)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    print(f"Gallery indexed using {torch.cuda.memory_allocated(device)/2**20:.0f} MiB of CUDA tensors", flush=True)
    if args.stage == "gallery":
        mapped, risk = predict_views(views, source, target, errors, support, count=64, penalty=1., carry=1.)
        base = {"neighbors": 64, "penalty": 1., "carry": 1., "views": 2, "fusion": "two"}
        def measure(name, gallery, options):
            adjusted = (gallery, *index[1:])
            scores = ranked_rows(mapped, risk, concepts, adjusted, "two", args.score_batch)
            write_result(args, split, name, scores, q_labels, p_labels, metadata, base | options)
        measure("predictable-reference", index[0], {"strength": 0.})
        diagnostics = {}
        for name, gallery, options in gallery_variants(source, target, smoothing, diagnostics):
            measure(name, gallery, options)
        (args.output / f"{split}-reliability-summary.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        return
    for count, penalty, carry, modes in settings(args.stage):
        mapped, risk = predict_views(views, source, target, errors, support, count=count, penalty=penalty, carry=carry)
        for mode in modes:
            name = f"transfer-{mode}-k{count}-g{penalty:g}-r{carry:g}"
            params = {"neighbors": count, "penalty": penalty, "carry": carry, "views": raw.size(1), "fusion": mode}
            write_result(args, split, name, ranked_rows(mapped, risk, concepts, index, mode, args.score_batch),
                         q_labels, p_labels, metadata, params)


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.clip_checkpoint, "rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af":
        raise ValueError("This comparison requires the original OpenAI ViT-B/32 checkpoint")
    args.memory_map = True
    gpu_name = torch.cuda.get_device_name().replace(" ", "-")
    projection_path = args.cache / f"iso-b32-150-50-{digest[:12]}-{torch.__version__}-{gpu_name}.pt"
    if not projection_path.is_file():
        projection = isoclip_maps(args, device, bands=((150, 50),))[(150, 50)]
        save_tensor(projection.cpu(), projection_path)
        del projection
    if args.stage == "projector":
        print(f"Verified spectral map {projection_path.name}", flush=True)
        return
    query_paths, queries = merge_inputs(args.root, args.splits, "sketch")
    photo_paths, photos = merge_inputs(args.root, args.splits, "photo")
    all_q, all_p, contours = load_global_inputs(args, query_paths, photo_paths, True)
    extras = load_view_features(args, AnchorSketchViews(query_paths), 4, "sketch", "ink-scales-mirror-v1")["features"]
    terms = read_words(args.vocabulary)
    sketch_words, photo_words = [v.to(device) for v in load_words(args, terms)]
    if args.stage == "prepare":
        return
    transform = torch.load(projection_path, map_location=device, weights_only=True)
    torch.cuda.empty_cache()
    root = Path(__file__).resolve().parents[1]
    git_id = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    metadata = {"backbone": "OpenAI CLIP ViT-B/32", "checkpoint_sha256": digest, "gpu": torch.cuda.get_device_name(),
                "torch": str(torch.__version__), "revision": git_id, "exploratory_unseen": True,
                "external_vocabulary": len(terms), "study": "predictable-gallery" if args.stage == "gallery" else "multiview-anchor-transfer"}
    for split in args.splits:
        q_ids, q_labels = queries[split]
        p_ids, p_labels = photos[split]
        ink = F.normalize(extras[q_ids, :1].to(device).float(), dim=-1)
        raw = torch.cat((ink, all_q[q_ids].to(device)[:, None]), dim=1)
        if args.stage == "views":
            raw = torch.cat((raw, F.normalize(extras[q_ids, 1:].to(device).float(), dim=-1)), dim=1)
        score_split(args, split, raw, all_p[p_ids].to(device), contours[p_ids].to(device), q_labels, p_labels,
                    sketch_words, photo_words, transform, metadata)


if __name__ == "__main__":
    run(arguments())
