"""Evaluate public-vocabulary transfer, without using benchmark class names."""

import argparse
import json
from pathlib import Path
import time

import torch

from .adaptive_fusion import adaptive_fusion_settings
from .anchor_context import anchor_context_settings
from .benchmark import merge_inputs
from .bimodal_graph import bimodal_settings
from .cluster_mine import cluster_mine_settings
from .cluster_prior import cluster_cache_settings, cluster_prior_settings, cluster_prototype_settings
from .compact_retrieval import compact_calibrated_settings, compact_retrieval_settings
from .contextual_rank import contextual_rank_settings
from .contour_render import load_xdog, xdog_transport_settings
from .coregistration import coregistration_settings
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .dense_views import dense_transport_settings, load_clearclip_gallery, tokencut_transport_settings
from .dominant_sets import dominant_set_settings
from .encoding import load_global_inputs
from .foundation import load_words
from .gallery_diffusion import gallery_diffusion_settings, two_stage_settings
from .gem_regions import gem_region_settings, load_gem_regions
from .head_selection import head_selection_settings
from .isoclip import iso_context_settings, iso_paired_settings, iso_transport_settings, isoclip_maps
from .isolated_regions import isolated_region_settings, load_isolated_regions
from .manifold import fused_features
from .manifold_benchmark import dot_rows
from .metrics import evaluate_rows
from .mta import iso_mta_settings, load_augmentations, mta_settings
from .multiscale import load_multiscale_patches, multiscale_settings
from .relevance import relevance_settings
from .score_calibration import calibrated_settings, normalized_score_rows, querybank_settings
from .signed_feedback import signed_feedback_settings
from .structured_transport import structured_transport_settings
from .semantic import (
    agreement_rows, agreement_settings, lexical_consensus_settings, paired_concept_settings, read_words,
    semantic_manifold_settings, semantic_settings,
)
from .semantic_transport import (
    conditional_similarity, diversified_transport_settings, partial_transport_settings,
    transport_rows, transport_settings, transport_view_settings,
)
from .union_density import density_fusion_settings, density_prefix_settings, density_settings
from .view_hypotheses import load_regional_sketches

GALLERY_STUDIES = {
    "agreement": agreement_settings,
    "consensus": lexical_consensus_settings,
    "paired-concepts": paired_concept_settings,
    "hubness": calibrated_settings,
    "querybank": querybank_settings,
    "cluster-mine": cluster_mine_settings,
    "transport": transport_settings,
}
CROP_STUDIES = {
    "transport-views": transport_view_settings,
    "partial-transport": partial_transport_settings,
    "diversified-transport": diversified_transport_settings,
}
DENSE_STUDIES = {
    "dense-transport": dense_transport_settings,
    "tokencut-transport": tokencut_transport_settings,
}
HEAD_STUDIES = ("head-selection", "view-hypotheses", "foreground-fusion", "gaussian-verification")
COMPACT_STUDIES = ("compact-retrieval", "compact-calibrated")
SCORE_STUDIES = {
    "gallery-diffusion": gallery_diffusion_settings,
    "two-stage-transport": two_stage_settings,
    "bimodal-graph": bimodal_settings,
    "adaptive-fusion": adaptive_fusion_settings,
    "coreg-spectral": coregistration_settings,
    "contextual-rank": contextual_rank_settings,
    "multiscale": multiscale_settings,
    "xdog-transport": xdog_transport_settings,
    "anchor-context": anchor_context_settings,
    "isoclip-transport": iso_transport_settings,
    "isolated-regions": isolated_region_settings,
    "iso-paired": iso_paired_settings,
    "iso-context": iso_context_settings,
    "union-density": density_settings,
    "union-prefix": density_prefix_settings,
    "union-fusion": density_fusion_settings,
    "signed-feedback": signed_feedback_settings,
    "finch-prior": cluster_prior_settings,
    "finch-prototype": cluster_prototype_settings,
    "finch-cache": cluster_cache_settings,
    "iso-mta": iso_mta_settings,
    "gem-regions": gem_region_settings,
    "dominant-sets": dominant_set_settings,
    "structured-transport": structured_transport_settings,
    "head-selection": head_selection_settings,
    "view-hypotheses": head_selection_settings,
    "foreground-fusion": head_selection_settings,
    "gaussian-verification": head_selection_settings,
    "compact-retrieval": compact_retrieval_settings,
    "compact-calibrated": compact_calibrated_settings,
    "lame-relevance": relevance_settings,
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", choices=UNSEEN_CLASSES, nargs="+", default=SKETCHY_SPLITS)
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--study", choices=("core", "manifold", "mta", *DENSE_STUDIES, *CROP_STUDIES, *GALLERY_STUDIES, *SCORE_STUDIES), default="core")
    parser.add_argument("--clip-checkpoint")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--crop-batch-size", type=int, help="Override the number of CLIP inputs in an augmentation batch.")
    parser.add_argument("--crop-workers", type=int, help="Override the augmentation data workers.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--metric-workers", type=int, default=4)
    parser.add_argument("--score-batch", type=int, default=256)
    parser.add_argument("--transport-buffer-mib", type=int, help="Memory budget for gathered gallery views in each transport batch.")
    parser.add_argument("--exploratory-unseen", action="store_true", required=True)
    args = parser.parse_args()
    if args.transport_buffer_mib is not None and args.transport_buffer_mib < 1:
        parser.error("--transport-buffer-mib must be positive")
    if args.crop_batch_size is not None and args.crop_batch_size < 1:
        parser.error("--crop-batch-size must be positive")
    if args.crop_workers is not None and args.crop_workers < 0:
        parser.error("--crop-workers must be nonnegative")
    return args


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    splits = args.splits
    query_paths, queries = merge_inputs(args.root, splits, "sketch")
    photo_paths, photos = merge_inputs(args.root, splits, "photo")
    all_q, all_p, contours = load_global_inputs(args, query_paths, photo_paths, args.study not in ("core", "manifold"))
    words = read_words(args.vocabulary)
    sketch_text, photo_text = [view.to(device) for view in load_words(args, words)]
    gem_regions = (
        load_gem_regions(args, query_paths, photo_paths, all_q, all_p, sketch_text, photo_text)
        if args.study in ("gem-regions", "dominant-sets", "structured-transport", *HEAD_STUDIES) else None
    )
    centers = (
        load_gem_regions(args, query_paths, photo_paths, all_q, all_p, sketch_text, photo_text, centers=True)
        if args.study == "structured-transport" else None
    )
    augmentations = (
        load_augmentations(args, query_paths)
        if args.study in ("mta", "iso-mta", "dominant-sets", "structured-transport", *HEAD_STUDIES) else None
    )
    regional_sketches = (
        load_regional_sketches(args, query_paths)
        if args.study in ("view-hypotheses", "foreground-fusion", "gaussian-verification", *COMPACT_STUDIES) else None
    )
    needs_crops = args.study in CROP_STUDIES or args.study in ("two-stage-transport", "adaptive-fusion")
    photo_views = load_augmentations(args, photo_paths, views=9, mode="photo") if needs_crops else None
    needs_dense = args.study in (*DENSE_STUDIES, "multiscale", "isolated-regions", "union-density")
    dense_views = load_clearclip_gallery(args, photo_paths) if needs_dense else None
    multiscale = load_multiscale_patches(args, query_paths, photo_paths) if args.study == "multiscale" else None
    isolated = load_isolated_regions(args, photo_paths, dense_views) if args.study in ("isolated-regions", "union-density") else None
    rendered = load_xdog(args, photo_paths) if args.study in ("xdog-transport", "union-density") else None
    iso_studies = (
        "isoclip-transport", "iso-paired", "iso-context", "union-density", "union-prefix", "union-fusion",
        "lame-relevance", "signed-feedback", "finch-prior", "finch-prototype", "finch-cache", "iso-mta", "gem-regions",
        "dominant-sets", "structured-transport", *COMPACT_STUDIES, *HEAD_STUDIES,
    )
    iso_maps = isoclip_maps(args, device, paired=args.study == "iso-paired") if args.study in iso_studies else None
    args.output.mkdir(parents=True, exist_ok=True)
    for split in splits:
        q_ids, q_labels = queries[split]
        p_ids, p_labels = photos[split]
        q, p = all_q[q_ids].to(device), all_p[p_ids].to(device)
        if args.study == "core":
            settings = {"rgb-cosine": (q, p)}
            settings.update(semantic_settings(q, p, sketch_text, photo_text))
            sq, sp = settings["visual-prototypes-0.3"]
            for blend in (0.15, 0.3):
                settings[f"semantic-plus-rgb-{blend:g}"] = fused_features(sq, sp, q, p, blend)
            variants = ((name, *pair) for name, pair in settings.items())
        elif args.study == "manifold":
            variants = semantic_manifold_settings(q, p, sketch_text, photo_text)
        elif args.study == "mta":
            variants = mta_settings(
                q, augmentations["features"][q_ids].to(device), augmentations["logit_scale"],
                p, contours[p_ids].to(device), sketch_text, photo_text,
            )
        elif args.study in CROP_STUDIES:
            variants = CROP_STUDIES[args.study](
                q, p, contours[p_ids].to(device), sketch_text, photo_text, photo_views["features"][p_ids].to(device),
            )
        elif args.study in DENSE_STUDIES:
            variants = DENSE_STUDIES[args.study](q, p, contours[p_ids].to(device), sketch_text, photo_text, dense_views[p_ids])
        elif args.study in SCORE_STUDIES:
            extra_inputs = (photo_views["features"][p_ids].to(device),) if needs_crops else ()
            extra_options = {}
            if args.study == "union-density":
                extra_inputs = (iso_maps, rendered[p_ids].to(device), isolated[p_ids])
            elif args.study in COMPACT_STUDIES:
                extra_inputs = (iso_maps, regional_sketches["features"][q_ids, 0].to(device))
            elif args.study == "iso-mta":
                extra_inputs = (iso_maps, augmentations["features"][q_ids].to(device), augmentations["logit_scale"])
            elif args.study in ("dominant-sets", *HEAD_STUDIES):
                extra_inputs = (iso_maps, augmentations["features"][q_ids].to(device), augmentations["logit_scale"],
                                gem_regions[0][q_ids], gem_regions[1][p_ids])
                if args.study in ("view-hypotheses", "gaussian-verification"):
                    extra_options["regional_views"] = regional_sketches["features"][q_ids].to(device)
                    extra_options["verify_crops"] = args.study == "gaussian-verification"
                elif args.study == "foreground-fusion":
                    extra_options["foreground_view"] = regional_sketches["features"][q_ids, 0].to(device)
            elif args.study == "structured-transport":
                extra_inputs = (iso_maps, augmentations["features"][q_ids].to(device), augmentations["logit_scale"],
                                gem_regions[0][q_ids], gem_regions[1][p_ids], centers[0][q_ids], centers[1][p_ids])
            elif gem_regions is not None:
                extra_inputs = (iso_maps, gem_regions[0][q_ids], gem_regions[1][p_ids])
            elif multiscale is not None:
                extra_inputs = (dense_views[p_ids], (multiscale[0][q_ids], multiscale[1][p_ids]))
            elif isolated is not None:
                extra_inputs = (isolated[p_ids],)
            elif rendered is not None:
                extra_inputs = (rendered[p_ids].to(device),)
            elif iso_maps is not None:
                extra_inputs = (iso_maps,)
            settings = SCORE_STUDIES[args.study](
                q, p, contours[p_ids].to(device), sketch_text, photo_text, *extra_inputs, args.score_batch, **extra_options,
            )
            variants = ((name, None, None, rows) for name, rows in settings)
        else:
            variants = GALLERY_STUDIES[args.study](q, p, contours[p_ids].to(device), sketch_text, photo_text)
        has_correction = args.study in ("hubness", "querybank", "transport", *DENSE_STUDIES, *CROP_STUDIES, *SCORE_STUDIES)
        scored = variants if has_correction else ((name, q, p, None) for name, q, p in variants)
        for name, target_q, target_p, correction in scored:
            filename = args.output / f"{split}-{name}.json"
            if filename.is_file():
                continue
            start = time.perf_counter()
            if args.study in SCORE_STUDIES:
                rows = correction()
            elif name.startswith(("conditional-transport-", "partial-transport-")):
                _, _, views, shortlist, _, *scorers = correction
                if args.transport_buffer_mib is None:
                    transport_batch = min(64, 384 // views.size(1), args.score_batch)
                else:
                    bytes_per_query = shortlist * views.size(1) * views.size(2) * views.element_size()
                    transport_batch = min(max(1, (args.transport_buffer_mib * 2**20) // bytes_per_query), args.score_batch)
                score = scorers[0] if scorers else conditional_similarity
                rows = transport_rows(target_q, target_p, *correction[:5], batch_size=transport_batch, score=score)
            elif correction is not None:
                offset, activation = correction
                rows = normalized_score_rows(target_q, target_p, offset, args.score_batch, activation)
            elif name.startswith("agreement-"):
                rows = agreement_rows(target_q, target_p, args.score_batch)
            else:
                rows = dot_rows(target_q, target_p, args.score_batch)
            ap, precision, map_k, p_k = evaluate_rows(
                rows, q_labels, p_labels, split, f"{split} {name}", workers=args.metric_workers,
            )
            result = {
                "split": split, "method": name, "backbone": "OpenAI CLIP ViT-B/32",
                "mAP": float(ap), "precision": float(precision), "map_k": map_k, "p_k": p_k,
                "queries": len(q), "gallery": len(p), "study": args.study,
                "external_vocabulary": 0 if args.study == "compact-retrieval" and name.startswith("compact-image-") else len(words),
                "exploratory_unseen": True, "evaluation_seconds": round(time.perf_counter() - start, 2),
            }
            filename.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    run(arguments())
