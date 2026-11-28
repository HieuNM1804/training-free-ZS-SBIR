"""Audit reference-domain effects on query classes shared across Sketchy splits and on disjoint classes."""

import argparse
import json

import torch

from .concept_benchmark import atomic_json, checksum
from .description_block_results import Population
from .description_gallery_transfer import OUT
from .native_block_probe import SEED, deterministic
from .photo_name_interaction import verified
from .wordnet_renaming_pilot import SPLITS


def analyze(split, names, shared, visual_revision):
    path = OUT / f"visual-summary-{split}.pt"
    data = verified(path, visual_revision)
    labels = data["classes"]
    subjects = data["source_ids"]
    draw = data["metrics"]["drawing_centroid"]["rgb_centroid"]
    rgb = data["metrics"]["rgb_centroid"]["rgb_centroid"]
    draw_correct = data["choices"]["drawing_centroid"] == labels
    rgb_correct = data["choices"]["rgb_centroid"] == labels
    shared_mask = torch.tensor([name in shared for name in names])[labels]
    if len(names) != len(labels.unique()) or draw.shape != (len(labels), 2):
        raise ValueError("Unexpected class inventory or KD metric dimensions")
    margin = draw - rgb
    rows = {}
    for name, mask in (("common", shared_mask), ("unique", ~shared_mask)):
        pop = Population(labels[mask], subjects[mask])
        classes = labels[mask].unique()
        per_class = {}
        for cid in classes.tolist():
            take = labels == cid
            group = Population(labels[take], subjects[take])
            per_class[names[cid]] = group.effect(margin[take])["subject_class_points"]
        rows[name] = {
            "queries": int(mask.sum()), "subjects": len(subjects[mask].unique()), "classes": len(classes),
            "draw_minus_rgb_KD": pop.effect(margin[mask]),
            "draw_minus_semantic_KD": pop.effect((draw - data["semantic"])[mask]),
            "draw_minus_rgb_accuracy": pop.effect((draw_correct.double() - rgb_correct.double())[mask]),
            "draw_subject_class_accuracy": pop.effect(draw_correct[mask])["subject_class_points"],
            "rgb_subject_class_accuracy": pop.effect(rgb_correct[mask])["subject_class_points"],
            "positive_class_map": sum(effect[0] > 0 for effect in per_class.values()),
            "per_class_delta_map_precision": per_class,
        }
    return {"held_gallery": len(data["gallery_ids"]), "input_sha256": checksum(path), "groups": rows}


def main(revision, visual_revision, table_revision):
    deterministic()
    names = {}
    for split in SPLITS:
        info = json.loads((OUT / f"{split}-source_membership.json").read_bytes())
        if info["revision"] != table_revision or info["split"] != split:
            raise ValueError("Class names do not belong to the reference study")
        names[split] = info["classes"]
    shared = set.intersection(*(set(values) for values in names.values()))
    result = {s: analyze(s, names[s], shared, visual_revision) for s in SPLITS}
    atomic_json(OUT / "cross-split-classes.json", {
        "revision": revision, "seed": SEED, "visual_revision": visual_revision,
        "table_revision": table_revision, "common_classes": sorted(shared),
        "device": torch.cuda.get_device_name(),
        "analysis": "Post hoc subgrouping of queries only. Scoring gallery and original KD metrics held fixed.",
        "splits": result,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--visual-revision", required=True)
    parser.add_argument("--table-revision", required=True)
    args = parser.parse_args()
    main(args.revision, args.visual_revision, args.table_revision)
