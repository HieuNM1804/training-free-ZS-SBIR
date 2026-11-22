"""Restore the locked regional readout only for the sketches entering this study."""

import torch
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json, read
from .cross_view_gallery import run as build_gallery
from .cross_view_region_cache import two_pose_regions
from .cross_view_regions import CandidateRegions
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .data import Images, configure_image_worker, normal_transform
from .encoding import load_frozen_clip, save_tensor
from .regional_intervention_study import index as compact_index
from .visual_attribution_cache import contributions


def encode_selected(cfg, paths, ids, native):
    vision = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda")).visual
    loader = DataLoader(Images([paths[j] for j in ids.tolist()], normal_transform(), "rgb"),
                        batch_size=80, num_workers=5, worker_init_fn=configure_image_worker,
                        pin_memory=True, prefetch_factor=2)
    out = {"heads": [], "flip": [], "regions": []}
    cursor = 0
    worst = [0., 0.]
    for pixels in loader:
        pixels = pixels.cuda(non_blocking=True)
        pieces, embedding, errs = contributions(vision, pixels.to(vision.conv1.weight.dtype))
        native_batch = native[ids[cursor:cursor+len(pixels)]].cuda()
        worst[0] = max(worst[0], float((1-(embedding*native_batch).sum(1)).abs().max()))
        worst[1] = max(worst[1], float(errs[-1]))
        if worst[0] > .002 or worst[1] > .015:
            raise ValueError(f"Frozen direct readout changed {worst}")
        flipped, regions = two_pose_regions(vision, pixels)
        out["heads"].append(pieces[:, :48].cpu())
        out["flip"].append(flipped.cpu())
        out["regions"].append(regions.cpu())
        cursor += len(pixels)
        if cursor % 400 == 0:
            print("paired source views", cursor, len(ids), flush=True)
    if cursor != len(ids):
        raise ValueError("Missing pilot input embeddings")
    return {k: torch.cat(v) for k, v in out.items()}, worst


def prepare(w, family, ds, cfg, inputs, source, target, scale, samples):
    paths, queries, photo_paths, galleries, native, photo = inputs
    selected = torch.cat([queries[s][0][idx] for s, idx in samples.items()]).unique(sorted=True)
    folder = w / "cache" / family / ds
    stamp = digest({"model": CHECKPOINT, "selected": [str(paths[j]) for j in selected.tolist()],
                    "views": "13 corresponding two-pose regions, four native head blocks v1"})
    file = folder / ("source-" + stamp[:16] + ".pt")
    meta = file.with_suffix(".json")
    if file.is_file() and meta.is_file() and read(meta)["signature"] == stamp:
        parts = torch.load(file, weights_only=True, mmap=True)
    else:
        parts, worst = encode_selected(cfg, paths, selected, native)
        save_tensor(parts, file)
        atomic_json(meta, {"signature": stamp, "queries": len(selected),
                           "native_1_minus_cos": worst[0], "component_l2_error": worst[1]})
    build_gallery(w, ds)
    result, positions = {}, {}
    for s, ids in samples.items():
        qi, labels = queries[s]
        pi, gallery_labels = galleries[s]
        use = selected.searchsorted(qi[ids])
        if not torch.equal(selected[use], qi[ids]):
            raise ValueError("Pilot query mismatched its embedding")
        gallery = photo[pi].cuda()
        source_hash = digest({"gallery": [str(photo_paths[j]) for j in pi.tolist()],
                              "model": CHECKPOINT, "source": "regional-64/RGB16"})
        words, compact = compact_index(w, s, gallery, target, scale, source_hash[:16])
        historical = torch.load(w / "cache/pragmatic-gallery-competition" / (s+"-named-rgb16.pt"),
                                mmap=True, weights_only=True)
        reference = []
        for start in range(0, len(ids), 40):
            batch = slice(start, start+40)
            ix = use[batch]
            poses = parts["regions"][ix].cuda()
            reader = CandidateRegions(native[qi[ids[batch]]].cuda(), parts["flip"][ix].cuda(),
                                      poses[:, :, 0], poses[:, :, 1], source, words, compact,
                                      source.new_empty((0, len(source))), historical[ids[batch]].cuda())
            reader.prepare(scale)
            reference.append((.65*reader.first + .35*reader.read("self", 1., "gallery")).cpu())
        loc = torch.full((len(qi),), -1, dtype=torch.long)
        loc[ids] = torch.arange(len(ids))
        positions[s] = loc
        result[s] = ({"regional_score": torch.cat(reference), "query_native": native[qi],
                      "selected_heads": parts["heads"][use]}, labels, gallery_labels)
    return result, positions, read(meta)
