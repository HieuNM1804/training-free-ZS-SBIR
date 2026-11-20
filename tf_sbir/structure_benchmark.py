"""Test an image-only cross-domain retrieval graph and transfer its Sketchy-selected setting unchanged."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from .benchmark import merge_inputs
from .contextual_rank import jaccard_seed
from .contour_query_expansion import pagerank_tail
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT, FrameViews, projection, runtime
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .domain_prompt_gallery import visual_rows
from .encoding import load_global_inputs
from .gallery_semantic_balance import compact_union
from .global_methods import centered
from .mta import load_view_features
from .paired_neighborhood_scale import scale_index
from .reflection_transfer import file_tensor
from .reranking import standardize
from .transfer_benchmark import write_result
from .union_density import candidate_union, density_rows, grow_density
from .universal_structure import structure_bank


NO_STRUCTURE = "struct-none-agreement"


def configurations():
    modes = {}
    for layer, k in (("both", 192), ("early", 192), ("late", 192), ("both", 112), ("both", 320)):
        for option in ("w010", "w030", "w050", "adaptive"):
            modes[f"struct-{layer}-k{k}-{option}"] = (layer, k, option)
    modes["struct-both-k192-single"] = ("both", 192, "single")
    modes["struct-shuffled-k192-w030"] = ("shuffled", 192, "w030")
    return modes


def select_layer(features, layer):
    if layer == "early":
        return features[..., 0, :].float()
    if layer == "late":
        return features[..., 1, :].float()
    return F.normalize(features.flatten(-2).float(), dim=-1)


def rank_structure(qv, qs, index, modes):
    gallery, descriptors, banks, graph, walk = index
    visual = qv @ gallery.T
    structural = qs @ descriptors.T
    image_votes = torch.stack([jaccard_seed(s, banks[0]) for s in visual.unbind(1)], dim=1)
    shape_votes = torch.stack([jaccard_seed(s, banks[1]) for s in structural.unbind(1)], dim=1)
    appearance, pattern = standardize(visual).mean(1), standardize(structural).mean(1)
    initial = standardize(appearance) + 0.35 * standardize(pattern)
    tail = pagerank_tail(appearance, pattern, initial, banks, *walk, len(qv))
    candidates = compact_union(
        *candidate_union((*visual.unbind(1), *structural.unbind(1)), min(1280, len(gallery)))
    )

    def weighted_votes(source):
        mass = source.sum(dim=-1)
        weights = (mass.clamp_min(1e-8).log() * 2).softmax(1)
        return (weights[:, :, None] * source).sum(1), (weights * mass).sum(1)

    a, am = weighted_votes(image_votes)
    b, bm = weighted_votes(shape_votes)
    heads = {}
    for mode in modes:
        if mode == "adaptive":
            fraction = (bm / (bm + am).clamp_min(1e-8))[:, None]
        else:
            fraction = {"w010": 0.1, "w030": 0.3, "w050": 0.5, "single": 0.3}[mode]
        seed = a.lerp(b, fraction)
        heads[mode] = grow_density(64 * seed, initial, *candidates, graph, min(256, len(gallery)))
    return tail, heads


def group_rows(args, split, qv, qs, index, k, layer, names, labels, meta):
    q_labels, p_labels = labels
    n, g = len(q_labels), len(p_labels)
    depth = min(256, g)
    with TemporaryDirectory(dir=args.output) as work:
        folder = Path(work)
        tail = file_tensor(folder, "tail", (n, g))
        heads = {
            mode: torch.from_file(
                str(folder / (mode + ".i32")), shared=True, size=n * depth, dtype=torch.int32
            ).view(n, depth)
            for _, mode in names
        }
        for start in range(0, n, args.score_batch):
            block = slice(start, start + args.score_batch)
            t, h = rank_structure(qv[block], qs[block], index, list(heads))
            tail[block] = t.cpu()
            for mode in heads:
                heads[mode][block] = h[mode].to("cpu", dtype=torch.int32)
        for name, mode in names:

            def rows():
                for a, b in zip(tail.split(256), heads[mode].split(256), strict=True):
                    yield from density_rows(a.cuda(), b.cuda().long(), len(a), None)

            write_result(
                args,
                split,
                name,
                rows(),
                q_labels,
                p_labels,
                meta,
                {"layer": layer, "gallery_k": k, "fusion": mode, "views": qv.size(1)},
            )


def freeze_structure(workspace):
    folder = workspace / "results/structure-sketchy"
    built = {}
    for name in (NO_STRUCTURE, *configurations()):
        result = {}
        for split in SKETCHY_SPLITS:
            r = json.loads((folder / f"{split}-{name}.json").read_text())
            rows = torch.load(folder / "per-query" / f"{split}-{name}.pt", map_location="cpu", weights_only=True)
            expected = (14179, 17100, 100, 0) if split == "sketchy_1" else (12694, 12552, 200, 200)
            if (
                r["revision"] != revision()
                or not r["exploratory_unseen"]
                or r["checkpoint_sha256"] != CHECKPOINT
                or r["external_vocabulary"] != 0
                or (r["queries"], r["gallery"], r["p_k"], r["map_k"]) != expected
                or rows.shape != (expected[0], 2)
                or not torch.allclose(rows.mean(0), torch.tensor([r["mAP"], r["precision"]]), atol=5e-7, rtol=0)
            ):
                raise ValueError(f"Invalid Sketchy result for {name}")
            result[split] = {k: r[k] for k in ("mAP", "precision", "parameters")}
        built[name] = result

    def score(name):
        return tuple(sum(built[name][s][m] for s in SKETCHY_SPLITS) / 2 for m in ("precision", "mAP"))

    best = max((n for n in built if "shuffled" not in n), key=score)
    best_structure = max((n for n in configurations() if "shuffled" not in n), key=score)
    lock = {
        "revision": revision(),
        "checkpoint_sha256": CHECKPOINT,
        "selection_splits": list(SKETCHY_SPLITS),
        "ledger_sha256": digest(built),
        "best_including_ablation": best,
        "best_using_structure": best_structure,
        "heldout_controls": [NO_STRUCTURE],
        "selection_metric": "equal mean precision then equal mean mAP",
        "parameters": {n: built[n]["sketchy_1"]["parameters"] for n in (NO_STRUCTURE, best, best_structure)},
    }
    lock["freeze_sha256"] = digest(lock)
    path = workspace / "results/structure-sketchy-to-tuberlin-lock.json"
    payload = json.dumps(lock, indent=2) + "\n"
    if path.exists() and path.read_text() != payload:
        raise ValueError("A different frozen structure setting already exists")
    if not path.exists():
        temp = path.with_suffix(".pending")
        temp.write_text(payload)
        temp.replace(path)
    return lock


@torch.inference_mode()
def run(workspace, stage):
    if stage == "freeze":
        print(json.dumps(freeze_structure(workspace), indent=2))
        return
    held = stage == "heldout"
    dataset = "tuberlin" if held else "sketchy"
    if held:
        lock = json.loads((workspace / "results/structure-sketchy-to-tuberlin-lock.json").read_text())
        signed = lock.copy()
        sig = signed.pop("freeze_sha256")
        if (
            sig != digest(signed)
            or lock["revision"] != revision()
            or lock["selection_splits"] != list(SKETCHY_SPLITS)
        ):
            raise ValueError("The structure configuration must come from the two Sketchy splits")
        if lock["heldout_controls"] != [NO_STRUCTURE]:
            raise ValueError("The visual-only control must be preserved")
        requested = list(
            dict.fromkeys(
                (*lock["heldout_controls"], lock["best_including_ablation"], lock["best_using_structure"])
            )
        )
    else:
        lock = {}
        requested = [NO_STRUCTURE, *configurations()]
    args = runtime(workspace, dataset, batch=64, workers=2, crop_batch=192)
    args.crop_workers = 6
    args.score_batch = 256
    args.output = workspace / "results" / ("structure-tuberlin-frozen" if held else "structure-sketchy")
    args.output.mkdir(parents=True, exist_ok=True)
    splits = ("tuberlin",) if held else SKETCHY_SPLITS
    q_paths, q_idx = merge_inputs(args.root, splits, "sketch")
    p_paths, p_idx = merge_inputs(args.root, splits, "photo")
    raw, p_all, _ = load_global_inputs(args, q_paths, p_paths, False)
    f = load_view_features(args, FrameViews(q_paths), 5, "sketch", "neutral-frame-reflect-center180-144-v1")[
        "features"
    ]
    shape_q = structure_bank(args, q_paths, "sketch", 4)
    shape_p = structure_bank(args, p_paths, "photo", 1)
    proj = projection(args)
    for split in splits:
        if digest(UNSEEN_CLASSES[split]) != SPLIT_HASHES[split]:
            raise ValueError(split)
        args.metric_workers = 2 if split == "sketchy_2" else 4
        ids, ql = q_idx[split]
        pids, pl = p_idx[split]
        photos = p_all[pids].cuda()
        query = F.normalize(torch.cat((raw[ids, None], f[ids, :3].float()), dim=1).cuda(), dim=-1)
        meta = {
            "backbone": "OpenAI CLIP ViT-B/32",
            "checkpoint_sha256": CHECKPOINT,
            "gpu": torch.cuda.get_device_name(),
            "torch": str(torch.__version__),
            "revision": revision(),
            "exploratory_unseen": not held,
            "external_vocabulary": 0,
            "study": "universal-self-similarity",
            "configuration_origin": "locked-on-sketchy" if held else "tuned-on-sketchy",
            "freeze_sha256": lock.get("freeze_sha256"),
        }
        if NO_STRUCTURE in requested:
            visual_rows(args, split, query, photos, proj, [NO_STRUCTURE], (ql, pl), meta, write_result)
        target = F.normalize(photos @ proj, dim=-1)
        mean = target.mean(0)
        target = centered(target, mean)
        qv = centered(F.normalize(query @ proj, dim=-1), mean)
        groups = {}
        for name in requested:
            if name != NO_STRUCTURE:
                layer, k, mode = configurations()[name]
                groups.setdefault((layer, k), []).append((name, mode))
        for (layer, k), names in groups.items():
            gd = select_layer(shape_p[pids, 0].cuda(), layer)
            qm = gd.mean(0)
            gd = centered(gd, qm)
            qd = centered(select_layer(shape_q[ids].cuda(), layer), qm)
            if layer == "shuffled":
                perm = torch.randperm(
                    len(gd), device=gd.device, generator=torch.Generator(device=gd.device).manual_seed(731)
                )
                gd = gd[perm]
            index = scale_index(photos, target, gd, min(k, len(photos) - 1))
            for one in (False, True):
                group = [pair for pair in names if (pair[1] == "single") == one]
                if group:
                    top = 1 if one else query.size(1)
                    group_rows(args, split, qv[:, :top], qd[:, :top], index, k, layer, group, (ql, pl), meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sketchy", "freeze", "heldout"), required=True)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    options = parser.parse_args()
    run(options.workspace, options.stage)
