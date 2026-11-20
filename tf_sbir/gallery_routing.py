"""Choose a retrieval ranking by estimating its paired-gallery performance near one sketch."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import torch
from torch.nn import functional as F

from .contextual_rank import jaccard_seed
from .contour_query_expansion import gallery_word_codes
from .encoding import save_tensor
from .gallery_diffusion import nearest_gallery
from .gallery_semantic_balance import compact_union, query_evidence
from .global_methods import centered
from .multiview_anchor_transfer import evidence_weights, predict_views
from .paired_neighborhood_scale import scale_index
from .reflection_transfer import cache_reflection_features, file_tensor
from .reranking import standardize
from .transfer_reconstruction import anchor_errors, nearest_anchors
from .union_density import candidate_union, density_blocks, grow_density
from .view_hypotheses import view_concepts


EXPERTS = (("text35", 192), ("sharp", 320))
UTILITIES = ("neighbors100", "neighbors200", "instance")


def probe_utilities(prefix, gallery_ids, reference):
    """Compare probe rankings against its paired image and that image's independent RGB neighbors."""
    keep = prefix.ne(gallery_ids[:, None])
    position = keep.cumsum(dim=1)
    utility = []
    for cutoff in (100, 200):
        truth = reference[:, :cutoff].sort(dim=1).values
        pos = torch.searchsorted(truth, prefix.contiguous()).clamp_max(cutoff - 1)
        correct = truth.gather(1, pos).eq(prefix) & keep & position.le(cutoff)
        utility.append(correct.sum(dim=1).float() / cutoff)
    rank = torch.arange(1, prefix.size(1) + 1, device=prefix.device)
    utility.append((~keep / rank).amax(dim=1))
    return torch.stack(utility, dim=1)


def probe_rankings(mapped, uncertainty, concepts, index, expert):
    smooth, words, banks, graph, _ = index
    visual, lexical = mapped @ smooth.T, concepts @ words.T
    appearance = torch.stack([jaccard_seed(s, banks[0]) for s in visual.unbind(1)], dim=1)
    language = torch.stack([jaccard_seed(s, banks[1]) for s in lexical.unbind(1)], dim=1)
    initial = visual.new_full((2,), .5)
    if expert == "text35":
        weights = evidence_weights(appearance.sum(dim=-1), uncertainty.sum(dim=-1), initial, 0., 4.)
        seed = 2. * ((weights[:, :, None] * appearance).sum(1).lerp(language.mean(1), .35))
    else:
        votes = appearance + language
        weights = evidence_weights(votes.sum(dim=-1), uncertainty[..., 0], initial, 4., 4.)
        seed = (weights[:, :, None] * votes).sum(1)
    tied = standardize(standardize(visual).mean(1)) + standardize(standardize(lexical).mean(1))
    candidates = compact_union(*candidate_union((*visual.unbind(1), *lexical.unbind(1)), 1280))
    return grow_density(32. * seed, tied, *candidates, graph, 256)


def calibrate(source, target, center, transform, photos, contours, text_banks, indexes, batch_size):
    n = len(photos)
    device = photos.device
    reference = nearest_gallery(centered(photos, photos.mean(dim=0)), 200)[0]
    rng = torch.Generator(device=device).manual_seed(971)
    # Partition independently of filesystem order and category. Each proxy uses only other folds as anchor pairs.
    fold = torch.empty(n, dtype=torch.long, device=device)
    fold[torch.randperm(n, device=device, generator=rng)] = torch.arange(n, device=device) % 4
    values = target.new_empty((n, len(EXPERTS), len(UTILITIES)))
    for number in range(4):
        keep = fold.ne(number)
        keys, targets = source[keep], target[keep]
        errors = anchor_errors(keys, targets)
        for ids in (~keep).nonzero().flatten().split(batch_size):
            raw = F.normalize(contours[ids].float(), dim=-1)
            query = centered(F.normalize(raw @ transform, dim=-1), center)
            near = nearest_anchors(query.flatten(0, 1), keys, 64)
            mapped, risk = predict_views(query, keys, targets, errors, near, count=64, penalty=1., carry=1., risk_components=True)
            words = view_concepts(raw, *text_banks)
            for j, ((name, _), index) in enumerate(zip(EXPERTS, indexes, strict=True)):
                head = probe_rankings(mapped, risk, words, index, name)
                values[ids, j] = probe_utilities(head, ids, reference[ids])
    return values


def choose_experts(gallery_performance, gallery_neighbors, query_neighbors):
    delta = gallery_performance[:, 1] - gallery_performance[:, 0]
    rng = torch.Generator(device=delta.device).manual_seed(731)
    shuffled = delta[torch.randperm(len(delta), device=delta.device, generator=rng)]
    choices, parameters = {}, {}
    for k in (8, 32, 128):
        gallery = delta[gallery_neighbors[:, :k]].mean(dim=1)
        support = query_neighbors[:, :, :k]
        local = delta[support].mean(dim=(1, 2))
        for i, measure in enumerate(UTILITIES):
            thresholds = {"zero": 0.}
            for p in (10, 25, 50, 75, 90):
                thresholds[f"q{p}"] = float(torch.quantile(gallery[:, i], p / 100.))
            for origin, cutoff in thresholds.items():
                tag = f"route-{measure}-k{k}-{origin}"
                choices[tag] = local[:, i] > cutoff
                parameters[tag] = {"utility": measure, "neighbors": k, "offset": origin}
            if k == 32:
                tag = f"route-{measure}-shuffled"
                threshold = shuffled[gallery_neighbors[:, :k], i].mean(1).median()
                choices[tag] = shuffled[support, i].mean(dim=(1, 2)) > threshold
                parameters[tag] = {"utility": measure, "neighbors": k, "offset": "q50", "shuffled": True}
    return choices, parameters


def routed_tensor(first, second, choose_second):
    """Return the chosen expert's unmodified score row for each independent query."""
    return torch.where(choose_second[:, None].to(first.device), second, first)


def rank_banks(args, path, query, indexes, n, g, device):
    banks = [file_tensor(path, name + "-scores", (n, g)) for name, _ in EXPERTS]
    for start in range(0, n, args.score_batch):
        loc = slice(start, start + args.score_batch)
        vectors, risk, words = [t[loc].to(device) for t in query]
        for ((name, _), index, bank) in zip(EXPERTS, indexes, banks, strict=True):
            tail, heads = query_evidence(vectors, risk, words, index, modes=(name,), compact=True)
            bank[loc] = torch.cat(list(density_blocks(tail, heads[name], len(tail), None))).cpu()
    return banks


def evaluate_routes(args, split, choices, parameters, banks, labels, metadata, record):
    q_labels, p_labels = labels
    detail_dir = args.output / "per-query"
    parents = []
    for ((name, size), scores) in zip(EXPERTS, banks, strict=True):
        tag = "route-fixed-" + name
        record(args, split, tag, iter(scores), q_labels, p_labels, metadata, {"expert": name, "gallery_k": size})
        parents.append(torch.load(detail_dir / f"{split}-{tag}.pt", map_location="cpu", weights_only=True))
    verified = "route-neighbors100-k32-zero"
    for name, selection in choices.items():
        mask = selection.cpu()
        rows = routed_tensor(*parents, mask)
        params = parameters[name]
        if name == verified:
            def actual_scores():
                for a, b, c in zip(banks[0].split(128), banks[1].split(128), mask.split(128), strict=True):
                    yield from routed_tensor(a, b, c)
            record(args, split, name, actual_scores(), q_labels, p_labels, metadata, params)
            direct = torch.load(detail_dir / f"{split}-{name}.pt", map_location="cpu", weights_only=True)
            if not torch.equal(rows, direct):
                raise ValueError("Selecting official per-query metrics did not reproduce evaluating the selected ranking.")
            continue
        result = dict(metadata, split=split, method=name, parameters=params, queries=len(q_labels), gallery=len(p_labels),
                      mAP=float(rows[:, 0].contiguous().mean()), precision=float(rows[:, 1].contiguous().mean()),
                      map_k=0 if split=="sketchy_1" else 200, p_k=100 if split=="sketchy_1" else 200,
                      evaluation="Selecting each independent query's previously evaluated full score row")
        save_tensor(rows, detail_dir / f"{split}-{name}.pt")
        file = args.output / f"{split}-{name}.json"
        pending = file.with_suffix(".pending")
        pending.write_text(json.dumps(result, indent=2) + "\n")
        pending.replace(file)


def score_gallery_routes(args, split, ids, q_labels, p_labels, views, photos, contours, sketch_words, photo_words, transform, metadata, record):
    proxy = F.normalize(F.normalize(contours @ transform, dim=-1).mean(dim=1), dim=-1)
    center = proxy.mean(0)
    source = centered(proxy, center)
    target = F.normalize(photos @ transform, dim=-1)
    target = centered(target, target.mean(0))
    vocabulary = gallery_word_codes(photos, photo_words).half().float()
    indexes = [scale_index(photos, target, vocabulary, size) for _, size in EXPERTS]
    start = time.perf_counter()
    utility = calibrate(source, target, center, transform, photos, contours, (sketch_words, photo_words), indexes, args.score_batch)
    gallery_neighbors = nearest_gallery(source, 128)[0]
    query_neighbors = []
    for selected in ids.split(256):
        full = views[0][selected].to(photos.device)
        ink = F.normalize(views[1][selected, 0].to(photos.device).float(), dim=-1)
        raw = torch.stack((full, ink), dim=1)
        q = centered(F.normalize(raw @ transform, dim=-1), center)
        query_neighbors.append(nearest_anchors(q.flatten(0, 1), source, 128).view(len(selected), 2, 128))
    choices, params = choose_experts(utility, gallery_neighbors, torch.cat(query_neighbors))
    # Freeze the choices before reading either expert's outcome on any real sketch.
    save_tensor({k:v.cpu() for k, v in choices.items()}, args.output / f"{split}-routing-choices.pt")
    logged = {"proxy_queries": len(photos), "calibration_seconds": time.perf_counter()-start,
              "utilities": list(UTILITIES), "mean_gallery_utility": utility.mean(0).tolist(),
              "fraction_joint": {k:float(v.float().mean()) for k,v in choices.items()}}
    (args.output / f"{split}-calibration.json").write_text(json.dumps(logged, indent=2))
    with TemporaryDirectory(dir=args.output) as folder:
        path = Path(folder)
        query = cache_reflection_features(path, ids, views, source, target, center, transform, sketch_words, photo_words)
        banks = rank_banks(args, path, query, indexes, len(ids), len(photos), photos.device)
        evaluate_routes(args, split, choices, params, banks, (q_labels, p_labels), metadata, record)
