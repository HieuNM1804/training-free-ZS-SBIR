"""Select retrieval graph radii using local preservation of neighbors across paired domains."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from .contextual_rank import column_stochastic, jaccard_graph, neighborhood_overlap
from .contour_graph_index import gallery_walk, row_transition
from .contour_query_expansion import gallery_word_codes
from .gallery_diffusion import nearest_gallery
from .gallery_semantic_balance import query_evidence
from .global_methods import centered
from .reflection_transfer import cache_reflection_features, file_tensor
from .transfer_reconstruction import nearest_anchors
from .union_density import density_blocks, pack_affinity


RADII = (64, 112, 192, 320, 448)
FUSIONS = ("text35", "sharp")


def paired_preservation(source, target, radii):
    """Mid-rank the paired, leave-self-out Jaccard values within each gallery radius."""
    a = nearest_gallery(source, max(radii))[0]
    b = nearest_gallery(target, max(radii))[0]
    columns = []
    for k in radii:
        jaccard = torch.cat([neighborhood_overlap(x[:, :k], y[:, :k].sort(dim=1).values)
                             for x, y in zip(a.split(512), b.split(512), strict=True)])
        ordered = jaccard.sort().values
        lower = torch.searchsorted(ordered, jaccard)
        upper = torch.searchsorted(ordered, jaccard, right=True)
        columns.append((lower + upper).float() / (2 * len(source)))
    return torch.stack(columns, dim=1)


def query_reliability(query_ids, raw, extras, source, center, transform, bank):
    rng = torch.Generator(device=bank.device).manual_seed(731)
    shuffled = bank[torch.randperm(len(bank), device=bank.device, generator=rng)]
    scores, control = [], []
    for ids in query_ids.split(256):
        ink = F.normalize(extras[ids, 0].to(bank.device).float(), dim=1)
        views = torch.stack((raw[ids].to(bank.device), ink), dim=1)
        q = centered(F.normalize(views @ transform, dim=-1), center)
        matches = nearest_anchors(q.flatten(0, 1), source, 32).reshape(len(ids), -1)
        scores.append(bank[matches].mean(dim=1))
        control.append(shuffled[matches].mean(dim=1))
    return torch.cat(scores), torch.cat(control)


def radius_weights(confidence, temperature):
    if temperature is None:
        ties = confidence.eq(confidence.amax(dim=1, keepdim=True)).to(confidence.dtype)
        return ties / ties.sum(dim=1, keepdim=True)
    return (temperature * confidence).softmax(dim=1)


def scale_index(photos, target, vocabulary, k):
    _, *target_edges = jaccard_graph(target, k)
    words, *word_edges = jaccard_graph(vocabulary, k)
    _, *rgb_edges = jaccard_graph(photos, k)
    transition, dangling = row_transition(len(photos), (target_edges, word_edges, rgb_edges))
    smooth = gallery_walk(transition, dangling, target, .8, 4)
    visual, *smooth_edges = jaccard_graph(smooth, k)
    edges = (smooth_edges, word_edges, rgb_edges)
    return smooth, vocabulary, (visual, words), pack_affinity(len(photos), edges), column_stochastic(len(photos), edges)


def mixed_rows(banks, weights):
    for start in range(0, len(weights), 192):
        section = slice(start, start + 192)
        scores = torch.stack([bank[section].to(weights.device) for bank in banks], dim=1)
        combined = (scores * weights[section, :, None]).sum(dim=1)
        yield from combined.cpu()


def score_neighborhood_scale(args, split, ids, q_labels, p_labels, views, photos, contours, sketch_words,
                             photo_words, transform, metadata, record):
    proxy = F.normalize(F.normalize(contours @ transform, dim=-1).mean(dim=1), dim=-1)
    center = proxy.mean(dim=0)
    source = centered(proxy, center)
    target = F.normalize(photos @ transform, dim=-1)
    target = centered(target, target.mean(dim=0))
    vocabulary = gallery_word_codes(photos, photo_words).half().float()
    preservation = paired_preservation(source, target, RADII)
    confidence, shuffled = query_reliability(ids, views[0], views[1], source, center, transform, preservation)
    base = {"neighbors": 64, "penalty": 1., "carry": 1., "views": 8}
    device = target.device
    n, g = len(ids), len(photos)
    with TemporaryDirectory(dir=args.output) as folder:
        path = Path(folder)
        query = cache_reflection_features(path, ids, views, source, target, center, transform, sketch_words, photo_words)
        stored = {f: [] for f in FUSIONS}
        pending = []
        # Keep the GPU building the next graph while the single evaluator consumes completed, immutable score banks.
        with ThreadPoolExecutor(max_workers=1) as evaluator:
            for k in RADII:
                print(f"{split} paired neighborhoods k={k}", flush=True)
                index = scale_index(photos, target, vocabulary, k)
                banks = {f: file_tensor(path, f"scores-{k}-{f}", (n, g)) for f in FUSIONS}
                for start in range(0, n, args.score_batch):
                    loc = slice(start, start + args.score_batch)
                    tail, heads = query_evidence(*(q[loc].to(device) for q in query), index, modes=FUSIONS, compact=True)
                    for f in FUSIONS:
                        scores = torch.cat(list(density_blocks(tail, heads[f], len(tail), None)))
                        banks[f][loc] = scores.cpu()
                for f in FUSIONS:
                    stored[f].append(banks[f])
                    pending.append(evaluator.submit(record, args, split, f"scale-{f}-k{k}", iter(banks[f]), q_labels,
                                                    p_labels, metadata, base | {"fusion": f, "gallery_k": k}))
                del index
            for result in pending:
                result.result()
        options = [("uniform", confidence, 0.), ("paired2", confidence, 2.), ("paired8", confidence, 8.),
                   ("paired24", confidence, 24.), ("select", confidence, None), ("shuffled8", shuffled, 8.)]
        for f in FUSIONS:
            for name, affinity, temperature in options:
                weights = radius_weights(affinity, temperature)
                rows = mixed_rows(stored[f], weights)
                params = base | {"fusion": f, "gallery_k": list(RADII), "calibration": name, "temperature": temperature}
                record(args, split, f"scale-{f}-{name}", rows, q_labels, p_labels, metadata, params)
