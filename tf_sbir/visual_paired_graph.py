"""Evaluate the paired-domain graph using only the images of one query and its gallery."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from .contextual_rank import column_stochastic, jaccard_graph, jaccard_seed, personalized_walk
from .contour_graph_index import gallery_walk, row_transition
from .global_methods import centered
from .multiview_anchor_transfer import evidence_weights, predict_views
from .reflection_transfer import PairSetting, file_tensor, reflection_seed
from .reranking import standardize
from .transfer_reconstruction import anchor_errors, nearest_anchors
from .union_density import candidate_union, density_rows, grow_density, pack_affinity


ANCHORS = [(64, 1., 1., False), (16, 1., 1., False), (32, 1., 1., False), (128, 1., 1., False),
           (64, 0., 1., False), (64, 4., 1., False), (64, 1., .5, False), (64, 1., 1.5, False),
           (64, 1., 1., True)]


def image_index(photos, target, neighbors):
    _, *target_edges = jaccard_graph(target, neighbors)
    _, *photo_edges = jaccard_graph(photos, neighbors)
    transition, isolated = row_transition(len(photos), (target_edges, photo_edges))
    smooth = gallery_walk(transition, isolated, target, .8, 4)
    bank, *edges = jaccard_graph(smooth, neighbors)
    graphs = (edges, photo_edges)
    return smooth, bank, pack_affinity(len(photos), graphs), column_stochastic(len(photos), graphs)


def visual_tail(scores, neighbors, walk):
    prior = standardize(scores)
    mass = jaccard_seed(scores, neighbors)
    total = mass.sum(dim=1, keepdim=True)
    fallback = torch.zeros_like(mass).scatter_(1, prior.argmax(dim=1, keepdim=True), 1.)
    seed = torch.where(total > 0, mass / total.clamp_min(1e-12), fallback)
    stationary = personalized_walk(*walk, seed, .85)
    return prior, prior + 1.2 * standardize((stationary * scores.size(1)).log1p())


def paired_sum(values, weights):
    left, right = values.chunk(2, dim=1)
    a, b = weights.chunk(2, dim=1)
    return (left * a.unsqueeze(-1) + right * b.unsqueeze(-1)).sum(dim=1)


def score_setting(args, folder, split, query_data, labels, state, index, anchor, metadata, record):
    ids, original, extras, mirrors = query_data
    q_labels, p_labels = labels
    source, target, errors, center, transform = state
    smooth, bank, graph, walk = index
    count, penalty, carry, shuffled = anchor
    base = anchor == ANCHORS[0]
    name = "shuffled-errors" if shuffled else f"k{count}-g{penalty:g}-r{carry:g}"
    modes = [name] + (["full", "fixed", "sharp-source", "seed16", "seed128"] if base else [])
    n, num_photos = len(ids), len(target)
    tails = {"eight": file_tensor(folder, "eight", (n, num_photos))}
    if base:
        for key in ("full", "cosine-full", "cosine-eight"):
            tails[key] = file_tensor(folder, key, (n, num_photos))
    heads = {mode: torch.from_file(str(folder / (mode + ".i32")), shared=True, size=n * 256,
                                   dtype=torch.int32).view(n, 256) for mode in modes}
    prior = target.new_tensor([.33, .22, .225, .225]).repeat(2) * .5
    ordinary, fixed, sharp = PairSetting(False, 0., 4.), PairSetting(False, 0., 0.), PairSetting(False, 4., 4.)
    for start in range(0, n, args.score_batch):
        choice = ids[start:start + args.score_batch]
        block = slice(start, start + len(choice))
        crop = F.normalize(extras[choice].to(target.device).float(), dim=-1)
        flip = F.normalize(mirrors[choice].to(target.device).float(), dim=-1)
        raw = torch.cat((crop[:, :1], original[choice, None].to(target.device), crop[:, 1:3], crop[:, 3:4], flip), dim=1)
        queries = centered(F.normalize(raw @ transform, dim=-1), center)
        support = nearest_anchors(queries.flatten(0, 1), source, count)
        mapped, components = predict_views(queries, source, target, errors, support, count=count, penalty=penalty,
                                            carry=carry, risk_components=True)
        risk = components.sum(dim=-1)
        scores = mapped @ smooth.T
        tied = paired_sum(standardize(scores), prior.expand(len(choice), -1))
        tied, tail = visual_tail(tied, bank, walk)
        tails["eight"][block] = tail.cpu()
        ranks = candidate_union(scores.unbind(1), 1280)
        votes = torch.stack([jaccard_seed(s, bank) for s in scores.unbind(1)], dim=1)
        seed = reflection_seed(votes, risk, ordinary)
        def head(key, value, strength=64.):
            selected = grow_density(strength * value, tied, *ranks, graph, 256)
            heads[key][block] = selected.to(device="cpu", dtype=torch.int32)
        head(name, seed)
        if base:
            head("fixed", reflection_seed(votes, risk, fixed))
            head("sharp-source", reflection_seed(votes, components[..., 0], sharp))
            head("seed16", seed, 16.)
            head("seed128", seed, 128.)
            full = scores[:, 1]
            full_prior, full_tail = visual_tail(full, bank, walk)
            single = grow_density(64. * votes[:, 1], full_prior, *candidate_union((full,), 1280), graph, 256)
            heads["full"][block] = single.to(device="cpu", dtype=torch.int32)
            tails["full"][block] = full_tail.cpu()
            weights = evidence_weights(torch.ones_like(risk), risk, prior, 0., 4.)
            tails["cosine-full"][block] = (mapped[:, 1] @ target.T).cpu()
            tails["cosine-eight"][block] = (paired_sum(mapped, weights) @ target.T).cpu()

    def ranking(tail, head):
        for scores, selected in zip(tail.split(256), head.split(256), strict=True):
            yield from density_rows(scores.to(target.device), selected.to(device=target.device, dtype=torch.long), len(scores), None)

    params = {"neighbors": count, "penalty": penalty, "carry": carry, "shuffled_errors": shuffled, "gallery_graph_k": 192}
    for mode in modes:
        key = "full" if mode == "full" else "eight"
        record(args, split, "visual-graph-" + mode, ranking(tails[key], heads[mode]), q_labels, p_labels, metadata,
               params | {"views": 1 if key == "full" else 8, "fusion": mode})
    if base:
        for key in ("cosine-full", "cosine-eight"):
            record(args, split, "visual-graph-" + key, iter(tails[key]), q_labels, p_labels, metadata,
                   params | {"views": 1 if key == "cosine-full" else 8, "fusion": "cosine", "gallery_graph_k": 0})


def score_visual_graph(args, split, q_ids, q_labels, p_labels, query_data, photos, contours, transform, metadata, record):
    outlines = F.normalize(F.normalize(contours @ transform, dim=-1).mean(dim=1), dim=-1)
    center = outlines.mean(dim=0)
    source = centered(outlines, center)
    target = F.normalize(photos @ transform, dim=-1)
    target = centered(target, target.mean(dim=0))
    index = image_index(photos, target, 192)
    errors = anchor_errors(source, target)
    rng = torch.Generator(device=source.device).manual_seed(731)
    permuted = errors[torch.randperm(len(source), device=source.device, generator=rng)]
    for anchor in ANCHORS:
        print(f"{split} visual anchor comparison {anchor}", flush=True)
        state = (source, target, permuted if anchor[3] else errors, center, transform)
        with TemporaryDirectory(dir=args.output) as folder:
            score_setting(args, Path(folder), split, (q_ids, *query_data), (q_labels, p_labels), state,
                          index, anchor, metadata, record)
