"""Use cross-view directions of paired-domain reconstruction errors to weight retrieval evidence."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from .contextual_rank import jaccard_seed
from .contour_query_expansion import pagerank_tail
from .global_methods import centered
from .multiview_anchor_transfer import evidence_weights
from .reflection_transfer import file_tensor
from .reranking import standardize
from .transfer_reconstruction import anchor_errors, gallery_index, nearest_anchors, reconstruction_weights
from .union_density import candidate_union, density_rows, grow_density
from .view_hypotheses import view_concepts


@dataclass(frozen=True)
class CorrelationSetting:
    correlation: str = "both"
    fraction: float = 0.
    risk_power: float = 4.
    sharp: bool = False


def settings():
    out = {"independent": CorrelationSetting(), "sharp-independent": CorrelationSetting(sharp=True)}
    for sharp, tag in ((False, ""), (True, "sharp-")):
        for fraction in (.1, .25, .5, .75, .95):
            out[f"{tag}both-{fraction:g}"] = CorrelationSetting(fraction=fraction, sharp=sharp)
        for which in ("source", "target", "features", "shuffled"):
            out[f"{tag}{which}"] = CorrelationSetting(which, .5, sharp=sharp)
    for power in (1., 2.):
        out[f"power{power:g}-independent"] = CorrelationSetting(risk_power=power)
        out[f"power{power:g}-both"] = CorrelationSetting(fraction=.5, risk_power=power)
    return out


def minimum_error_weights(matrix, initial):
    """Minimize w.T H w over nonnegative weights summing to one by solving for each active set."""
    h = matrix.double()
    h = h / h.diagonal(dim1=1, dim2=2).amax(dim=1)[:, None, None].clamp_min(1e-18)
    weights = initial.double()
    weights = weights / weights.sum(dim=1, keepdim=True)
    free = torch.ones_like(weights, dtype=torch.bool)
    done = torch.zeros_like(weights[:, 0], dtype=torch.bool)
    columns = torch.arange(weights.size(1), device=weights.device)[None, :]
    for _ in range(2 ** weights.size(1)):
        restricted = h * (free[:, :, None] & free[:, None, :])
        restricted.diagonal(dim1=1, dim2=2).add_(~free)
        solution = torch.linalg.solve(restricted, free.to(h.dtype).unsqueeze(-1)).squeeze(-1)
        proposal = solution / solution.sum(dim=1, keepdim=True)
        negative = free & (proposal < -1e-11)
        move = negative.any(dim=1) & ~done
        ratios = (weights / (weights - proposal).clamp_min(1e-18)).masked_fill(~negative, torch.inf)
        step, remove = ratios.min(dim=1)
        next_weights = torch.where(move[:, None], weights.lerp(proposal, step.clamp_max(1.)[:, None]), proposal)
        next_weights = next_weights.clamp_min(0.)
        next_weights = next_weights / next_weights.sum(dim=1, keepdim=True)
        costs = (h @ next_weights.unsqueeze(-1)).squeeze(-1)
        level = (costs * next_weights).sum(dim=1)
        cheapest, add = costs.masked_fill(free, torch.inf).min(dim=1)
        release = ~move & ~done & (cheapest < level - 1e-9 * level.abs().clamp_min(1e-7))
        weights = torch.where(done[:, None], weights, next_weights)
        free &= ~(move[:, None] & columns.eq(remove[:, None]))
        free |= release[:, None] & columns.eq(add[:, None])
        done |= ~move & ~release
        if done.all():
            return weights.to(initial.dtype)
    raise RuntimeError("The simplex solve did not satisfy the active-set optimality conditions.")


def error_directions(views, source, target, errors):
    """Keep the source and target residual vectors of each view in their respective spaces."""
    dimensions = views.shape[:2]
    queries = views.flatten(0, 1)
    ids = nearest_anchors(queries, source, 64)
    mapped, local, transfer, source_risk, target_risk = [], [], [], [], []
    for q, neighbors in zip(queries.split(128), ids.split(128), strict=True):
        keys, err = source[neighbors], errors[neighbors]
        w = reconstruction_weights(q, keys, .1, err, 1.).unsqueeze(1)
        residual = q - (w @ keys).squeeze(1)
        s = (keys - q[:, None]).square().sum(dim=2).mean(dim=1).clamp_min(1e-8)
        predicted_error = (w @ err).squeeze(1)
        local.append(residual / s.sqrt()[:, None])
        transfer.append(predicted_error)
        source_risk.append(residual.square().sum(dim=1) / s)
        target_risk.append(predicted_error.square().sum(dim=1))
        mapped.append(F.normalize((w @ target[neighbors]).squeeze(1) + residual, dim=-1))
    return tuple(torch.cat(parts).unflatten(0, dimensions) for parts in (mapped, local, transfer, source_risk, target_risk))


def as_correlation(gram):
    rms = gram.diagonal(dim1=1, dim2=2).clamp_min(1e-15).sqrt()
    corr = gram / (rms[:, None] * rms[:, :, None])
    corr.diagonal(dim1=1, dim2=2).fill_(1.)
    return corr


def correlated_weights(base_weights, correlation, fraction):
    if fraction == 0.:
        return base_weights
    mixed = fraction * correlation.double()
    mixed.diagonal(dim1=1, dim2=2).add_(1. - fraction)
    inv_root = base_weights.double().clamp_min(1e-10).rsqrt()
    hessian = mixed * inv_root[:, None] * inv_root[:, :, None]
    return minimum_error_weights(hessian, base_weights)


def symmetric_sum(votes, weights):
    a, b = votes.chunk(2, dim=1)
    first, second = weights.chunk(2, dim=1)
    return (a * first[:, :, None] + b * second[:, :, None]).sum(dim=1)


def build_rankings(args, data, target_state, text_state, index, config, tail_store, prefixes):
    q_ids, full, extras, mirrored = data
    source, target, errors, center, transform = target_state
    sketch_words, photo_words = text_state
    gallery, vocabulary, banks, graph, walk = index
    device = gallery.device
    shape_prior = gallery.new_tensor([.33, .22, .225, .225])
    prior = shape_prior.repeat(2) * .5
    perm = torch.tensor([1, 2, 3, 0, 5, 6, 7, 4], device=device)
    diagnostics = {name: torch.zeros(2, device=device, dtype=torch.float64) for name in config}
    for start in range(0, len(q_ids), args.score_batch):
        ids = q_ids[start:start + args.score_batch]
        loc = slice(start, start + len(ids))
        crop = F.normalize(extras[ids].to(device).float(), dim=-1)
        flips = F.normalize(mirrored[ids].to(device).float(), dim=-1)
        raw = torch.cat((crop[:, :1], full[ids, None].to(device), crop[:, 1:3], crop[:, 3:4], flips), dim=1)
        queries = centered(F.normalize(raw @ transform, dim=-1), center)
        mapped, u, e, sr, tr = error_directions(queries, source, target, errors)
        source_gram = u.double() @ u.double().mT
        target_gram = e.double() @ e.double().mT
        correlations = {"source": as_correlation(source_gram), "target": as_correlation(target_gram),
                        "both": as_correlation(source_gram + target_gram), "features": as_correlation(raw.double() @ raw.double().mT)}
        correlations["shuffled"] = correlations["both"][:, perm][:, :, perm]
        appearance = mapped @ gallery.T
        lexical = view_concepts(raw, sketch_words, photo_words) @ vocabulary.T
        def symmetric_scores(scores):
            left, right = standardize(scores).chunk(2, dim=1)
            return ((left + right) * (.5 * shape_prior[None, :, None])).sum(dim=1)
        visual, language = symmetric_scores(appearance), symmetric_scores(lexical)
        tied = standardize(visual) + standardize(language)
        tail_store[loc] = pagerank_tail(visual, language, tied, banks, *walk, len(ids)).cpu()
        candidates = candidate_union((*appearance.unbind(1), *lexical.unbind(1)), 1280)
        shapes = torch.stack([jaccard_seed(v, banks[0]) for v in appearance.unbind(1)], dim=1)
        words = torch.stack([jaccard_seed(v, banks[1]) for v in lexical.unbind(1)], dim=1)
        joined = shapes + words
        shape_mass, joined_mass = shapes.sum(dim=-1), joined.sum(dim=-1)
        text_weights = evidence_weights(words.sum(dim=-1), sr + tr, prior, 0., 0.)
        text_seed = symmetric_sum(words, text_weights)
        for name, setting in config.items():
            if setting.sharp:
                initial = evidence_weights(joined_mass, sr, prior, 4., setting.risk_power)
            else:
                initial = evidence_weights(shape_mass, sr + tr, prior, 0., setting.risk_power)
            weights = correlated_weights(initial, correlations[setting.correlation], setting.fraction)
            seed = symmetric_sum(joined, weights) if setting.sharp else 2. * symmetric_sum(shapes, weights).lerp(text_seed, .35)
            chosen = grow_density(seed * 32., tied, *candidates, graph, 256)
            prefixes[name][loc] = chosen.to(device="cpu", dtype=torch.int32)
            diagnostics[name] += torch.stack((weights.square().sum(dim=1).reciprocal().sum(), (weights < 1e-7).sum()))
        if start % (args.score_batch * 20) == 0:
            print(f"Correlated prefixes {start + len(ids)}/{len(q_ids)}", flush=True)
    return {name: {"effective_views": float(values[0] / len(q_ids)), "weight_near_zero_fraction": float(values[1] / (len(q_ids) * 8))}
            for name, values in diagnostics.items()}


def _evaluate(args, folder, split, data, labels, state, text_state, index, metadata, record):
    q_labels, p_labels = labels
    n, g = len(q_labels), len(p_labels)
    tail = file_tensor(folder, "tail", (n, g))
    config = settings()
    prefixes = {name: torch.from_file(str(folder / (name + ".i32")), shared=True, size=n * 256, dtype=torch.int32).view(n, 256)
                for name in config}
    diagnostics = build_rankings(args, data, state, text_state, index, config, tail, prefixes)
    def scores(head):
        for values, order in zip(tail.split(256), head.split(256), strict=True):
            yield from density_rows(values.cuda(), order.to(device="cuda", dtype=torch.long), len(values), None)
    for name, setting in config.items():
        params = {"neighbors": 64, "carry": 1., "views": 8, **asdict(setting)}
        record(args, split, "correlated-" + name, scores(prefixes[name]), q_labels, p_labels, metadata, params)
    (args.output / f"{split}-view-weights.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")


def score_correlated(args, split, q_ids, q_labels, p_labels, query_data, photos, contours, sketch_words,
                     photo_words, transform, metadata, record):
    proxies = F.normalize(F.normalize(contours @ transform, dim=-1).mean(dim=1), dim=-1)
    center = proxies.mean(dim=0)
    source = centered(proxies, center)
    target = F.normalize(photos @ transform, dim=-1)
    target = centered(target, target.mean(dim=0))
    errors = anchor_errors(source, target)
    index = gallery_index(photos, target, photo_words)
    torch.cuda.empty_cache()
    with TemporaryDirectory(dir=args.output) as folder:
        _evaluate(args, Path(folder), split, (q_ids, *query_data), (q_labels, p_labels), (source, target, errors, center, transform),
                  (sketch_words, photo_words), index, metadata, record)
