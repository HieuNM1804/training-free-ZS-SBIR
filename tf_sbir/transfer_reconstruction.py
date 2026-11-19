"""Transfer affine coordinates between paired anchor banks using held-out target errors."""

from functools import partial

import torch
from torch.nn import functional as F

from .contextual_rank import column_stochastic, jaccard_graph
from .contour_graph_index import gallery_walk, row_transition
from .contour_query_expansion import gallery_word_codes, pagerank_tail
from .gallery_diffusion import nearest_gallery
from .global_methods import centered
from .manifold_benchmark import dot_rows
from .reranking import standardize
from .sketch_view_graph_fusion import separate_view_prefixes
from .union_density import density_rows, pack_affinity
from .view_hypotheses import view_concepts


def nearest_anchors(queries, anchors, count):
    return torch.cat([(block @ anchors.T).topk(min(count, len(anchors)), dim=1).indices for block in queries.split(512)])


def reconstruction_weights(query, neighbors, ridge, error=None, strength=0., *, diagonal=False):
    """Solve a regularized local source reconstruction subject to sum(weights) = 1."""
    offsets = neighbors - query[:, None]
    gram = offsets @ offsets.mT
    scale = gram.diagonal(dim1=-2, dim2=-1).mean(dim=1).clamp_min(1e-5)
    if error is not None and strength:
        if diagonal:
            gram.diagonal(dim1=-2, dim2=-1).add_(strength * scale[:, None] * error.square().sum(dim=-1))
        else:
            gram = gram + (strength * scale[:, None, None]) * (error @ error.mT)
    gram.diagonal(dim1=-2, dim2=-1).add_((ridge * scale).clamp_min(1e-6)[:, None])
    ones = torch.ones_like(gram[:, :, :1])
    coeffs = torch.linalg.solve_ex(gram, ones).result.squeeze(-1)
    return coeffs / coeffs.sum(dim=1, keepdim=True).clamp_min(1e-12)


def transfer_coordinates(queries, source, target, support, ridge, *, errors=None, strength=0., residual=0., diagonal=False):
    """Read out target anchors. Residual carryover is optional for shared coordinate axes."""
    if len(source) != len(target):
        raise ValueError("Source and target anchor ids must correspond")
    if residual and source.size(1) != target.size(1):
        raise ValueError("A source residual requires shared feature axes")
    translated = []
    for q, ids in zip(queries.split(256), support.split(256), strict=True):
        local = source[ids]
        error = errors[ids] if errors is not None and strength else None
        w = reconstruction_weights(q, local, ridge, error, strength, diagonal=diagonal).unsqueeze(1)
        prediction = (w @ target[ids]).squeeze(1)
        if residual:
            prediction = prediction + residual * (q - (w @ local).squeeze(1))
        translated.append(prediction)
    return torch.cat(translated)


def anchor_errors(source, target, count=16, ridge=0.1):
    """Predict each target from other pairs. Scale errors once in the target bank."""
    if len(source) < 2:
        raise ValueError("At least two pairs are needed to hold one pair out")
    ids, _ = nearest_gallery(source, min(count, len(source) - 1))
    held_out = transfer_coordinates(source, source, target, ids, ridge)
    errors = target - held_out
    return errors / errors.square().sum(dim=1).mean().sqrt().clamp_min(1e-8)


def gallery_index(photos, target, photo_words, *, return_smoothing=False):
    vocabulary = gallery_word_codes(photos, photo_words)
    if photos.is_cuda:
        vocabulary = vocabulary.half().float()
    _, *visual_edges = jaccard_graph(target, 192)
    word_bank, *word_edges = jaccard_graph(vocabulary, 192)
    _, *rgb_edges = jaccard_graph(photos, 192)
    transition, isolated = row_transition(len(photos), (visual_edges, word_edges, rgb_edges))
    smooth = gallery_walk(transition, isolated, target, 0.8, 4)
    visual_bank, *smooth_edges = jaccard_graph(smooth, 192)
    edges = (smooth_edges, word_edges, rgb_edges)
    index = smooth, vocabulary, (visual_bank, word_bank), pack_affinity(len(photos), edges), column_stochastic(len(photos), edges)
    return (index, (transition, isolated)) if return_smoothing else index


def transferred_ranking(mapped, gallery, lexical, banks, graph, walk, score_batch):
    visual = tuple(block @ gallery.T for block in mapped.chunk(2))
    appearance = standardize(visual[0]).lerp(standardize(visual[1]), 0.35)
    words = standardize(lexical[0]).lerp(standardize(lexical[1]), 0.65)
    prior = standardize(appearance) + standardize(words)
    prefix = separate_view_prefixes(visual, lexical, prior, banks, graph, 0.35, 0.65, True, score_batch)
    tail = pagerank_tail(appearance, words, prior, banks, *walk, score_batch)
    return partial(density_rows, tail, prefix, score_batch, None)


def residual_moments(target, prediction, direction):
    residual = target - prediction
    return torch.stack(((residual * direction).sum(dim=-1), direction.square().sum(dim=-1)), dim=-1)


def local_gain(moments, neighbors, *, maximum=2.):
    """Minimize pooled target squared error on the supporting leave-one-out pairs."""
    cross, energy = moments[neighbors].mean(dim=1).unbind(dim=-1)
    return (cross / energy.clamp_min(1e-12)).clamp(0., maximum)[:, None]


def calibrated_transfer_settings(query, source, target, errors, support, rank):
    anchor_support, _ = nearest_gallery(source, min(128, len(source) - 1))
    perm = torch.randperm(len(source), device=source.device, generator=torch.Generator(device=source.device).manual_seed(811))
    joint = torch.cat((target, source), dim=1)
    for count in (16, 64, 128):
        neighbors = support[:, :count]
        for strength in (0., 1.):
            predicted, fit = transfer_coordinates(query, source, joint, neighbors, .1, errors=errors,
                                                  strength=strength).split((target.size(1), source.size(1)), dim=1)
            direction = query - fit

            def ranking(gain):
                return rank(F.normalize(predicted + gain * direction, dim=-1))

            gains = (1., 1.5, 2.) if count != 16 else (1., 1.5)
            for gain in gains:
                yield f"transfer-sweep-k{count}-g{strength:g}-r{gain:g}", ranking(gain)
            if count == 64 and strength == 0.:
                # LOO predictions use other anchors' target values and source coordinates only.
                anchor_prediction, anchor_fit = transfer_coordinates(
                    source, source, joint, anchor_support[:, :count], .1,
                ).split((target.size(1), source.size(1)), dim=1)
                moments = residual_moments(target, anchor_prediction, source - anchor_fit)
                for width in (16, 64):
                    gain = local_gain(moments, support[:, :width])
                    yield f"transfer-sweep-local{width}", ranking(gain)
                global_stats = moments.mean(dim=0)
                global_gain = (global_stats[0] / global_stats[1].clamp_min(1e-12)).clamp(0., 2.)
                yield "transfer-sweep-global", ranking(global_gain)
                yield "transfer-sweep-shuffled", ranking(local_gain(moments[perm], neighbors))


def image_transfer_variants(original, source, target, support, errors, perm, score_batch):
    yield "transfer-image-identity", partial(dot_rows, original, target, score_batch)
    for suffix, penalty, target_bank in (("plain", 0., target), ("calibrated", 1., target), ("unpaired", 0., target[perm])):
        predicted = transfer_coordinates(original, source, target_bank, support[:, :16], .1, errors=errors, strength=penalty)
        yield f"transfer-image-{suffix}", partial(dot_rows, F.normalize(predicted, dim=-1), target, score_batch)


def transfer_reconstruction_settings(queries, photos, contours, sketch_words, photo_words, maps, ink, score_batch, *, sweep=False):
    ink = F.normalize(ink.float(), dim=-1)
    views = torch.cat((ink, queries))
    transform = maps[(150, 50)]
    proxies = F.normalize(F.normalize(contours @ transform, dim=-1).mean(dim=1), dim=-1)
    source_center = proxies.mean(dim=0)
    source = centered(proxies, source_center)
    target = F.normalize(photos @ transform, dim=-1)
    target = centered(target, target.mean(dim=0))
    query = centered(F.normalize(views @ transform, dim=-1), source_center)
    support = nearest_anchors(query, source, 128 if sweep else 64)
    errors = anchor_errors(source, target)
    generator = torch.Generator(device=source.device).manual_seed(731)
    perm = torch.randperm(len(source), device=source.device, generator=generator)

    # The single-view cosine controls use only the index of unlabeled image pairs.
    if not sweep:
        yield from image_transfer_variants(query[len(queries):], source, target, support[len(queries):],
                                           errors, perm, score_batch)

    smooth, vocabulary, banks, graph, walk = gallery_index(photos, target, photo_words)
    text_vectors = view_concepts(views[:, None], sketch_words, photo_words).squeeze(1)
    reference_lexical = tuple(block @ vocabulary.T for block in text_vectors.chunk(2))

    def rank(mapped, lexical=reference_lexical):
        return transferred_ranking(mapped, smooth, lexical, banks, graph, walk, score_batch)

    if sweep:
        yield from calibrated_transfer_settings(query, source, target, errors, support, rank)
        return

    def translate(count, ridge, strength, residual, *, error_bank=errors, diagonal=False):
        mapped = transfer_coordinates(query, source, target, support[:, :count], ridge,
                                      errors=error_bank, strength=strength, residual=residual, diagonal=diagonal)
        return F.normalize(mapped, dim=-1)

    yield "transfer-graph-identity", rank(query)
    for count in (8, 16, 32, 64):
        for strength in (0., 1.):
            yield f"transfer-graph-k{count}-g{strength:g}", rank(translate(count, .1, strength, 0.5))

    for ridge in (.01, .5):
        yield f"transfer-graph-ridge{ridge:g}", rank(translate(16, ridge, 1., .5))
    for strength in (.1, 4.):
        yield f"transfer-graph-strength{strength:g}", rank(translate(16, .1, strength, .5))
    for residual in (0., 1.):
        yield f"transfer-graph-residual{residual:g}", rank(translate(16, .1, 1., residual))
    yield "transfer-graph-diagonal", rank(translate(16, .1, 1., .5, diagonal=True))
    yield "transfer-graph-shuffled-errors", rank(translate(16, .1, 1., .5, error_bank=errors[perm]))

    # A second pair bank maps raw gallery features to their graph-smoothed features.
    target_errors = anchor_errors(target, smooth)
    for strength in (0., 1.):
        first = translate(16, .1, strength, .5)
        second_support = nearest_anchors(first, target, 16)
        second = transfer_coordinates(first, target, smooth, second_support, .1, errors=target_errors,
                                      strength=strength, residual=.25)
        yield f"transfer-graph-two-stage-g{strength:g}", rank(F.normalize(second, dim=-1))

    # The same operator also supports template pairs of public concepts.
    # Both prompt banks use the same ordered words, with one row per concept.
    prompt_errors = anchor_errors(sketch_words, photo_words)
    prompt_support = nearest_anchors(views, sketch_words, 24)
    visual_query = translate(16, .1, 1., .5)
    for count in (8, 24):
        for strength in (0., 1.):
            text = transfer_coordinates(views, sketch_words, photo_words, prompt_support[:, :count], .1,
                                        errors=prompt_errors, strength=strength)
            lexical = tuple(F.normalize(block, dim=-1) @ vocabulary.T for block in text.chunk(2))
            yield f"transfer-prompt-k{count}-g{strength:g}", rank(visual_query, lexical)
