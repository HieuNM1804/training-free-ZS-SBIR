"""Rank using a sketch's views and their uncertainty under the paired anchor map."""

from functools import partial

import torch
from torch.nn import functional as F

from .contextual_rank import jaccard_seed
from .contour_query_expansion import pagerank_tail
from .global_methods import centered
from .reranking import standardize
from .sketch_canvas_fusion import sketch_canvas_fusion_settings
from .transfer_reconstruction import anchor_errors, gallery_index, nearest_anchors, reconstruction_weights, transferred_ranking
from .union_density import candidate_union, density_rows, grow_density
from .view_hypotheses import view_concepts


def predict_views(views, source, target, errors, support, *, count, penalty, carry, risk_components=False):
    flat = views.flatten(0, 1)
    if carry and source.size(1) != target.size(1):
        raise ValueError("A source residual requires shared coordinate axes")
    mapped = target.new_empty((len(flat), target.size(1)))
    risks = source.new_empty((len(flat), 2) if risk_components else (len(flat),))
    for start in range(0, len(flat), 192):
        query = flat[start:start + 192]
        indices = support[start:start + len(query), :count]
        keys = source[indices]
        local_error = errors[indices]
        weights = reconstruction_weights(query, keys, .1, local_error, penalty).unsqueeze(1)
        residual = query - (weights @ keys).squeeze(1)
        dispersion = (keys - query[:, None]).square().sum(dim=-1).mean(dim=1)
        source_risk = residual.square().sum(dim=-1) / dispersion.clamp_min(1e-8)
        target_risk = (weights @ local_error).squeeze(1).square().sum(dim=1)
        prediction = (weights @ target[indices]).squeeze(1)
        if carry:
            prediction = prediction + carry * residual
        mapped[start:start + len(query)] = F.normalize(prediction, dim=-1)
        risks[start:start + len(query)] = torch.stack((source_risk, target_risk), dim=-1) if risk_components else source_risk + target_risk
    return mapped.unflatten(0, views.shape[:2]), risks.unflatten(0, views.shape[:2])


def evidence_weights(cohesion, risk, prior, power, risk_power):
    logits = prior.clamp_min(1e-12).log() + power * cohesion.clamp_min(1e-8).log()
    return (logits - risk_power * risk.clamp_min(1e-8).log()).softmax(dim=1)


def transfer_prefixes(visual, words, risk, target, vocabulary, banks, graph, initial, settings, score_batch):
    rankings = {name: [] for name, *_ in settings}
    prior = visual.new_tensor([.33, .22, .15, .15, .15])
    batch = min(score_batch, 192)
    parts = (value.split(batch) for value in (visual, words, risk, initial))
    for vectors, concepts, uncertainty, tied in zip(*parts, strict=True):
        appearance = (vectors @ target.T).unbind(dim=1)
        lexical = (concepts @ vocabulary.T).unbind(dim=1)
        votes = torch.stack([jaccard_seed(a, banks[0]) + jaccard_seed(b, banks[1])
                             for a, b in zip(appearance, lexical, strict=True)], dim=1)
        cohesion = votes.sum(dim=-1)
        candidates = candidate_union((*appearance, *lexical), 1280)
        for name, power, risk_power in settings:
            values = uncertainty.roll(1, dims=1) if name == "shuffled" else uncertainty
            weights = evidence_weights(cohesion, values, prior, power, risk_power)
            seed = (weights[:, None] @ votes).squeeze(1)
            rankings[name].append(grow_density(seed * 32, tied, *candidates, graph, 256))
    return {name: torch.cat(parts) for name, parts in rankings.items()}


def multiview_anchor_settings(queries, photos, contours, sketch_words, photo_words, maps, ink, canvases, score_batch):
    # One matched reference reruns the earlier canvas method on precisely these query ids.
    for name, rows in sketch_canvas_fusion_settings(queries, photos, contours, sketch_words, photo_words,
                                                    maps, ink, canvases, score_batch):
        if name in ("canvas-fusion-adaptive-all-sharp", "canvas-fusion-mix-scale"):
            yield name.replace("canvas-fusion", "multitransfer-canvas"), rows
    del rows
    transform = maps[(150, 50)]
    proxy = F.normalize(F.normalize(contours @ transform, dim=-1).mean(dim=1), dim=-1)
    center = proxy.mean(dim=0)
    source = centered(proxy, center)
    target = F.normalize(photos @ transform, dim=-1)
    target = centered(target, target.mean(dim=0))
    gallery, vocabulary, banks, graph, walk = gallery_index(photos, target, photo_words)
    raw = F.normalize(torch.cat((ink[:, None], queries[:, None], canvases), dim=1).float(), dim=-1)
    views = centered(F.normalize(raw @ transform, dim=-1), center)
    words = view_concepts(raw, sketch_words, photo_words)
    lexical = tuple(words[:, i] @ vocabulary.T for i in (0, 1))
    support = nearest_anchors(views.flatten(0, 1), source, 128)
    errors = anchor_errors(source, target)
    for tag, count, penalty, carry in (("small", 16, 1., 1.), ("carry", 16, 1., 1.5), ("wide", 64, 1., 1.),
                                        ("plain", 64, 0., 1.5), ("broad", 128, 1., 1.5)):
        mapped, risk = predict_views(views, source, target, errors, support, count=count, penalty=penalty, carry=carry)
        if tag == "small":
            baseline = torch.cat((mapped[:, 0], mapped[:, 1]))
            yield "multitransfer-two-views", transferred_ranking(baseline, gallery, lexical, banks, graph, walk, score_batch)
        appearance = standardize(mapped[:, 0] @ gallery.T).lerp(standardize(mapped[:, 1] @ gallery.T), .35)
        language = standardize(lexical[0]).lerp(standardize(lexical[1]), .65)
        initial = standardize(appearance) + standardize(language)
        tail = pagerank_tail(appearance, language, initial, banks, *walk, score_batch)
        del appearance, language
        settings = [("cohesion", 4., 0.), ("risk1", 4., 1.), ("risk4", 4., 4.)]
        if tag in ("small", "wide"):
            settings.extend([("fixed", 0., 0.), ("risk-only", 0., 2.), ("shuffled", 4., 4.)])
        for name, prefix in transfer_prefixes(mapped, words, risk, gallery, vocabulary, banks, graph,
                                              initial, settings, score_batch).items():
            yield f"multitransfer-{tag}-{name}", partial(density_rows, tail, prefix, score_batch, None)
        del initial, tail
