"""Frozen vocabulary inversion with gallery-grounded concept prototypes."""

import csv
from dataclasses import dataclass
import unicodedata

import torch
from torch.nn import functional as F

from .global_methods import prepare_methods
from .manifold import add_gallery_feedback, build_graph, fused_features


@dataclass(frozen=True)
class VisualMemory:
    centroids: torch.Tensor
    support: torch.Tensor
    gallery_words: torch.Tensor
    gallery_prototypes: torch.Tensor
    term_ids: torch.Tensor
    term_weights: torch.Tensor


def read_words(csv_path):
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        terms = {
            unicodedata.normalize("NFKC", row["DisplayName"]).strip().casefold()
            for row in csv.DictReader(stream)
        }
    return sorted(word for word in terms if word)


def top_codes(scores, words_per_sample, temperature=0.03):
    values, indices = scores.topk(min(words_per_sample, scores.size(1)), dim=1)
    return indices, (values / temperature).softmax(dim=1)


def code_vectors(vectors, indices, weights):
    return F.normalize((vectors[indices] * weights.unsqueeze(-1)).sum(dim=1), dim=-1)


def gallery_memory(photos, words):
    """Ground public terms in the object appearances actually present in the index."""
    total = torch.zeros_like(words)
    weight_sum = words.new_zeros(len(words))
    assignments = []
    for batch in photos.split(256):
        ids, weights = top_codes(batch @ words.T, 16)
        partial = batch[:, None, :] * weights[:, :, None]
        total.index_add_(0, ids.flatten(), partial.flatten(0, 1))
        weight_sum.index_add_(0, ids.flatten(), weights.flatten())
        assignments.append((ids, weights))
    # A weak text prior regularizes terms with very few visual observations.
    visual_words = F.normalize(total + 0.5 * words, dim=-1)
    photo_words, photo_proto = [], []
    for ids, weights in assignments:
        photo_words.append(code_vectors(words, ids, weights))
        photo_proto.append(code_vectors(visual_words, ids, weights))
    return VisualMemory(
        visual_words, weight_sum, torch.cat(photo_words), torch.cat(photo_proto),
        torch.cat([ids for ids, _ in assignments]), torch.cat([weights for _, weights in assignments]),
    )


def guided_term_scores(sketch, photos, sketch_words, photo_words, memory, blend):
    sketch_score = sketch @ sketch_words.T
    if not blend:
        return sketch_score
    # Derive candidate photos from a preliminary sketch-to-text prediction.
    text_ids, text_weights = top_codes(sketch_score, 12)
    text_query = code_vectors(photo_words, text_ids, text_weights)
    ids, weights = top_codes(text_query @ photos.T, 16, temperature=0.05)
    neighborhood = code_vectors(photos, ids, weights)
    # Gallery-average directional similarity and prevalence are used only for
    # confidence, not as target labels or as a fixed benchmark vocabulary.
    gallery_evidence = neighborhood @ memory.centroids.T
    trust = (memory.support / (memory.support + 0.5)).sqrt()
    return sketch_score + blend * gallery_evidence * trust


def frequency_transfer(queries, photos, sketch_words, photo_words):
    """FreeDom-style control with the same prompt banks and photo-only visual memory."""
    terms = torch.cat([(batch @ photo_words.T).topk(7).indices for batch in photos.split(256)])
    projected = []
    for batch in queries.split(256):
        neighbors = (batch @ photos.T).topk(19).indices
        own = (batch @ sketch_words.T).topk(7).indices
        votes = torch.cat((terms[neighbors].flatten(1), own), dim=1)
        counts = photo_words.new_zeros((len(batch), len(photo_words)))
        counts.scatter_add_(1, votes, torch.ones_like(votes, dtype=counts.dtype))
        values, indices = counts.topk(7, dim=1)
        projected.append(code_vectors(photo_words, indices, values / values.sum(dim=1, keepdim=True)))
    return torch.cat(projected)


def semantic_settings(queries, photos, sketch_words, photo_words):
    memory = gallery_memory(photos, photo_words)
    results = {"frequency-transfer": (frequency_transfer(queries, photos, sketch_words, photo_words), photos)}
    for blend in (0.0, 0.3, 0.6):
        text_codes, visual_codes = [], []
        for batch in queries.split(256):
            scores = guided_term_scores(batch, photos, sketch_words, photo_words, memory, blend)
            ids, weights = top_codes(scores, 12)
            text_codes.append(code_vectors(photo_words, ids, weights))
            visual_codes.append(code_vectors(memory.centroids, ids, weights))
        tag = f"{blend:g}"
        q_text, q_proto = torch.cat(text_codes), torch.cat(visual_codes)
        results[f"text-to-photo-{tag}"] = (q_text, photos)
        results[f"word-memory-{tag}"] = fused_features(q_text, photos, q_text, memory.gallery_words, 0.35)
        results[f"visual-prototypes-{tag}"] = fused_features(q_text, photos, q_proto, memory.gallery_prototypes, 0.5)
    return results


def semantic_manifold_settings(queries, photos, sketch_words, photo_words):
    memory = gallery_memory(photos, photo_words)
    query_codes = []
    for batch in queries.split(256):
        logits = guided_term_scores(batch, photos, sketch_words, photo_words, memory, 0.3)
        indices, weights = top_codes(logits, 12)
        query_codes.append(code_vectors(photo_words, indices, weights))
    text_query = torch.cat(query_codes)
    gallery_words = memory.gallery_words
    yield "word-memory", *fused_features(text_query, photos, text_query, gallery_words, 0.35)
    rgb_geometry = F.normalize(photos - photos.mean(dim=0), dim=-1)
    word_geometry = F.normalize(gallery_words - gallery_words.mean(dim=0), dim=-1)
    for k, lexical_weight in ((48, 0.0), (160, 0.0), (48, 0.25), (160, 0.25)):
        name = f"k{k}-word{lexical_weight:g}"
        graph = build_graph(rgb_geometry, word_geometry, k, lexical_weight, reciprocal=lexical_weight > 0)
        photo_walk = F.normalize(graph.diffuse(photos, restart=0.1, steps=4), dim=-1)
        word_walk = F.normalize(graph.diffuse(gallery_words, restart=0.1, steps=4), dim=-1)
        yield f"word-walk-{name}", *fused_features(text_query, photo_walk, text_query, word_walk, 0.35)
        feedback = add_gallery_feedback(text_query, photo_walk, 64, 1.0)
        yield f"word-feedback-walk-{name}", *fused_features(feedback, photo_walk, text_query, word_walk, 0.35)


def agreement_settings(queries, photos, contours, sketch_words, photo_words, topology=None):
    memory = gallery_memory(photos, photo_words)
    language_batches = []
    for batch in queries.split(256):
        scores = guided_term_scores(batch, photos, sketch_words, photo_words, memory, 0.3)
        indices, weights = top_codes(scores, 12)
        language_batches.append(code_vectors(photo_words, indices, weights))
    word_query = torch.cat(language_batches)

    visual_query, rgb_geometry = prepare_methods(
        queries, photos, contours, ["local_residual"],
    )["local_residual"]
    proxies = F.normalize(contours.mean(dim=1), dim=-1)
    contour_geometry = F.normalize(proxies - proxies.mean(dim=0), dim=-1)
    word_geometry = F.normalize(memory.gallery_words - memory.gallery_words.mean(dim=0), dim=-1)
    topologies = ((160, "rgb"), (160, "lexicon"), (320, "lexicon"), (320, "rgb")) if topology is None else (topology,)
    for k, support in topologies:
        tag = f"k{k}-{support}"
        lexical_graph = support == "lexicon"
        graph = build_graph(
            rgb_geometry, word_geometry if lexical_graph else contour_geometry, k,
            shape_weight=0.2 if lexical_graph else 0.0, reciprocal=lexical_graph,
        )
        photo_walk = F.normalize(graph.diffuse(rgb_geometry, 0.1, 4), dim=-1)
        feedback = add_gallery_feedback(visual_query, photo_walk, 64, 2.0)
        rgb_smooth = F.normalize(graph.diffuse(photos, 0.1, 4), dim=-1)
        word_smooth = F.normalize(graph.diffuse(memory.gallery_words, 0.1, 4), dim=-1)
        lexical_q, lexical_p = fused_features(word_query, rgb_smooth, word_query, word_smooth, 0.35)
        yield f"visual-{tag}", feedback, photo_walk
        for weight in (0.2, 0.4, 0.6):
            yield f"fuse-{tag}-w{weight:g}", *fused_features(feedback, photo_walk, lexical_q, lexical_p, weight)
        yield f"agreement-{tag}", (feedback, photo_walk), (lexical_q, lexical_p)


def agreement_rows(visual, lexical, batch_size):
    visual_q, visual_p = visual
    lexical_q, lexical_p = lexical
    for first, second in zip(visual_q.split(batch_size), lexical_q.split(batch_size), strict=True):
        score_visual = first @ visual_p.T
        score_lexical = second @ lexical_p.T
        # Use two independent rank lists to decide when language corroborates visual retrieval.
        a, ia = score_visual.topk(64)
        b, ib = score_lexical.topk(64)
        agreement = (ia[:, :, None] == ib[:, None, :]).any(dim=2).float().mean(dim=1, keepdim=True)
        relative_strength = (b[:, :8].mean(dim=1, keepdim=True) - b[:, -24:].mean(dim=1, keepdim=True)) / (
            a[:, :8].mean(dim=1, keepdim=True) - a[:, -24:].mean(dim=1, keepdim=True)
        ).clamp_min(0.005)
        weight = (0.10 + 0.65 * agreement + 0.10 * relative_strength.clamp(0, 2)).clamp(0.1, 0.75)
        yield from torch.lerp(score_visual, score_lexical, weight).cpu()


def neighborhood_term_posteriors(image_scores, memory, neighbors=48):
    image_ids, image_weights = top_codes(image_scores, neighbors, temperature=0.05)
    terms = memory.term_ids[image_ids].flatten(1)
    votes = (memory.term_weights[image_ids] * image_weights.unsqueeze(-1)).flatten(1)
    posterior = votes.new_zeros((len(votes), len(memory.support)))
    return posterior.scatter_add_(1, terms, votes)


def refine_public_words(query, graphic_query, graphic_bank, photos, sketch_words, photo_words, memory, boost):
    primary = query @ sketch_words.T
    preliminary_ids, preliminary_weights = top_codes(primary, 12)
    lexical_query = code_vectors(photo_words, preliminary_ids, preliminary_weights)
    vis_evidence = neighborhood_term_posteriors(graphic_query @ graphic_bank.T, memory)
    lex_evidence = neighborhood_term_posteriors(lexical_query @ photos.T, memory)
    prior = memory.support / len(photos)
    cross_support = (vis_evidence * lex_evidence).sqrt()
    combined = 0.5 * vis_evidence + 0.2 * lex_evidence + 0.3 * cross_support
    calibrated = (12.0 * combined / (prior + 0.0001).sqrt()).log1p()
    indices, weights = top_codes(primary + boost * calibrated, 12)
    return code_vectors(photo_words, indices, weights)


def lexical_consensus_settings(queries, photos, contours, sketch_words, photo_words):
    memory = gallery_memory(photos, photo_words)
    visual, p = prepare_methods(queries, photos, contours, ["local_residual"])["local_residual"]
    for k in (160, 320):
        graph = build_graph(p, p, k)
        p_walk = F.normalize(graph.diffuse(p, 0.1, 4), dim=-1)
        visual = F.normalize(visual, dim=-1)
        feedback = add_gallery_feedback(visual, p_walk, 64, 2.0)
        rgb_walk = F.normalize(graph.diffuse(photos, 0.1, 4), dim=-1)
        words_walk = F.normalize(graph.diffuse(memory.gallery_words, 0.1, 4), dim=-1)
        for boost in (0.02, 0.04, 0.07):
            codes = torch.cat([
                refine_public_words(q, v, p_walk, photos, sketch_words, photo_words, memory, boost)
                for q, v in zip(queries.split(256), feedback.split(256), strict=True)
            ])
            text_q, text_p = fused_features(codes, rgb_walk, codes, words_walk, 0.35)
            for weight in (0.4, 0.65):
                name = f"verified-words-k{k}-b{boost:g}-w{weight:g}"
                yield name, *fused_features(feedback, p_walk, text_q, text_p, weight)


def contour_term_prototypes(contours, memory):
    """Ground open-vocabulary terms in synthetic contours paired to the assigned gallery RGB."""
    mean = contours.mean(dim=0)
    features = F.normalize(contours - mean, dim=-1)
    prototypes = features.new_zeros((len(memory.support), features.size(1)))
    batches = zip(features.split(256), memory.term_ids.split(256), memory.term_weights.split(256), strict=True)
    for batch, ids, weights in batches:
        votes = batch[:, None, :] * weights.unsqueeze(-1)
        prototypes.index_add_(0, ids.flatten(), votes.flatten(0, 1))
    trust = (memory.support / (memory.support + 2.0)).sqrt()
    return mean, F.normalize(prototypes, dim=-1) * trust[:, None]


def paired_concept_settings(queries, photos, contours, sketch_words, photo_words):
    memory = gallery_memory(photos, photo_words)
    contour_vectors = F.normalize(contours.mean(dim=1), dim=-1)
    mean, contour_prototypes = contour_term_prototypes(contour_vectors, memory)
    query_contours = F.normalize(queries - mean, dim=-1)
    visual, p = prepare_methods(queries, photos, contours, ["local_residual"])["local_residual"]
    graph = build_graph(p, p, 320)
    visual_bank = F.normalize(graph.diffuse(p, 0.1, 4), dim=-1)
    visual_query = add_gallery_feedback(visual, visual_bank, 64, 2.0)
    rgb_walk = F.normalize(graph.diffuse(photos, 0.1, 4), dim=-1)
    words_walk = F.normalize(graph.diffuse(memory.gallery_words, 0.1, 4), dim=-1)
    boosts = (0.0, 0.05, 0.12, 0.24, 0.36)
    all_codes = [[] for _ in boosts]
    for q, q_contour in zip(queries.split(256), query_contours.split(256), strict=True):
        primary = guided_term_scores(q, photos, sketch_words, photo_words, memory, 0.3)
        contour_evidence = q_contour @ contour_prototypes.T
        for codes, boost in zip(all_codes, boosts, strict=True):
            ids, weights = top_codes(primary + boost * contour_evidence, 12)
            codes.append(code_vectors(photo_words, ids, weights))
    for boost, codes in zip(boosts, all_codes, strict=True):
        query = torch.cat(codes)
        word_query, word_bank = fused_features(query, rgb_walk, query, words_walk, 0.35)
        for weight in (0.4, 0.6):
            name = f"paired-contour-concepts-b{boost:g}-w{weight:g}"
            yield name, *fused_features(visual_query, visual_bank, word_query, word_bank, weight)
