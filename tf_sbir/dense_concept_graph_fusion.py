"""Use open-vocabulary ClearCLIP patch evidence to index gallery photos."""

from functools import partial

import torch
from torch.nn import functional as F

from .contextual_rank import column_stochastic, jaccard_graph
from .contour_graph_index import gallery_walk, row_transition
from .contour_query_expansion import gallery_word_codes, pagerank_tail
from .global_methods import prepare_methods
from .manifold import iterative_feedback
from .reranking import standardize
from .semantic import code_vectors, top_codes
from .sketch_view_graph_fusion import separate_view_prefixes
from .union_density import density_prefixes, density_rows, pack_affinity
from .view_hypotheses import view_concepts


def patch_concept_evidence(logits, temperature=0.03):
    """Reduce patch logits to evidence that a public concept is present in the image."""
    strongest_patch = logits.amax(dim=1).float()
    log_probabilities = (logits.float() / temperature).log_softmax(dim=-1)
    regional_posterior = temperature * log_probabilities.logsumexp(dim=1)
    return strongest_patch, regional_posterior


def patch_gallery_word_codes(photos, patches, photo_words):
    dtype = torch.float16 if photos.is_cuda else torch.float32
    prompts = photo_words.to(dtype)
    variants = {name: [] for name in ("patch-max", "patch-posterior", "patch-mix0.35", "patch-mix0.65")}
    for global_vectors, local_vectors in zip(photos.split(96), patches.split(96), strict=True):
        local_vectors = local_vectors.to(device=photos.device, dtype=dtype)
        logits = local_vectors @ prompts.T
        peak, posterior = patch_concept_evidence(logits)
        global_logits = global_vectors @ photo_words.T
        regional_evidence = (peak, posterior, global_logits.lerp(posterior, 0.35), global_logits.lerp(posterior, 0.65))
        for blocks, scores in zip(variants.values(), regional_evidence, strict=True):
            ids, weights = top_codes(scores, 16)
            blocks.append(code_vectors(photo_words, ids, weights))
    return {name: torch.cat(blocks) for name, blocks in variants.items()}


def dense_concept_graph_settings(queries, photos, contours, sketch_words, photo_words, ink, patches, score_batch):
    ink = F.normalize(ink.float(), dim=-1)
    vocabulary_views = tuple(view_concepts(view[:, None], sketch_words, photo_words).squeeze(1) for view in (ink, queries))
    crop, gallery = prepare_methods(ink, photos, contours, ("local_residual",))["local_residual"]
    full, _ = prepare_methods(queries, photos, contours, ("local_residual",))["local_residual"]
    global_words = gallery_word_codes(photos, photo_words)
    if photos.is_cuda:
        global_words = global_words.half().float()
    _, *gallery_edges = jaccard_graph(gallery, 192)
    _, *rgb_edges = jaccard_graph(photos, 192)
    global_bank, *global_edges = jaccard_graph(global_words, 192)
    move, isolated = row_transition(len(photos), (gallery_edges, global_edges, rgb_edges))
    smooth = gallery_walk(move, isolated, gallery, 0.8, 4)
    gallery_bank, *smoothed_edges = jaccard_graph(smooth, 192)
    visual = tuple(iterative_feedback(view, smooth, 16, 2, 2.) @ smooth.T for view in (crop, full))
    visual_standard = tuple(standardize(scores) for scores in visual)
    early_visual = visual_standard[0].lerp(visual_standard[1], 0.5)
    late_visual = visual_standard[0].lerp(visual_standard[1], 0.35)
    del visual_standard

    def variants(name, words, word_bank, word_edges):
        banks = (gallery_bank, word_bank)
        edges = (smoothed_edges, word_edges, rgb_edges)
        graph = pack_affinity(len(photos), edges)
        transition, dangling = column_stochastic(len(photos), edges)
        lexical = tuple(view @ words.T for view in vocabulary_views)
        standardized = tuple(standardize(scores) for scores in lexical)
        for late, appearance, weight in ((False, early_visual, 0.5), (True, late_visual, 0.65)):
            terms = standardized[0].lerp(standardized[1], weight)
            prior = standardize(appearance) + standardize(terms)
            if late:
                prefix = separate_view_prefixes(visual, lexical, prior, banks, graph, 0.35, weight, True, score_batch)
            else:
                prefix = density_prefixes(appearance, terms, (appearance, terms), prior, banks, graph, 1280, score_batch, 256, 64.)
            tail = pagerank_tail(appearance, terms, prior, banks, transition, dangling, score_batch)
            mode = "late" if late else "early"
            yield f"dense-concepts-{name}-{mode}", partial(density_rows, tail, prefix, score_batch, None)

    yield from variants("global", global_words, global_bank, global_edges)
    for name, words in patch_gallery_word_codes(photos, patches, photo_words).items():
        if photos.is_cuda:
            words = words.half().float()
        word_bank, *word_edges = jaccard_graph(words, 192)
        yield from variants(name, words, word_bank, word_edges)
