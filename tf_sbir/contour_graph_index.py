"""Apply the retrieval graph to the gallery index before contour query expansion."""

from functools import partial

import torch
from torch.nn import functional as F

from .contextual_rank import column_stochastic, jaccard_graph
from .contour_query_expansion import gallery_word_codes, pagerank_tail
from .global_methods import prepare_methods
from .manifold import iterative_feedback
from .reranking import standardize
from .union_density import density_prefixes, density_rows, pack_affinity
from .view_hypotheses import view_concepts


def restarted_features(transition, dangling, features, decay, rounds):
    """Compute the expected feature around each gallery node under a restarted walk."""
    current = features
    restart = (1 - decay) * features
    for _ in range(rounds):
        current = restart + decay * torch.sparse.mm(transition, current)
        current = torch.where(dangling[:, None], features, current)
    return current


def gallery_walk(transition, dangling, features, decay, rounds):
    return F.normalize(restarted_features(transition, dangling, features, decay, rounds), dim=-1)


def row_transition(size, graphs):
    transition, dangling = column_stochastic(size, graphs)
    by_source = transition.to_sparse_coo().transpose(0, 1).coalesce().to_sparse_csr()
    return by_source, dangling


def index_rows(query, gallery, lexical, language, banks, edges, score_batch, count=16):
    return prepare_index_ranker(gallery, lexical, language, banks, edges, score_batch)(query, count=count)


def prepare_index_ranker(gallery, lexical, language, banks, edges, score_batch):
    graph = pack_affinity(len(gallery), edges)
    transition, dangling = column_stochastic(len(gallery), edges)

    def rank(query, count=16):
        expanded = iterative_feedback(query, gallery, count, 2, 2.)
        visual = expanded @ gallery.T
        prior = standardize(visual) + language
        prefix = density_prefixes(visual, lexical, (visual, lexical), prior, banks, graph, 1280, score_batch, 256, 64.)
        tail = pagerank_tail(visual, lexical, prior, banks, transition, dangling, score_batch)
        return partial(density_rows, tail, prefix, score_batch, None)

    return rank


def spectral_views(ink, photos, contours, transform):
    return (F.normalize(vectors @ transform, dim=-1) for vectors in (ink, photos, contours))


def graph_inputs(photos, gallery, ink, sketch_words, photo_words):
    words = gallery_word_codes(photos, photo_words)
    if photos.is_cuda:
        words = words.half().float()
    lexical = view_concepts(ink[:, None], sketch_words, photo_words).squeeze(1) @ words.T
    language = standardize(lexical)
    iso_bank, *iso_edges = jaccard_graph(gallery, 192)
    text_bank, *text_edges = jaccard_graph(words, 192)
    _, *rgb_edges = jaccard_graph(photos, 192)
    banks = (iso_bank, text_bank)
    graphs = (iso_edges, text_edges, rgb_edges)
    return lexical, language, banks, graphs


def contour_graph_index_settings(_queries, photos, contours, sketch_words, photo_words, maps, ink, score_batch):
    ink = F.normalize(ink.float(), dim=-1)
    sketch, target, outlines = spectral_views(ink, photos, contours, maps[(150, 50)])
    paired, gallery = prepare_methods(sketch, target, outlines, ("local_residual",))["local_residual"]
    lexical, language, banks, graphs = graph_inputs(photos, gallery, ink, sketch_words, photo_words)
    iso_edges, text_edges, rgb_edges = graphs
    _, text_bank = banks

    yield "contour-index-reference", index_rows(paired, gallery, lexical, language, banks, graphs, score_batch)
    for graph_name, sources, rebuild_decay in (("spectral", (iso_edges,), 0.9), ("fused", graphs, 0.8)):
        move, isolated = row_transition(len(gallery), sources)
        for decay in (0.5, 0.8, 0.9):
            smooth = gallery_walk(move, isolated, gallery, decay, 4)
            name = f"contour-index-{graph_name}-d{decay:g}"
            yield name, index_rows(paired, smooth, lexical, language, banks, graphs, score_batch)
            if decay == rebuild_decay:
                current_bank, *current_edges = jaccard_graph(smooth, 192)
                updated_banks = (current_bank, text_bank)
                updated_graphs = (current_edges, text_edges, rgb_edges)
                yield f"{name}-rebuild", index_rows(paired, smooth, lexical, language, updated_banks, updated_graphs, score_batch)
                if graph_name == "fused":
                    yield f"{name}-rebuild-k64", index_rows(
                        paired, smooth, lexical, language, updated_banks, updated_graphs, score_batch, count=64,
                    )
