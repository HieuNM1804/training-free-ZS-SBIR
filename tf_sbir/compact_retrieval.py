"""Whole-sketch pooling, conditional transport, and query-specific graph retrieval."""

from functools import partial

import torch
from torch.nn import functional as F

from .contextual_rank import column_stochastic, contextual_blocks, jaccard_graph
from .gallery_diffusion import regional_cosines
from .global_methods import centered
from .reranking import standardize
from .semantic import top_codes
from .union_density import density_prefixes, density_rows, pack_affinity
from .view_hypotheses import view_concepts


def paired_graph(left, right, neighbors):
    left_bank, *left_edges = jaccard_graph(left, neighbors)
    right_bank, *right_edges = jaccard_graph(right, neighbors)
    return (left_bank, right_bank), pack_affinity(len(left), (left_edges, right_edges))


def conditional_words(query, sketch_words, photo_words, targets, score_batch):
    dtype = torch.float16 if query.is_cuda else torch.float32
    prompts = sketch_words.to(dtype)
    indices = [top_codes((batch.to(dtype) @ prompts.T).float(), 5)[0] for batch in query.split(score_batch)]
    return regional_cosines(photo_words[torch.cat(indices)].to(dtype), targets, min(score_batch, 192))


def gallery_concept_views(photos, edges, photo_words):
    words = view_concepts(photos[:, None], photo_words, photo_words).squeeze(1)
    targets = torch.stack((photos, edges, words), dim=1)
    return words, targets.half() if photos.is_cuda else targets


def compact_retrieval_settings(queries, photos, contours, sketch_words, photo_words, maps, ink, score_batch):
    ink = F.normalize(ink.float(), dim=-1)
    pooled = F.normalize(queries + ink, dim=-1)
    edges = F.normalize(contours.float().mean(dim=1), dim=-1)

    rgb, shape = pooled @ photos.T, pooled @ edges.T
    image = standardize(rgb) + standardize(shape)
    yield "compact-image-control", partial(density_rows, image, None, score_batch, None)
    image_banks, image_graph = paired_graph(photos, edges, 192)
    prefix = density_prefixes(rgb, shape, (rgb, shape), image, image_banks, image_graph, 1280, score_batch, 256, 16.)
    yield "compact-image-graph", partial(density_rows, image, prefix, score_batch, None)
    del rgb, shape, prefix, image_banks, image_graph

    image_space = F.normalize(torch.cat((photos, edges), dim=1), dim=-1)
    transform = maps[(150, 50)]
    selected_query, selected_photos, selected_edges = (F.normalize(view @ transform, dim=-1) for view in (pooled, photos, edges))
    iso_space = F.normalize(torch.cat((selected_photos, selected_edges), dim=1), dim=-1)
    iso_scores = standardize(selected_query @ selected_photos.T) + standardize(selected_query @ selected_edges.T)
    iso_prior = standardize(image) + standardize(iso_scores)
    iso_banks, iso_graph = paired_graph(image_space, iso_space, 192)
    for anchor in (4., 16.):
        prefix = density_prefixes(image, iso_scores, (image, iso_scores), iso_prior, iso_banks, iso_graph,
                                  1280, score_batch, 256, anchor)
        yield f"compact-image-isoclip-a{anchor:g}", partial(density_rows, iso_prior, prefix, score_batch, None)
    del iso_space, iso_scores, iso_banks, iso_graph, iso_prior, prefix

    # Project Open Images terms into the photo prompt space, then match both gallery views.
    gallery_text, targets = gallery_concept_views(photos, edges, photo_words)
    banks, graph = paired_graph(image_space, gallery_text, 192)
    for name, view in (("original", queries), ("ink", ink), ("pooled", pooled)):
        visual = image if name == "pooled" else standardize(view @ photos.T) + standardize(view @ edges.T)
        lexical = conditional_words(view, sketch_words, photo_words, targets, score_batch)
        prior = standardize(visual) + standardize(lexical)
        yield f"compact-{name}-transport", partial(density_rows, prior, None, score_batch, None)
        for anchor in ((4., 16., 64.) if name == "pooled" else (16.,)):
            prefix = density_prefixes(visual, lexical, (visual, lexical), prior, banks, graph, 1280,
                                      score_batch, 384, anchor)
            yield f"compact-{name}-a{anchor:g}-k256", partial(density_rows, prior, prefix[:, :256], score_batch, None)
            if name == "pooled":
                yield f"compact-{name}-a{anchor:g}-k384", partial(density_rows, prior, prefix, score_batch, None)


def compact_calibrated_settings(queries, photos, contours, sketch_words, photo_words, maps, ink, score_batch):
    del queries
    ink = F.normalize(ink.float(), dim=-1)
    edges = F.normalize(contours.float().mean(dim=1), dim=-1)
    transform = maps[(150, 50)]
    sketch, gallery, shape = (F.normalize(features @ transform, dim=-1) for features in (ink, photos, edges))
    sketch = centered(sketch, shape.mean(dim=0))
    gallery = centered(gallery, gallery.mean(dim=0))
    shape = centered(shape, shape.mean(dim=0))
    visual = standardize(sketch @ gallery.T) + standardize(sketch @ shape.T)

    gallery_text, targets = gallery_concept_views(photos, edges, photo_words)
    semantic = conditional_words(ink, sketch_words, photo_words, targets, score_batch)
    prior = standardize(visual) + standardize(semantic)
    yield "compact-calibrated-control", partial(density_rows, prior, None, score_batch, None)

    appearance = F.normalize(torch.cat((gallery, shape), dim=1), dim=-1)
    visual_bank, *visual_edges = jaccard_graph(appearance, 192)
    text_bank, *text_edges = jaccard_graph(gallery_text, 192)
    banks = (visual_bank, text_bank)
    graphs = (visual_edges, text_edges)
    graph = pack_affinity(len(photos), graphs)
    transition, dangling = column_stochastic(len(photos), graphs)
    diffusion = torch.cat(list(contextual_blocks(visual, semantic, prior, *banks, transition, dangling, min(256, score_batch))))
    walk_prior = prior + 1.2 * diffusion
    yield "compact-calibrated-pagerank", partial(density_rows, walk_prior, None, score_batch, None)

    for anchor in (4., 16., 64.):
        prefix = density_prefixes(visual, semantic, (visual, semantic), prior, banks, graph,
                                  1280, score_batch, 256, anchor)
        yield f"compact-calibrated-a{anchor:g}", partial(density_rows, prior, prefix, score_batch, None)
    prefix = density_prefixes(visual, semantic, (visual, semantic, diffusion), walk_prior, banks, graph,
                              1280, score_batch, 256, 16.)
    yield "compact-calibrated-pagerank-a16", partial(density_rows, walk_prior, prefix, score_batch, None)
