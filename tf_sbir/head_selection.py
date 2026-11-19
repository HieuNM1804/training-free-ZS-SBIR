"""Select a whole ranked list using unsupervised query-performance predictors."""

from functools import partial

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from .dominant_sets import dominant_gallery, dominant_support
from .gallery_diffusion import full_gallery_inputs, regional_cosines
from .isoclip import contextual_scores, local_feedback, projected_feedback
from .mta import sketch_view_modes
from .reranking import standardize
from .union_density import density_blocks, density_prefixes, density_rows, gallery_density_graph


def rank_biased_overlap(heads, references, gallery_size, persistence):
    """Finite RBO extrapolation at equal depth, averaged over reference rankings."""
    batch, head_count, depth = heads.shape
    ranks = torch.arange(depth, dtype=torch.int32, device=heads.device)
    inverse = torch.full((*references.shape[:2], gallery_size), depth, dtype=torch.int32, device=heads.device)
    inverse.scatter_(2, references, ranks.expand_as(references))
    selected = heads.reshape(batch, 1, -1).expand(-1, references.size(1), -1)
    positions = inverse.gather(2, selected).view(batch, references.size(1), head_count, depth)
    entrance = torch.maximum(positions, ranks).long()
    overlap = torch.zeros((*entrance.shape[:-1], depth + 1), device=heads.device)
    overlap.scatter_add_(-1, entrance, torch.ones_like(entrance, dtype=torch.float32))
    agreement = overlap[..., :depth].cumsum(dim=-1) / (ranks + 1)
    weights = (1 - persistence) * persistence ** ranks.float()
    weights[-1] += persistence ** depth
    return (agreement * weights).sum(dim=-1).mean(dim=1)


def head_agreement(heads, reference):
    """Use WIG-style gallery-standardized scores and rank agreement on the same lists."""
    if reference.ndim == 2:
        reference = reference.unsqueeze(1)
    logits = standardize(reference.float())
    selected = heads.flatten(1)[:, None].expand(-1, reference.size(1), -1)
    gain = logits.gather(2, selected).unflatten(2, heads.shape[1:]).mean(dim=(1, 3))
    refs = reference.topk(heads.size(-1), dim=-1).indices
    result = [gain]
    for persistence in (1 - 1 / 64, 1 - 1 / 160):
        result.append(rank_biased_overlap(heads, refs, reference.size(-1), persistence))
    return torch.stack(result)


def predict_heads(rankings, view_queries, gallery, edge_gallery, lexical, regional, score_batch):
    batch = min(score_batch, 160)
    depth = min(256, len(gallery))
    selectors = []
    blocks = [density_blocks(initial, prefix, batch, weight) for initial, prefix, weight in rankings]
    refs = (scores.split(batch) for scores in (view_queries, lexical, regional))
    work = zip(*refs, *blocks, strict=True)
    for views, semantic, region, *scores in tqdm(work, total=(len(view_queries) + batch - 1) // batch, desc="Ranking agreement"):
        heads = torch.stack([row.topk(depth, dim=-1).indices for row in scores], dim=1)
        dtype = torch.float16 if views.is_cuda else torch.float32
        views = views.to(dtype)
        visual = views @ gallery.T
        contour = views @ edge_gallery.T
        evidence = (
            head_agreement(heads, visual), head_agreement(heads, contour),
            head_agreement(heads, semantic), head_agreement(heads, region),
        )
        selectors.append(torch.stack(evidence).mean(dim=0))
    return torch.cat(selectors, dim=1)


def selected_rows(rankings, choice, score_batch):
    batch = min(score_batch, 256)
    blocks = [density_blocks(initial, prefix, batch, weight) for initial, prefix, weight in rankings]
    for indices, *scores in zip(choice.split(batch), *blocks, strict=True):
        selection = torch.stack(scores, dim=1)
        yield from selection[torch.arange(len(indices), device=indices.device), indices].cpu()


def head_selection_settings(
    queries, photos, contours, sketch_words, photo_words, maps, sketch_views, logit_scale,
    sketch_regions, photo_regions, score_batch,
):
    dtype = torch.float16 if photos.is_cuda else torch.float32
    transform = maps[(150, 50)]
    _, mode = sketch_view_modes(queries, sketch_views, sketch_words, logit_scale)
    embedded, gallery = projected_feedback(torch.cat((queries, mode)), photos, contours, transform)
    original, visual = (vectors @ gallery.T for vectors in embedded.split(len(queries)))
    base, local, _, concepts, views = full_gallery_inputs(queries, photos, contours, sketch_words, photo_words, score_batch)
    words = views[:, 2].float()
    context = contextual_scores(photos, words, base, local, score_batch)
    del base, local, views
    banks, graph = gallery_density_graph(photos, gallery, words)
    lexical = standardize(context)
    mta_prior = standardize(visual) + lexical
    mta_prefix = density_prefixes(visual, context, (visual, context), mta_prior, banks, graph, 1280, score_batch, 256, 16.)
    yield "head-mta", partial(density_rows, mta_prior, mta_prefix, score_batch, None)

    q_regions = sketch_regions[:, 0].to(device=queries.device, dtype=dtype).contiguous()
    p_regions = photo_regions[:, 0].to(device=queries.device, dtype=dtype).contiguous()
    regional = regional_cosines(q_regions, p_regions, min(score_batch, 192))
    joint = standardize(original) + lexical + 0.25 * standardize(regional)
    sources = (original, context, regional)
    cds_prefix = density_prefixes(original, context, sources, joint, banks, graph, 1280, score_batch)
    affinity, penalty, kernel = dominant_gallery(graph, (gallery, words, photos))
    support, representative = dominant_support(joint, sources, affinity, penalty, 2., score_batch, kernel)
    cds_prior = joint + 0.25 * standardize((representative.to(dtype) @ kernel.T).float())
    del affinity, support, kernel, joint, sources, original
    yield "head-cds", partial(density_rows, cds_prior, cds_prefix, score_batch, 0.8)

    shape_query, shape_gallery = local_feedback(queries, photos, contours)
    shape = shape_query @ shape_gallery.T
    shape_prior = standardize(shape)
    visual_bank, text_bank = banks
    shape_prefix = density_prefixes(shape, shape, (shape, regional), shape_prior,
                                    (visual_bank, visual_bank), graph, 1280, score_batch, 256, 16.)
    yield "head-contour", partial(density_rows, shape_prior, shape_prefix, score_batch, None)
    word_prior = lexical + 0.25 * standardize(regional)
    word_prefix = density_prefixes(context, context, (context, regional), word_prior,
                                   (text_bank, text_bank), graph, 1280, score_batch, 256, 16.)
    yield "head-concept", partial(density_rows, word_prior, word_prefix, score_batch, None)
    del shape

    rankings = ((mta_prior, mta_prefix, None), (cds_prior, cds_prefix, 0.8),
                (shape_prior, shape_prefix, None), (word_prior, word_prefix, None))
    edge_gallery = F.normalize(F.normalize(contours @ transform, dim=-1).mean(dim=1), dim=-1)
    center = edge_gallery.mean(dim=0)
    edge_gallery = F.normalize(edge_gallery - center, dim=-1).to(dtype)
    aug = torch.cat((queries[:, None], F.normalize(sketch_views.float(), dim=-1)), dim=1)
    view_queries = F.normalize(F.normalize(aug @ transform, dim=-1) - center, dim=-1)
    semantic = F.normalize(concepts.float().mean(dim=1), dim=-1) @ words.T
    gain, overlap, broad = predict_heads(rankings, view_queries, gallery.to(dtype), edge_gallery, semantic, regional, score_batch)
    joint = 0.5 * (standardize(gain) + standardize(broad))
    for name, quality in (("wig", gain), ("rbo64", overlap), ("rbo160", broad), ("joint", joint)):
        yield f"head-qpp-{name}", partial(selected_rows, rankings, quality.argmax(dim=1), score_batch)
    yield "head-qpp-pair", partial(selected_rows, rankings, joint[:, :2].argmax(dim=1), score_batch)
    for index, name in enumerate(("mta", "cds")):
        guarded = joint.clone()
        guarded[:, index] += 0.75
        yield f"head-qpp-guard-{name}", partial(selected_rows, rankings, guarded.argmax(dim=1), score_batch)
