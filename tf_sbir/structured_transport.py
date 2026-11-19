"""Rerank shortlist candidates with bijective fused Gromov–Wasserstein scores."""

from functools import partial
from itertools import permutations

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from .gallery_diffusion import full_gallery_inputs, regional_cosines
from .isoclip import contextual_scores, projected_feedback
from .mta import sketch_view_modes
from .reranking import standardize
from .union_density import candidate_union, density_prefixes, density_rows, gallery_density_graph


STRUCTURE_WEIGHTS = (0., 0.15, 0.3)


def part_distances(centers):
    distances = torch.cdist(centers.float(), centers.float())
    spread = distances.square().mean(dim=(-2, -1), keepdim=True).sqrt()
    return distances / spread.clamp_min(0.04)


def matching_terms(similarity, sketch_distances, photo_distances, assignments):
    """Score all uniform bijections using semantic affinity and squared GW distortion."""
    sites = sketch_distances.size(-1)
    routes = F.one_hot(assignments, num_classes=sites).flatten(1).to(similarity.dtype) / sites
    inter = (similarity.flatten(-2) @ routes.T).float()
    inverse = assignments.argsort(dim=-1)
    reordered = sketch_distances[:, inverse[:, :, None], inverse[:, None, :]].flatten(-2)
    cross = (photo_distances.flatten(-2) @ reordered.transpose(1, 2)).float() / sites ** 2
    source = sketch_distances.float().square().mean(dim=(-2, -1))[:, None, None]
    target = photo_distances.float().square().mean(dim=(-2, -1)).unsqueeze(-1)
    return inter, (source + target - 2 * cross).clamp_min_(0.)


def masked_standardize(scores, visible):
    count = visible.sum(dim=1, keepdim=True)
    mean = scores.masked_fill(~visible, 0.).sum(dim=1, keepdim=True) / count
    centered = (scores - mean).masked_fill_(~visible, 0.)
    spread = (centered.square().sum(dim=1, keepdim=True) / count).sqrt().clamp_min(1e-6)
    return centered / spread


def structured_cues(sketch, photos, sketch_centers, photo_centers, streams, batch_size):
    assignments = torch.tensor(list(permutations(range(sketch.size(1)))), device=photos.device)
    sketch_distance, photo_distance = (part_distances(points).to(photos.dtype) for points in (sketch_centers, photo_centers))
    batch = min(batch_size, 256)
    parts = (tensor.split(batch) for tensor in (sketch, sketch_distance, *streams))
    records = []
    batches = zip(*parts, strict=True)
    for sketches, distances, *streams_part in tqdm(batches, total=(len(sketch) + batch - 1) // batch, desc="FGW candidate pairs"):
        indices, valid = candidate_union(streams_part, min(512, len(photos)))
        terms = []
        for subset in indices.split(128, dim=1):
            similarity = sketches[:, None] @ photos[subset].transpose(-2, -1)
            inter, intra = matching_terms(similarity, distances, photo_distance[subset], assignments)
            terms.append(torch.stack([
                ((1 - alpha) * inter - alpha * intra).amax(dim=-1) for alpha in STRUCTURE_WEIGHTS
            ]))
        values = torch.cat(terms, dim=-1)
        cues = [masked_standardize(score, valid) for score in values]
        records.append((indices.to(torch.int32), torch.stack(cues)))
    return records


def structured_bonus(reference, records, index):
    batch = records[0][0].size(0)
    bonus = torch.zeros_like(reference)
    for block, (candidates, scores) in zip(bonus.split(batch), records, strict=True):
        block.scatter_add_(1, candidates.long(), scores[index])
    return bonus


def structured_transport_settings(
    queries, photos, contours, sketch_words, photo_words, maps, sketch_views, logit_scale,
    sketch_patches, photo_patches, sketch_centers, photo_centers, score_batch,
):
    _, robust = sketch_view_modes(queries, sketch_views, sketch_words, logit_scale)
    encoded, gallery = projected_feedback(torch.cat((queries, robust)), photos, contours, maps[(150, 50)])
    first, augmented = encoded.split(len(queries))
    original = first @ gallery.T
    mta = augmented @ gallery.T
    base, regional, _, _, views = full_gallery_inputs(queries, photos, contours, sketch_words, photo_words, score_batch)
    words = views[:, 2].float()
    context = contextual_scores(photos, words, base, regional, score_batch)
    del base, regional, views
    lexical = standardize(context)
    banks, graph = gallery_density_graph(photos, gallery, words)
    mta_prior = standardize(mta) + lexical
    prefix = density_prefixes(mta, context, (mta, context), mta_prior, banks, graph, 1280, score_batch, 256, 16.)
    yield "fgw-mta-graph-control", partial(density_rows, mta_prior, prefix, score_batch, None)

    dtype = torch.float16 if queries.is_cuda else torch.float32
    sketch = sketch_patches[:, 0].to(device=queries.device, dtype=dtype).contiguous()
    photo = photo_patches[:, 0].to(device=queries.device, dtype=dtype).contiguous()
    region = regional_cosines(sketch, photo, min(score_batch, 192))
    original_prior = standardize(original) + lexical + 0.25 * standardize(region)
    prefix = density_prefixes(original, context, (original, context, region), original_prior, banks, graph, 1280, score_batch)
    yield "fgw-clearclip-graph-control", partial(density_rows, original_prior, prefix, score_batch, 0.8)

    scores = structured_cues(sketch, photo, sketch_centers.to(queries.device), photo_centers.to(queries.device),
                            (mta, original, context, region), score_batch)
    visual = standardize(mta)
    for index, alpha in enumerate(STRUCTURE_WEIGHTS):
        added = structured_bonus(mta, scores, index)
        for weight in (0.25, 0.75):
            refined = visual + weight * added
            prior = mta_prior + weight * added
            mta_prefix = density_prefixes(refined, context, (refined, context, region), prior, banks, graph,
                                          1280, score_batch, 256, 16.)
            yield f"fgw-mta-a{alpha:g}-w{weight:g}", partial(density_rows, prior, mta_prefix, score_batch, None)
        prior = original_prior + 0.25 * added
        yield f"fgw-clearclip-a{alpha:g}-w0.25", partial(density_rows, prior, prefix, score_batch, 0.8)
