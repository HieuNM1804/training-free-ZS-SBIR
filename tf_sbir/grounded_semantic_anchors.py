"""Transfer a source-domain prompt neighborhood to grounded target-domain concept anchors."""

from functools import partial

import torch
from torch.nn import functional as F

from .contour_query_expansion import pagerank_tail
from .global_methods import centered
from .manifold_benchmark import dot_rows
from .reranking import standardize
from .transfer_reconstruction import gallery_index, nearest_anchors, transfer_coordinates
from .union_density import density_prefixes, density_rows


def concept_support(prompt, photos, *, seed, count):
    """Retrieve each concept's supporting photos separately in two disjoint gallery halves."""
    generator = torch.Generator(device=photos.device).manual_seed(seed)
    halves = torch.randperm(len(photos), device=photos.device, generator=generator).tensor_split(2)
    if len(halves[1]) < count:
        raise ValueError("Each gallery half needs enough photos for the requested support")
    return [half[nearest_anchors(prompt, photos[half], count)] for half in halves]


def ground_concepts(prompt, photos, gallery, supports, count):
    halves = [transfer_coordinates(prompt, photos, gallery, ids[:, :count], .1) for ids in supports]
    average = F.normalize((halves[0] + halves[1]) * .5, dim=-1)
    disagreement = (halves[0] - halves[1]) * .5
    rms = disagreement.square().sum(dim=1).mean().sqrt()
    return average, disagreement / rms.clamp_min(1e-8)


def grounded_ranking(visual, lexical, banks, graph, walk, score_batch):
    prior = standardize(visual) + standardize(lexical)
    prefix = density_prefixes(visual, lexical, (visual, lexical), prior, banks, graph, 1280, score_batch, 256, 64.)
    tail = pagerank_tail(visual, lexical, prior, banks, *walk, score_batch)
    return partial(density_rows, tail, prefix, score_batch, None)


def grounded_semantic_settings(queries, photos, sketch_words, photo_words, maps, score_batch):
    yield "ground-cosine-native", partial(dot_rows, queries, photos, score_batch)

    transform = maps[(150, 50)]
    projected = F.normalize(photos @ transform, dim=-1)
    mean = projected.mean(dim=0)
    projected = centered(projected, mean)
    native = centered(F.normalize(queries @ transform, dim=-1), mean)
    gallery, word_codes, banks, graph, walk = gallery_index(photos, projected, photo_words)
    direct = standardize(native @ gallery.T)
    query_ids = nearest_anchors(queries, sketch_words, 64)
    words = transfer_coordinates(queries, sketch_words, photo_words, query_ids[:, :24], .1)
    lexical = F.normalize(words, dim=-1) @ word_codes.T

    def rank(visual):
        return grounded_ranking(visual, lexical, banks, graph, walk, score_batch)

    yield "ground-graph-reference", rank(direct)
    supports = concept_support(photo_words, photos, seed=831, count=32)
    generator = torch.Generator(device=photos.device).manual_seed(832)
    permutation = torch.randperm(len(sketch_words), device=photos.device, generator=generator)
    for count in (8, 32):
        anchors, disagreement = ground_concepts(photo_words, photos, gallery, supports, count)

        def predict(k, penalty, *, target=anchors, error=disagreement, diagonal=False):
            mapped = transfer_coordinates(queries, sketch_words, target, query_ids[:, :k], .1,
                                          errors=error, strength=penalty, diagonal=diagonal)
            return F.normalize(mapped, dim=-1)

        for k in (8, 24, 64):
            for penalty in (0., 1.):
                transferred = standardize(predict(k, penalty) @ gallery.T)
                yield f"ground-r{count}-k{k}-g{penalty:g}", rank(direct.lerp(transferred, .6))
                del transferred
        if count == 32:
            calibrated = standardize(predict(24, 1.) @ gallery.T)
            for fraction in (.25, .85, 1.):
                yield f"ground-fraction{fraction:g}", rank(direct.lerp(calibrated, fraction))
            del calibrated
            shuffled = predict(24, 1., error=disagreement[permutation])
            yield "ground-shuffled-errors", rank(direct.lerp(standardize(shuffled @ gallery.T), .6))
            diagonal = predict(24, 1., diagonal=True)
            yield "ground-diagonal", rank(direct.lerp(standardize(diagonal @ gallery.T), .6))
            unpaired = predict(24, 0., target=anchors[permutation])
            yield "ground-unpaired", rank(direct.lerp(standardize(unpaired @ gallery.T), .6))
            yield "ground-cosine-unpaired", partial(dot_rows, unpaired, gallery, score_batch)
            yield "ground-cosine-paired", partial(dot_rows, predict(24, 0.), gallery, score_batch)
