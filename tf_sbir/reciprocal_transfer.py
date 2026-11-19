"""Match a source query against target appearances and their leave-one-out, source-domain predictions."""

from typing import NamedTuple

import torch
from torch.nn import functional as F

from .contextual_rank import jaccard_seed
from .contour_graph_index import gallery_walk
from .contour_query_expansion import pagerank_tail
from .gallery_diffusion import nearest_gallery
from .global_methods import centered
from .multiview_anchor_transfer import predict_views
from .reranking import standardize
from .transfer_channels import ChannelSetting, channel_seed
from .transfer_reconstruction import transfer_coordinates
from .union_density import candidate_union, density_rows, grow_density


class ReverseSetting(NamedTuple):
    bank: str
    reverse_fraction: float
    power: float
    adaptive: bool = False
    intersection: float = 0.


def reciprocal_settings():
    settings = {}
    for power in (0, 4):
        settings[f"forward-p{power}"] = ReverseSetting("forward", 0., power)
        for fraction in (.2, .5, .8):
            settings[f"predicted-b{fraction:g}-p{power}"] = ReverseSetting("predicted", fraction, power)
        for bank in ("observed", "hybrid"):
            settings[f"{bank}-p{power}"] = ReverseSetting(bank, .5, power)
        settings[f"adaptive-p{power}"] = ReverseSetting("predicted", .5, power, True)
        settings[f"mutual-p{power}"] = ReverseSetting("predicted", .5, power, False, .25)
    settings["shuffled"] = ReverseSetting("shuffled", .5, 0.)
    return settings


def reverse_readout(source, target, count):
    """Reconstruct every gallery candidate's source representation from other paired ids."""
    ids, _ = nearest_gallery(target, min(count, len(target) - 1))
    return F.normalize(transfer_coordinates(target, target, source, ids, .1), dim=-1)


def reverse_fraction(forward, reverse, setting):
    prior = setting.reverse_fraction
    if not setting.adaptive:
        return prior
    original = forward.sum(dim=-1, keepdim=True)
    other = reverse.sum(dim=-1, keepdim=True)
    total = (1 - prior) * original + prior * other
    return torch.where(total > 0, prior * other / total.clamp_min(1e-8), prior)


def directional_votes(forward, backward, fraction, intersection):
    votes = forward.lerp(backward, fraction)
    if intersection:
        votes = votes.lerp(torch.minimum(forward, backward), intersection)
    return votes


def reciprocal_rows(views, mapped, risk, concepts, index, reverse, reverse_bank, setting, batch_size):
    gallery, vocabulary, banks, graph, walk = index
    priors = mapped.new_tensor([.33, .22, .15, .15, .15])
    fusion = ChannelSetting(setting.power, setting.power, 2., .35, True)
    blocks = (v.split(batch_size) for v in (views, mapped, risk, concepts))
    for source, vectors, uncertainty, words in zip(*blocks, strict=True):
        appearance = (vectors @ gallery.T).unbind(dim=1)
        lexical = (words @ vocabulary.T).unbind(dim=1)
        forward = torch.stack([jaccard_seed(a, banks[0]) for a in appearance], dim=1)
        semantic = torch.stack([jaccard_seed(a, banks[1]) for a in lexical], dim=1)
        visual_tail = standardize(appearance[0]).lerp(standardize(appearance[1]), .35)
        streams = [*appearance, *lexical]
        if setting.reverse_fraction:
            other = (source @ reverse.T).unbind(dim=1)
            streams.extend(other)
            backward = torch.stack([jaccard_seed(a, reverse_bank) for a in other], dim=1)
            fraction = reverse_fraction(forward, backward, setting)
            forward = directional_votes(forward, backward, fraction, setting.intersection)
            backward_tail = standardize(other[0]).lerp(standardize(other[1]), .35)
            tail_fraction = fraction[:, 0].lerp(fraction[:, 1], .35) if setting.adaptive else fraction
            visual_tail = visual_tail.lerp(backward_tail, tail_fraction)
        language = standardize(lexical[0]).lerp(standardize(lexical[1]), .65)
        initial = standardize(visual_tail) + standardize(language)
        seed = channel_seed(forward, semantic, uncertainty, priors, fusion)
        candidates = candidate_union(streams, 1280)
        prefix = grow_density(seed * 64., initial, *candidates, graph, 256)
        tail = pagerank_tail(visual_tail, language, initial, banks, *walk, batch_size)
        yield from density_rows(tail, prefix, batch_size, None)


def score_reverse_settings(args, split, views, concepts, source, target, errors, support, index, smoothing,
                           q_labels, p_labels, metadata, write_result):
    mapped, risk = predict_views(views, source, target, errors, support, count=64, penalty=1., carry=1.)
    predicted = reverse_readout(source, target, 64)
    generator = torch.Generator(device=source.device).manual_seed(947)
    permutation = torch.randperm(len(source), device=source.device, generator=generator)
    settings = reciprocal_settings()
    for bank in ("forward", "predicted", "observed", "hybrid", "shuffled"):
        gallery = neighbors = None
        if bank != "forward":
            features = source if bank == "observed" else predicted
            if bank == "hybrid":
                features = F.normalize(predicted + source, dim=-1)
            elif bank == "shuffled":
                features = predicted[permutation]
            gallery = gallery_walk(*smoothing, features, .8, 4)
            del features
            ids, _ = nearest_gallery(centered(gallery, gallery.mean(dim=0)), 192)
            neighbors = ids.sort(dim=1).values
            del ids
        for name, config in settings.items():
            if config.bank == bank:
                scores = reciprocal_rows(views, mapped, risk, concepts, index, gallery, neighbors, config, args.score_batch)
                params = {"neighbors": 64, "penalty": 1., "carry": 1., "views": 5, **config._asdict()}
                write_result(args, split, f"reciprocal-{name}", scores, q_labels, p_labels, metadata, params)
