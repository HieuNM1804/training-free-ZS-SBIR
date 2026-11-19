"""Attach the uncertainty of the visual transfer to its own evidence in a multiview ranker."""

from typing import NamedTuple

import torch

from .contextual_rank import jaccard_seed
from .multiview_anchor_transfer import evidence_weights
from .union_density import candidate_union, grow_density


class ChannelSetting(NamedTuple):
    appearance_power: float
    lexical_power: float
    risk_power: float
    lexical_fraction: float
    separate: bool
    shuffle_views: bool = False


def channel_settings():
    settings = {}
    for p in (0, 2, 4, 8):
        for language in (.35, .5, .65):
            settings[f"channel-p{p}-l{language:g}"] = ChannelSetting(p, p, 2., language, True)
        settings[f"joint-p{p}"] = ChannelSetting(p, p, 2., .5, False)
    settings.update({
        "channel-no-risk": ChannelSetting(4., 4., 0., .5, True),
        "channel-strong-risk": ChannelSetting(4., 4., 4., .5, True),
        "channel-shuffled": ChannelSetting(4., 4., 2., .5, True, True),
        "channel-visual-sharp": ChannelSetting(4., 0., 2., .5, True),
        "channel-lexical-sharp": ChannelSetting(0., 4., 2., .5, True),
    })
    return settings


def channel_seed(appearance, lexical, risk, prior, config):
    fraction = config.lexical_fraction
    if config.shuffle_views:
        risk = risk.roll(1, dims=1)
    if config.separate:
        a = evidence_weights(appearance.sum(dim=-1), risk, prior, config.appearance_power, config.risk_power)
        b = evidence_weights(lexical.sum(dim=-1), risk, prior, config.lexical_power, 0.)
        return ((a[:, None] @ appearance) * (1 - fraction) + (b[:, None] @ lexical) * fraction).squeeze(1)
    joined = appearance.lerp(lexical, fraction)
    weights = evidence_weights(joined.sum(dim=-1), risk, prior, config.appearance_power, config.risk_power)
    return (weights[:, None] @ joined).squeeze(1)


def channel_prefix(vectors, concepts, risk, target, vocabulary, banks, graph, tied, config):
    appearance = (vectors @ target.T).unbind(dim=1)
    lexical = (concepts @ vocabulary.T).unbind(dim=1)
    candidates = candidate_union((*appearance, *lexical), 1280)
    first = torch.stack([jaccard_seed(scores, banks[0]) for scores in appearance], dim=1)
    second = torch.stack([jaccard_seed(scores, banks[1]) for scores in lexical], dim=1)
    prior = vectors.new_tensor([.33, .22, .15, .15, .15])
    seed = channel_seed(first, second, risk, prior, config)
    return grow_density(seed * 64, tied, *candidates, graph, 256)
