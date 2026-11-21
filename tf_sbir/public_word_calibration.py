"""Inverted softmax on a fixed public text dictionary for pairwise image retrieval."""

from dataclasses import dataclass, replace

import torch

from .shared_concept_likelihood import emission


DECISION = "public-word-calibration/strict-joint.json"


@dataclass(frozen=True)
class Config:
    temperature_ratio: float = 1.
    include_identity: bool = True
    side: str = "both"
    shuffled: bool = False


def canonical():
    return Config()


def candidates():
    return [Config(ratio, identity) for ratio in (.25, .5, 1.) for identity in (True, False)]


def controls(c):
    return {"primary": c, "original": canonical(), "uniform": replace(c, side="none"),
            "query_only": replace(c, side="query"), "photo_only": replace(c, side="photo"),
            "identity_control": replace(c, include_identity=not c.include_identity),
            "permuted_words": replace(c, shuffled=True)}


def cache_key(c):
    return c.temperature_ratio, c.include_identity, c.shuffled, c.side


def public_partitions(source, target, scale, include_identity):
    """Column and row log partitions of the same cross-template similarity matrix."""
    source_parts = source.new_empty(len(source))
    target_parts = source.new_full((len(target),), -torch.inf)
    for begin in range(0, len(source), 512):
        sims = (source[begin:begin+512] @ target.T) * scale
        if not include_identity:
            pos = torch.arange(len(sims), device=sims.device)
            sims[pos, begin + pos] = -torch.inf
        source_parts[begin:begin+len(sims)] = sims.logsumexp(1)
        target_parts = torch.logaddexp(target_parts, sims.logsumexp(0))
    return source_parts, target_parts


def weight(log_part, shuffle):
    if shuffle:
        order = torch.randperm(len(log_part), generator=torch.Generator().manual_seed(55127))
        log_part = log_part[order.to(log_part.device)]
    return (log_part.min() - log_part).exp()


def adjusted(rows, correction, query):
    if correction is None:
        return rows
    adjusted_rows = rows * correction
    norm = adjusted_rows.amax(1, keepdim=True) if query else adjusted_rows.sum(1, keepdim=True)
    return adjusted_rows / norm.clamp_min(torch.finfo(rows.dtype).tiny)


class Index:
    def __init__(self, data):
        self.data = data
        self.partitions = {}
        self.info = {"reference": "only fixed sketch/photo text prompts for public words",
                     "image_or_label_rows_used_to_calibrate": 0, "log_partitions": {}}

    def weights(self, c):
        if c.side == "none":
            return None, None
        key = c.temperature_ratio, c.include_identity
        if key not in self.partitions:
            d = self.data
            src, tgt = public_partitions(d.source, d.target, d.scale * c.temperature_ratio, c.include_identity)
            self.partitions[key] = src, tgt
            quantiles = torch.tensor([0., .1, .5, .9, 1.], device=src.device)
            self.info["log_partitions"][str(key)] = {
                "query_min_decile_median_upper_decile_max": torch.quantile(src, quantiles).tolist(),
                "photo_min_decile_median_upper_decile_max": torch.quantile(tgt, quantiles).tolist()}
        src, tgt = self.partitions[key]
        return (weight(src, c.shuffled) if c.side in ("query", "both") else None,
                weight(tgt, c.shuffled) if c.side in ("photo", "both") else None)

    def score(self, c):
        if c.side == "none":
            return self.data.base
        query_weight, photo_weight = self.weights(c)
        q = adjusted(self.data.query_emission, query_weight, True)
        p = adjusted(self.data.a, photo_weight, False)
        return q @ p.T

    def score_images(self, c, sketches, photos):
        """Independent entry point accepting arbitrary query and gallery batches."""
        query_weight, photo_weight = self.weights(c)
        d = self.data
        q = emission(sketches, d.source, d.scale)
        p = emission(photos, d.target, d.scale)
        p /= p.sum(1, keepdim=True)
        return adjusted(q, query_weight, True) @ adjusted(p, photo_weight, False).T
