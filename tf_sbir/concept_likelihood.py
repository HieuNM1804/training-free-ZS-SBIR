"""Single-sketch retrieval through paired public concepts and an unlabeled photo gallery."""

from dataclasses import dataclass, replace
import math

import torch
import torch.nn.functional as F

from .gallery_diffusion import mutual_transition, solve_diffusion
from .reranking import standardize


@dataclass(frozen=True)
class Config:
    target_temp: float = .02
    query_temp: float = .02
    terms: int = 16
    prior_power: float = .5
    direct: float = .35
    walk: float = .6
    neighbors: int = 96
    seeds: int = 128
    pairing: str = "paired"
    channel: str = "mixture"

    @property
    def name(self):
        return (f"{self.channel}-tp{self.target_temp:g}-tq{self.query_temp:g}-c{self.terms}"
                f"-b{self.prior_power:g}-v{self.direct:g}-w{self.walk:g}-k{self.neighbors}"
                f"-s{self.seeds}-{self.pairing}")


def pilot_configs():
    base = Config()
    configs = {base}
    for temp in (.01, .02, .04):
        for power in (0., .5, 1.):
            for walk in (0., .6):
                configs.add(replace(base, target_temp=temp, prior_power=power, walk=walk))
    for field, values in {
        "query_temp": (.01, .04), "terms": (4, 8, 32, 64), "direct": (0., .7, 1.),
        "walk": (.3, .85, .95), "neighbors": (32, 192), "seeds": (48, 320),
    }.items():
        configs.update(replace(base, **{field: v}) for v in values)
    configs.update(replace(base, channel=name, walk=0.) for name in ("forward", "reverse", "wordcos", "direct"))
    configs.add(replace(base, walk=0., direct=0., pairing="shuffled"))
    return sorted(configs, key=lambda c: c.name)


class ConceptIndex:
    """All learned quantities come from frozen CLIP. All index statistics are photo-only."""

    def __init__(self, photos, source_words, target_words, temperature):
        self.photos, self.source, self.target = photos, source_words, target_words
        self.shuffle = torch.randperm(len(target_words), generator=torch.Generator().manual_seed(2179)).to(photos.device)
        self.likelihood = torch.empty((len(target_words), len(photos)), device=photos.device, dtype=torch.float16)
        mass = photos.new_zeros(len(target_words))
        inverse_shuffle = self.shuffle.argsort()
        source_codes, target_codes, shuffled_codes = [], [], []
        for start in range(0, len(photos), 384):
            block = photos[start:start + 384]
            logits = block @ target_words.T
            values, terms = logits.topk(16)
            weights = (values / .03).softmax(1).unsqueeze(-1)
            source_codes.append(F.normalize((weights * source_words[terms]).sum(1)))
            target_codes.append(F.normalize((weights * target_words[terms]).sum(1)))
            shuffled_codes.append(F.normalize((weights * source_words[inverse_shuffle[terms]]).sum(1)))
            # Store posteriors relative to a uniform vocabulary for half-precision range.
            relative = (logits / temperature).softmax(1) * len(target_words)
            mass += relative.sum(0)
            self.likelihood[:, start:start + len(block)] = relative.T.half()
        self.frequency = (mass / len(photos)).clamp_min(1e-9)
        self.source_codes = torch.cat(source_codes)
        self.target_codes = torch.cat(target_codes)
        self.shuffled_codes = torch.cat(shuffled_codes)

    def query_codes(self, query, c):
        bank = self.target if c.pairing == "neutral" else self.source
        values, ids = (query @ bank.T).topk(c.terms)
        if c.pairing == "shuffled":
            ids = self.shuffle[ids]
        return ids, (values / c.query_temp).softmax(1)

    def direct_scores(self, query, ids, weights, c):
        query_photo = F.normalize((weights[:, :, None] * self.target[ids]).sum(1))
        forward = query_photo @ self.photos.T
        if c.channel == "forward":
            return forward
        gallery_source = self.target_codes if c.pairing == "neutral" else self.source_codes
        if c.pairing == "shuffled":
            gallery_source = self.shuffled_codes
        reverse = query @ gallery_source.T
        if c.channel == "reverse":
            return reverse
        words = query_photo @ self.target_codes.T
        if c.channel == "wordcos":
            return words
        return (standardize(forward) + standardize(reverse) + standardize(words)) / 3.

    def evidence(self, query, c):
        ids, weights = self.query_codes(query, c)
        direct = self.direct_scores(query, ids, weights, c)
        if c.channel != "mixture" or c.direct == 1.:
            return standardize(direct)
        # At power 1 this is a shared-concept likelihood ratio under a common prior.
        # Keep the prior estimated on the entire gallery, independent of the incoming sketch.
        adjusted = weights / self.frequency[ids].pow(c.prior_power)
        blocks = []
        for choices, mass in zip(ids.split(16), adjusted.split(16), strict=True):
            scores = torch.bmm(mass[:, None, :], self.likelihood[choices].float()).squeeze(1)
            blocks.append(scores.clamp_min(1e-15).log())
        return standardize(torch.cat(blocks)).lerp(standardize(direct), c.direct)


def photo_graph(photos, k):
    geometry = F.normalize(photos - photos.mean(0))
    return mutual_transition(geometry, min(k, len(photos) - 1))


def propagate(base, transition, seeds, fraction):
    if not fraction:
        return base
    values, ids = base.topk(min(seeds, base.size(1)), dim=1)
    seed = torch.zeros_like(base).scatter_(1, ids, values.softmax(dim=1))
    posterior = solve_diffusion(transition, seed, .90, iterations=12)
    direct = base.log_softmax(1)
    walk = (posterior / posterior.sum(1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-30).log()
    return torch.logaddexp(direct + math.log1p(-fraction), walk + math.log(fraction))


def ranked_scores(scores):
    """Encode an ordered list as distinct representable values for the KD-SBIR evaluator."""
    order = scores.argsort(dim=1, descending=True, stable=True)
    rank = torch.linspace(.9, -.9, scores.size(1), device=scores.device).expand_as(scores)
    return torch.empty_like(scores).scatter_(1, order, rank)
