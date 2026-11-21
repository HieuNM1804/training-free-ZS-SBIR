"""Translate between public words using CLIP text neighbors with local bandwidths."""

from dataclasses import dataclass, replace

import torch

from .shared_concept_likelihood import emission


DECISION = "public-semantic-translation/strict-joint.json"
USES_LITERAL_BANK = True
NEIGHBORS = 7


@dataclass(frozen=True)
class Config:
    support: str = "all"
    mass: float = .5
    bandwidth: str = "local"
    shuffle: bool = False


def canonical():
    return Config()


def candidates():
    return [Config(support, mass) for support in ("all", "nearest") for mass in (.25, .5, 1.)]


def controls(c):
    return {"primary": c, "original": canonical(), "literal_match": replace(c, mass=0.),
            "translation_only": replace(c, mass=1.), "checkpoint_bandwidth": replace(c, bandwidth="native"),
            "permuted_translation": replace(c, shuffle=True),
            "alternative_support": replace(c, support="nearest" if c.support=="all" else "all")}


def cache_key(c):
    return c.support, c.bandwidth, c.shuffle, c.mass


class Lexicon:
    def __init__(self, bank, scale):
        self.bank = bank
        self.scale = scale
        radii, neighbors = [], []
        for start in range(0, len(bank), 512):
            distances = self.distances(start)
            near, ids = distances.topk(NEIGHBORS, largest=False)
            radii.append(near[:, -1].clamp_min(1e-8).sqrt())
            neighbors.append(ids)
        self.radii = torch.cat(radii)
        self.neighbors = torch.cat(neighbors)
        self.permutation = torch.randperm(len(bank), generator=torch.Generator().manual_seed(7180)).to(bank.device)

    def distances(self, start):
        block = self.bank[start:start+512]
        d = (block @ self.bank.T).mul_(-2.).add_(2.).clamp_min_(0.)
        ids = torch.arange(len(block), device=d.device)
        d[ids, start+ids] = torch.inf
        return d

    def translate(self, queries, c):
        translated = torch.empty_like(queries)
        columns = queries.T.contiguous() if c.support == "nearest" else None
        for start in range(0, len(self.bank), 512):
            d = self.distances(start)
            n = len(d)
            if c.bandwidth == "local":
                logits = -d / (self.radii[start:start+n, None] * self.radii)
            elif c.bandwidth == "uniform":
                logits = torch.zeros_like(d).masked_fill_(~d.isfinite(), -torch.inf)
            else:
                logits = d * (-.5*self.scale)
            if c.support == "nearest":
                ids = self.neighbors[start:start+n]
                weights = logits.gather(1, ids).softmax(1)
                # A small CSR matrix applies the same public seven-word channel to every image.
                offsets = torch.arange(0, ids.numel()+1, NEIGHBORS, device=d.device)
                channel = torch.sparse_csr_tensor(offsets, ids.flatten(), weights.flatten(),
                                                  size=(n, len(self.bank)))
                translated[:, start:start+n] = (channel @ columns).T
            else:
                translated[:, start:start+n] = queries @ logits.softmax(1).T
        return translated[:, self.permutation] if c.shuffle else translated


class Index:
    def __init__(self, data):
        self.data = data
        self.lexicon = Lexicon(data.literal, data.scale)
        quantiles = torch.tensor([.1, .5, .9], device=data.q.device)
        self.info = {"text_only": True, "neighbors_for_radius": NEIGHBORS,
                     "radius_deciles_and_median": torch.quantile(self.lexicon.radii, quantiles).tolist()}
        self.last_key = None
        self.last_score = None

    def score(self, c):
        if c.mass == 0.:
            return self.data.base
        key = c.support, c.bandwidth, c.shuffle
        if self.last_key != key:
            self.last_score = None
            translated = self.lexicon.translate(self.data.query_emission, c)
            self.last_score = translated @ self.data.a.T
            self.last_key = key
        return self.data.base.lerp(self.last_score, c.mass)

    def score_images(self, c, sketches, photos):
        d = self.data
        q = emission(sketches, d.source, d.scale)
        p = emission(photos, d.target, d.scale)
        p /= p.sum(1, keepdim=True)
        if c.mass:
            q = q.lerp(self.lexicon.translate(q, c), c.mass)
        return q @ p.T
