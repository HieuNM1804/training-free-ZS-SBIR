"""Index each photo in the query domain, then expand image evidence on the target gallery."""

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from .concept_likelihood import photo_graph
from .gallery_diffusion import solve_diffusion
from .reranking import standardize


@dataclass(frozen=True, order=True)
class Setting:
    terms: int = 16
    word_temp: float = .03
    neighbors: int = 96
    diffusion: float = .75
    support: int = 96
    expansion: float = .6
    expand_temp: float = .6
    direction: str = "reverse"
    pairing: str = "paired"

    @property
    def name(self):
        return (f"{self.direction}-{self.pairing}-c{self.terms}-t{self.word_temp:g}-k{self.neighbors}"
                f"-d{self.diffusion:g}-n{self.support}-e{self.expansion:g}-u{self.expand_temp:g}")


def candidates():
    base = Setting()
    grid = {replace(base, diffusion=d, expansion=e, direction=v)
            for v in ("reverse", "forward", "average")
            for d in (0., .5, .85) for e in (0., .35, .7, 1.)}
    for attr, values in {
        "terms": (4, 8, 32, 64, 128), "word_temp": (.012, .02, .05), "neighbors": (32, 192, 320),
        "support": (16, 48, 192, 384), "expand_temp": (.25, 1., 2.), "expansion": (.85,),
        "diffusion": (.92,),
    }.items():
        grid.update(replace(base, **{attr: value}) for value in values)
    return sorted(grid | {base, replace(base, expansion=0., diffusion=0.)})


def matched_controls(winner):
    single = replace(winner, expansion=0., diffusion=0.)
    return {
        "primary": winner, "no_expansion": replace(winner, expansion=0.),
        "no_diffusion": replace(winner, diffusion=0.), "no_context": single,
        "reverse": replace(winner, direction="reverse"), "forward": replace(winner, direction="forward"),
        "reverse_alone": replace(single, direction="reverse"), "forward_alone": replace(single, direction="forward"),
        "shuffled_pairs": replace(winner, pairing="shuffled"),
        "neutral_prompts": replace(winner, pairing="neutral"),
        "native_same_context": replace(winner, direction="native"),
    }


def term_average(vectors, mapping, indices, values, count, temperature):
    weights = (values[:, :count] / temperature).softmax(1)
    pooled = []
    for ids, mass in zip(indices[:, :count].split(256), weights.split(256), strict=True):
        pooled.append(F.normalize(torch.bmm(mass[:, None], vectors[mapping[ids]]).squeeze(1)))
    return torch.cat(pooled)


def nearest_words(images, prompts, width=128):
    rows = [(block @ prompts.T).topk(min(width, len(prompts))) for block in images.split(384)]
    return torch.cat([r.indices for r in rows]), torch.cat([r.values for r in rows])


class PhotoIndex:
    def __init__(self, photos, source_words, photo_words):
        self.rgb, self.source, self.target = photos, source_words, photo_words
        self.visual = F.normalize(photos - photos.mean(0))
        self.ids, self.values = nearest_words(photos, photo_words)
        self.identity = torch.arange(len(photo_words), device=photos.device)
        self.permutation = torch.randperm(len(photo_words), generator=torch.Generator().manual_seed(2179)).to(photos.device)
        self.graphs, self.word_banks, self.visual_banks = {}, {}, {}

    def smooth(self, features, k, alpha):
        if alpha == 0.:
            return features
        if k not in self.graphs:
            self.graphs[k] = photo_graph(self.rgb, k)
        # Iscen's symmetrically normalized mutual graph. Each column is a feature channel.
        result = solve_diffusion(self.graphs[k], features.T, alpha, iterations=12)
        return F.normalize(result.T.contiguous())

    def index(self, c):
        key = (c.neighbors, c.diffusion)
        if key not in self.visual_banks:
            self.visual_banks[key] = self.smooth(self.visual, *key)
        word_key = (c.terms, c.word_temp, c.pairing, *key)
        if word_key not in self.word_banks:
            bank = self.target if c.pairing == "neutral" else self.source
            mapping = self.permutation if c.pairing == "shuffled" else self.identity
            words = term_average(bank, mapping, self.ids, self.values, c.terms, c.word_temp)
            self.word_banks[word_key] = self.smooth(words, *key)
        return self.word_banks[word_key], self.visual_banks[key]

    def first_scores(self, query, c):
        words, visual = self.index(c)
        reverse = standardize(query @ words.T)
        if c.direction == "reverse":
            return reverse, visual
        fkey = ("rgb", c.neighbors, c.diffusion)
        if fkey not in self.visual_banks:
            self.visual_banks[fkey] = self.smooth(self.rgb, c.neighbors, c.diffusion)
        if c.direction == "native":
            return standardize(query @ self.visual_banks[fkey].T), visual
        source = self.target if c.pairing == "neutral" else self.source
        ids, values = nearest_words(query, source, c.terms)
        mapping = self.permutation.argsort() if c.pairing == "shuffled" else self.identity
        txt = term_average(self.target, mapping, ids, values, c.terms, c.word_temp)
        forward = standardize(txt @ self.visual_banks[fkey].T)
        return forward if c.direction == "forward" else (reverse + forward) * .5, visual

    def scores(self, query, c, block_size=160):
        for block in query.split(block_size):
            seed, visual = self.first_scores(block, c)
            if c.expansion:
                values, ids = seed.topk(min(c.support, len(visual)))
                weights = (values / c.expand_temp).softmax(1)
                feedback = F.normalize(torch.bmm(weights[:, None], visual[ids]).squeeze(1))
                seed = seed.lerp(standardize(feedback @ visual.T), c.expansion)
            yield seed
