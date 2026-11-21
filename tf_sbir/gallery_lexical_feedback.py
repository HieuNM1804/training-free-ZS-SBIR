"""Invert or reinforce the lexical observations predicted by a fixed gallery index."""

from types import SimpleNamespace

import torch

from .cross_diffusion_map import Config, Index


def normalize_rows(x):
    return x / x.sum(1, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)


class LexicalFeedback:
    def __init__(self, photos, photo_words, scale):
        self.config = Config(neighbors=16, bandwidth=.36, iterations=2, mode="rgb")
        self.graph = Index(SimpleNamespace(p=photos), rgb_only=True)
        self.graph.prepare(self.config)
        self.raw = torch.softmax(scale * (photos @ photo_words.T), 1)
        self.index = torch.empty_like(self.raw)
        for start in range(0, self.raw.size(1), 384):
            sl = slice(start, start + 384)
            self.index[:, sl] = self.graph.score(self.config, self.raw[:, sl].T.contiguous()).T
        self.mass_error = float((self.index.sum(1) - 1).abs().max())
        self.index.div_(self.index.sum(1, keepdim=True))
        self.count = self.index.sum(0)
        self.marginal = self.count / len(photos)
        self.excluded = torch.empty_like(self.index)
        for start in range(0, len(photos), 256):
            row = self.index[start:start + 256]
            self.excluded[start:start + 256] = row.square() / (self.count - row).clamp_min(1e-30)
        generator = torch.Generator().manual_seed(81971)
        self.perm = torch.randperm(len(photo_words), generator=generator).to(photos.device)

    def evaluate(self, likelihood):
        b = self.index
        seed = likelihood @ b.T
        original = self.graph.score(self.config, likelihood @ self.raw.T)
        a = normalize_rows(seed)
        target = normalize_rows(likelihood * self.marginal)
        predicted = a @ b
        gain = predicted / self.marginal.clamp_min(1e-30)
        matched = likelihood * gain
        witness = matched @ b.T
        echo = gain @ b.T
        # For each candidate subtract its own contribution from the word-conditioned
        # witness population, renormalizing that population separately for each word.
        witness_loo = witness + matched @ self.excluded.T
        normalizer = (likelihood * self.marginal).sum(1, keepdim=True).clamp_min(1e-30)
        witness_loo = witness_loo - (seed / normalizer) * (likelihood @ self.excluded.T)
        rows = {"diffusion": original, "factorized": seed, "translated": echo,
                "independent_words": seed * echo, "shared_word": witness,
                "distinct_witness": witness_loo, "sharp_words": likelihood.square() @ b.T,
                "shuffled_word_response": (likelihood * gain[:, self.perm]) @ b.T}
        gain2 = None
        objective = []
        for it in range(1, 5):
            pred = a @ b if it > 1 else predicted
            obs = target.clamp_min(1e-30)
            pred = pred.clamp_min(1e-30)
            objective.append((target * (obs.log() - pred.log())).sum(1))
            if it == 4:
                break
            multiplier = (target / pred) @ b.T
            if it == 1:
                gain2 = multiplier
            a = normalize_rows(a * multiplier)
            if it+1 in (2, 4):
                rows["inverse_" + str(it+1)] = a
        rows["inverse_residual_only"] = gain2
        return rows, torch.stack(objective, 1)

    def validate(self, likelihood):
        # Compare the leave-one-out formula against explicitly removing a row.
        b = self.index
        l = likelihood[:1].double()
        candidates = torch.linspace(0, len(b)-1, 11, device=b.device).long()
        weighted = l.new_zeros((len(candidates), b.size(1)))
        count = torch.zeros_like(weighted)
        total = l.new_zeros(())
        for start in range(0, len(b), 256):
            block = b[start:start+256].double()
            seed = l @ block.T
            total += seed.sum()
            ids = torch.arange(start, start+len(block), device=b.device)
            keep = (candidates[:, None] != ids[None, :]).double()
            weighted += (keep * seed) @ block
            count += keep @ block
        other_mean = weighted / count.clamp_min(1e-300)
        values = (l * b[candidates].double() * other_mean).sum(1) * len(b) / total
        return candidates, values.float()
