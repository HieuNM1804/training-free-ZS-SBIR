"""Measure whether gallery donors support a query through the same public concepts."""

from types import SimpleNamespace

import torch

from .cross_diffusion_map import Config, Index


class ConceptAgreement:
    def __init__(self, photos, probabilities, *, permutation_control=False):
        self.photos = probabilities
        self.config = Config(neighbors=16, bandwidth=.36, iterations=2, mode="rgb")
        self.graph = Index(SimpleNamespace(p=photos), rgb_only=True)
        self.graph.prepare(self.config)
        self._build_word_index(permutation_control)

    def diffuse(self, signals):
        return self.graph.score(self.config, signals)

    def _build_word_index(self, permutation_control):
        p = self.photos
        n, vocab = p.shape
        self.agreement_terms = torch.empty_like(p)
        self.within_half = None
        if permutation_control:
            rng = torch.Generator().manual_seed(48971)
            group = torch.zeros(n, dtype=p.dtype, device=p.device)
            group[torch.randperm(n, generator=rng)[:n // 2]] = 1
            self.word_permutation = torch.randperm(vocab, generator=rng).to(p.device)
            group_a = torch.empty_like(p)
            group_b = torch.empty_like(p)
        for start in range(0, vocab, 384):
            col = slice(start, start + 384)
            root_p = p[:, col].sqrt()
            mean = self.diffuse(root_p.T).T
            self.agreement_terms[:, col] = mean.square()
            if permutation_control:
                part = self.diffuse((root_p * group[:, None]).T).T
                group_a[:, col] = part
                group_b[:, col] = (mean - part).clamp_min(0)
        self.flat_agreement = self.agreement_terms.sum(1).clamp_min(1e-30)
        if permutation_control:
            self.cross_half = torch.empty_like(p)
            for start in range(0, vocab, 384):
                col = slice(start, start + 384)
                self.cross_half[:, col] = 2 * group_a[:, col] * group_b[:, self.word_permutation[col]]
            self.within_half = group_a.square_().addcmul_(group_b, group_b)
            self.shuffled_flat = (self.within_half.sum(1) + self.cross_half.sum(1)).clamp_min(1e-30)

    def scores(self, likelihood):
        seed = likelihood @ self.photos.T
        diffused = self.diffuse(seed)
        scalar_root = self.diffuse(seed.sqrt()).square().clamp_min_(1e-30)
        shared_word = likelihood @ self.agreement_terms.T
        agreement = shared_word / scalar_root
        conditional = diffused * agreement
        scores = {
            "direct": seed,
            "diffusion": diffused,
            "conditional": conditional,
            "relative": conditional / self.flat_agreement,
            "scalar_root": scalar_root,
            "word_root": shared_word / self.flat_agreement,
            "flat_bias": diffused / self.flat_agreement,
        }
        if self.within_half is not None:
            # Relabel the posterior coordinates in one random half of the donor gallery.
            # The query likelihood follows the same permutation inside that half.
            cross_likelihood = likelihood.sqrt() * likelihood[:, self.word_permutation].sqrt()
            shuffled = likelihood @ self.within_half.T + cross_likelihood @ self.cross_half.T
            scores["permuted_conditional"] = diffused * (shuffled / scalar_root)
            scores["permuted_relative"] = scores["permuted_conditional"] / self.shuffled_flat
        return scores, agreement
