"""Use unlabeled gallery word choices to define the measure in Bayes' rule."""

import argparse
import os
from pathlib import Path

import torch

from .correlated_choice_study import Study, run
from .correlated_lexical_choice import WordChoices
from .shared_concept_likelihood import fit_prior


def word_frequency(choices):
    # Accumulate counts across photos; the sum of each sparse row is one.
    n, words = choices.shape
    mass = torch.zeros(words, device=choices.device, dtype=torch.double)
    mass.scatter_add_(0, choices.col_indices().long(), choices.values().double())
    return (mass / n).float()


class GalleryReference(WordChoices):
    def __init__(self, photos, source, target, scale):
        super().__init__(photos, source, target, scale)
        natural, thinned = self.probabilities, self.thinned_probabilities
        pairs = torch.as_tensor([(left, right) for left, right, _ in self.pairs],
                                device=photos.device, dtype=torch.long)
        balanced = torch.ones_like(self.prior)
        balanced[pairs.flatten()] = .5
        priors = {
            "reference": word_frequency(self.sparse["correlated"]),
            "second_prior": word_frequency(self.sparse["second_seed"]),
            "misaligned_geometry": word_frequency(self.sparse["wrong_geometry"]),
            "soft_prior": natural.mean(0),
            "sharp_prior": self.tempered.mean(0),
            "hard_prior": torch.bincount(self.hard, minlength=len(target)).float() / len(photos),
            "balanced_pairs": balanced / balanced.sum(),
            "gallery_mle": self.prior,
        }
        self.weighted = {}
        for name, prior in priors.items():
            self.register(name, natural, prior)
        self.register("reference_thin", thinned, word_frequency(self.sparse["correlated_thin"]))
        thin_priors, _ = fit_prior(thinned)
        self.register("mle_thin", thinned, thin_priors["mle"])
        del self.sparse, self.tempered

    def register(self, name, likelihood, prior):
        normalizer = likelihood @ prior
        if not torch.all(normalizer > 0):
            raise ValueError("No vocabulary support for a gallery photo in " + name)
        self.weighted[name] = (likelihood, prior, normalizer)

    def seeds(self, likelihood):
        yield "diffusion", likelihood @ self.probabilities.T
        yield "baseline_thin", likelihood @ self.thinned_probabilities.T
        for name, (gallery, prior, normalizer) in self.weighted.items():
            yield name, ((likelihood * prior) @ gallery.T) / normalizer


REFERENCE = Study(
    "lexical-reference-measure", GalleryReference, "reference",
    ("gallery_mle", "hard_prior", "sharp_prior", "misaligned_geometry", "soft_prior",
     "balanced_pairs", "second_prior"),
    {"softmax_thinning": ("diffusion", "baseline_thin"),
     "reference_thinning": ("reference", "reference_thin"),
     "mle_thinning": ("gallery_mle", "mle_thin"),
     "two_reference_seeds": ("reference", "second_prior")},
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--workers", type=int, default=min(10, len(os.sched_getaffinity(0))))
    args = parser.parse_args()
    with torch.inference_mode():
        run(args.workspace, args.workers, REFERENCE)
