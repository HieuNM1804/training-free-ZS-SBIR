"""Build two gallery-grounded readings of a shared word and score one sketch at a time."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .foundation import load_words
from .selective_concept_listener import compile_index
from .shared_concept_likelihood import fit_prior


def principal_axes(photos, weights):
    mass = weights.sum(0).clamp_min(1e-12)
    means = (weights.T @ photos) / mass[:, None]
    generator = torch.Generator().manual_seed(29177)
    axes = F.normalize(torch.randn(means.shape, generator=generator).to(means))
    for _ in range(24):
        centered = photos @ axes.T - (means * axes).sum(1)
        moment = weights * centered
        covariance = moment.T @ photos - moment.sum(0)[:, None] * means
        axes = F.normalize(covariance)
    return axes


def quantiles(train, weights, test):
    """Evaluate the weighted CDF of each column, averaging the two limits at ties."""
    values, indices = train.T.contiguous().sort(dim=1, stable=True)
    sorted_weights = weights.T.gather(1, indices)
    cumulative = F.pad(sorted_weights.double().cumsum(1), (1, 0))
    cumulative = cumulative / cumulative[:, -1:].clamp_min(1e-20)
    values = values.contiguous()
    test = test.T.contiguous()
    left = torch.searchsorted(values, test)
    right = torch.searchsorted(values, test, right=True)
    return (.5 * (cumulative.gather(1, left) + cumulative.gather(1, right))).T.float()


def contexts(photos, weights, gates, target_words, scale, heads):
    """Rank possible contexts by their contribution to each end's KL divergence."""
    upper = weights * gates
    weights_both = torch.cat((upper, weights - upper), 1)
    support = weights.new_zeros((weights_both.shape[1], len(target_words)))
    for start in range(0, len(photos), 256):
        part = slice(start, start + 256)
        image_words = ((photos[part] @ target_words.T) * scale).softmax(1)
        support.addmm_(weights_both[part].T, image_words)
    plus, minus = support.chunk(2)
    both = plus + minus
    marginal = both / both.sum(1, keepdim=True).clamp_min(1e-20)
    answer = []
    for end in (plus, minus):
        freq = end / end.sum(1, keepdim=True).clamp_min(1e-20)
        information = freq * (freq.clamp_min(1e-20).log() - marginal.clamp_min(1e-20).log())
        information.scatter_(1, heads[:, None], -torch.inf)
        answer.append(information.argmax(1))
    return answer


@dataclass
class Fold:
    ids: torch.Tensor
    weights: torch.Tensor
    difference: torch.Tensor
    fraction: torch.Tensor
    source_texts: dict


class SenseIndex:
    def __init__(self, cfg, words, photos, source, target, scale, size=64):
        emission = photos.new_empty((len(photos), len(words)))
        for start in range(0, len(photos), 384):
            sl = slice(start, start + 384)
            emission[sl] = ((photos[sl] @ target.T) * scale).softmax(1)
        prior = fit_prior(emission)[0]["mle"]
        del emission
        heads, weights, _ = compile_index(photos, target, scale, prior, size, "visual_rgb")
        self.source = source[heads]
        self.scale = scale
        self.folds = []
        self.diagnostics = []
        generator = torch.Generator().manual_seed(61409)
        first, second = torch.randperm(len(photos), generator=generator).to(photos.device).tensor_split(2)
        for train_ids, apply_ids in ((first, second), (second, first)):
            train, b = photos[train_ids], weights[train_ids]
            apply, a = photos[apply_ids], weights[apply_ids]
            axis = principal_axes(train, b)
            coordinates = train @ axis.T
            g = quantiles(coordinates, b, coordinates)
            left, right = contexts(train, b, g, target, scale, heads)
            ids = torch.cat((left, right))
            prompts = [f"{words[i]} ({words[j]})" for i,j in zip(heads.repeat(2).tolist(), ids.tolist(), strict=True)]
            source_phrases, target_phrases = (v.to(photos) for v in load_words(cfg, prompts))
            readings = {
                "phrase": source_phrases.chunk(2),
                "sum": F.normalize(self.source.repeat(2, 1) + source[ids]).chunk(2),
                "context": source[ids].chunk(2),
            }
            heldout = quantiles(coordinates, b, apply @ axis.T)
            # Cast the two small vectors after reduction to preserve tiny word masses.
            mass = a.sum(0, dtype=torch.float64).clamp_min(1e-20)
            fraction = ((a * heldout).sum(0, dtype=torch.float64) / mass).float()
            difference = a * (heldout - fraction)
            rp, rm = target_phrases.chunk(2)
            effect = (difference * (apply @ (rp - rm).T)).sum(0, dtype=torch.float64) / mass
            context_effect = (difference * (apply @ (target[left] - target[right]).T)).sum(0, dtype=torch.float64) / mass
            residual = float((difference.sum(0, dtype=torch.float64).abs()/mass).max())
            if residual > 5e-6:
                raise ValueError("Conditional sense weights do not conserve gallery word mass")
            self.diagnostics.append({
                "photos_used_for_names": len(train), "photos_scored": len(apply),
                "mass_error": residual,
                "heldout_positive_context_covariance_share": float((effect > 0).double().mean()),
                "heldout_mean_context_covariance": float(effect.mean()),
                "heldout_auxiliary_word_covariance": float(context_effect.mean()),
                "same_context_fraction": float((left == right).float().mean()),
                "mean_gate_strength": float((a*(heldout-fraction).abs()).sum()/a.sum()),
                "word_contexts": [{"head":words[h],"positive":words[u],"negative":words[v],
                                   "positive_weight":float(p),"heldout_covariance":float(c)}
                                  for h,u,v,p,c in zip(heads.tolist(),left.tolist(),right.tolist(),
                                                     fraction.tolist(),effect.tolist(),strict=True)],
            })
            self.folds.append(Fold(apply_ids, a, difference, fraction, readings))
        self.photo_count = len(photos)

    def raw_scores(self, queries, kind, pooled_share, temperature, assignment):
        anchor = (queries @ self.source.T) * self.scale
        shift = anchor.amax(1, keepdim=True)
        result = queries.new_empty((len(queries), self.photo_count))
        for fold in self.folds:
            tp, tm = fold.source_texts[kind]
            plus, minus = queries @ tp.T * self.scale, queries @ tm.T * self.scale
            p = fold.fraction
            eps = torch.finfo(queries.dtype).tiny
            average = torch.logaddexp(plus + p.clamp_min(eps).log(), minus + (1-p).clamp_min(eps).log())
            amplitude = ((1-pooled_share)*anchor + pooled_share*average - shift).exp()
            evidence = amplitude @ fold.weights.T
            if assignment != "pooled":
                delta = (plus-minus)*temperature
                # Divide both likelihoods by their maximum to avoid exponential overflow.
                rp = delta.clamp_max(0).exp()
                rm = (-delta).clamp_max(0).exp()
                ratio = (rp-rm)/(p*rp+(1-p)*rm).clamp_min(eps)
                if assignment == "reverse":
                    # Swap the two likelihoods, retaining a positive, normalized score.
                    ratio = (rm-rp)/(p*rm+(1-p)*rp).clamp_min(eps)
                elif assignment != "conditional":
                    raise ValueError(assignment)
                evidence = evidence + (amplitude*ratio) @ fold.difference.T
            # Individual terms are nonnegative; allow cancellation at the last ulp.
            if not torch.isfinite(evidence).all() or (evidence < -1e-5*amplitude.amax(1,keepdim=True)).any():
                raise ValueError("Invalid conditional word likelihood")
            result[:, fold.ids] = evidence.clamp_min(eps)
        return result
