"""Integrate CLIP sketch likelihoods over correlated choices of public photo terms."""

import math

import torch
import torch.nn.functional as F

from .shared_concept_likelihood import fit_prior


def vocabulary_pairs(source, target, limit=1024, threshold=.92):
    # The dictionaries are sorted in exactly the same order as their public names.
    neighbors = []
    for start in range(0, len(source), 384):
        block = slice(start, start + 384)
        sim = (source[block] @ source.T + target[block] @ target.T) * .5
        row = torch.arange(len(sim), device=sim.device)
        sim[row, start + row] = -torch.inf
        neighbors.append(sim.argmax(1))
    neighbor = torch.cat(neighbors)
    idx = torch.arange(len(source), device=source.device)
    left = idx[(neighbor[neighbor] == idx) & (idx < neighbor)]
    right = neighbor[left]
    minimum = torch.minimum((source[left] * source[right]).sum(1), (target[left] * target[right]).sum(1))
    selected = minimum.argsort(descending=True, stable=True)
    selected = selected[minimum[selected] >= threshold][:limit]
    return left[selected], right[selected], minimum[selected]


def noise_bank(target, n_draws, seed):
    rng = torch.Generator(device=target.device).manual_seed(seed)
    gaussian = torch.randn((n_draws, target.shape[1]), generator=rng, device=target.device)
    projection = gaussian @ F.normalize(target).T
    return -(-torch.special.log_ndtr(projection)).log()


def winners(logits, noise, removed=None):
    result = []
    thinned = []
    for batch in logits.split(24):
        utility = batch[:, None, :] + noise[None, :, :]
        result.append(utility.argmax(-1).int())
        if removed is not None:
            utility[:, :, removed] = -torch.inf
            thinned.append(utility.argmax(-1).int())
    return torch.cat(result), torch.cat(thinned) if thinned else None


def choice_matrix(ids, vocabulary_size):
    n, samples = ids.shape
    rows = torch.arange(n, device=ids.device)[:, None]
    keys = (rows * vocabulary_size + ids.sort(dim=1).values).flatten()
    values, counts = torch.unique_consecutive(keys, return_counts=True)
    row_count = torch.bincount(values // vocabulary_size, minlength=n)
    row_ptr = F.pad(row_count.cumsum(0), (1, 0)).int()
    column = (values % vocabulary_size).int()
    matrix = torch.sparse_csr_tensor(row_ptr, column, counts.float() / samples, size=(n, vocabulary_size))
    # Unbiased among-draw collision frequency of the observed choice.
    collision = (counts.double() * (counts - 1)).sum() / (n * samples * (samples - 1))
    return matrix, float(collision)


def match_concentration(logits, collision):
    # Remove a scalar temperature explanation using only unlabeled gallery photos.
    reference = logits[torch.linspace(0, len(logits)-1, min(2048, len(logits))).long()]
    low, high = .01, 16.
    for _ in range(20):
        beta = math.sqrt(low * high)
        value = (reference * beta).softmax(1).square().sum(1).mean()
        if value < collision:
            low = beta
        else:
            high = beta
    return math.sqrt(low * high)


class WordChoices:
    samples = 256
    first_seed = 793187
    second_seed = 394733

    def __init__(self, photos, source, target, scale):
        left, right, distance = vocabulary_pairs(source, target)
        self.pairs = list(zip(left.tolist(), right.tolist(), distance.tolist(), strict=True))
        n, vocab = len(photos), len(target)
        device = photos.device
        rng = torch.Generator().manual_seed(97139)
        repeated = torch.randperm(vocab, generator=rng)[:vocab // 8].to(device)
        remap = torch.randperm(vocab, generator=rng).to(device)
        logits = (photos @ target.T) * scale
        probs = logits.softmax(1)
        self.probabilities = probs
        priors, _ = fit_prior(probs)
        self.prior = priors["mle"]
        self.marginal = (probs @ self.prior).clamp_min(1e-30)
        self.hard = probs.argmax(1)
        noise = noise_bank(target, self.samples, self.first_seed)
        ids, trimmed = winners(logits, noise, right)
        self.sparse = {}
        self.sparse["correlated"], collision = choice_matrix(ids, vocab)
        self.sparse["correlated_thin"], _ = choice_matrix(trimmed, vocab)
        ids_b, _ = winners(logits, noise_bank(target, self.samples, self.second_seed))
        self.sparse["second_seed"], _ = choice_matrix(ids_b, vocab)
        ids_shuffled, _ = winners(logits, noise[:, remap])
        self.sparse["wrong_geometry"], _ = choice_matrix(ids_shuffled, vocab)
        iid = torch.multinomial(probs, self.samples, replacement=True,
                                generator=torch.Generator(device=device).manual_seed(self.first_seed))
        self.sparse["iid_sample"], _ = choice_matrix(iid.int(), vocab)
        self.beta = match_concentration(logits, collision)
        self.tempered = (self.beta * logits).softmax(1)
        self.collision = {"correlated": collision,
                          "tempered":float(self.tempered.square().sum(1).mean()),
                          "original":float(probs.square().sum(1).mean())}
        self.repeated = repeated
        self.copy_mass = probs[:, repeated].sum(1)
        # Physically append duplicate alternatives on evenly spaced photos, under paired randomness.
        probe = torch.linspace(0, n-1, 48, device=device).long()
        original_ids = ids[probe].long()
        clone_logits = torch.cat((logits[probe], logits[probe][:, repeated]), dim=1)
        clone_noise = torch.cat((noise, noise[:, repeated]), dim=1)
        clone_ids, _ = winners(clone_logits, clone_noise)
        original_name = torch.cat((torch.arange(vocab, device=device), repeated))
        self.clone_check = {"draws": original_ids.numel(), "photos": len(probe),
                            "different_semantic_winners": int((original_name[clone_ids.long()] != original_ids).sum())}
        logits[:, right] = -torch.inf
        self.thinned_probabilities = logits.softmax(1)

    def seeds(self, likelihood):
        base = likelihood @ self.probabilities.T
        repeated = self.repeated
        yield "diffusion", base
        yield "gallery_mle", ((likelihood * self.prior) @ self.probabilities.T) / self.marginal
        yield "baseline_clones", (base + 7 * (likelihood[:, repeated] @ self.probabilities[:, repeated].T)) / (1 + 7*self.copy_mass)
        # Reevaluate the thinned distribution directly to avoid subtracting nearly equal dot products.
        yield "baseline_thin", likelihood @ self.thinned_probabilities.T
        yield "hard", likelihood[:, self.hard]
        yield "matched_temperature", likelihood @ self.tempered.T
        for name, probs in self.sparse.items():
            yield name, torch.sparse.mm(probs, likelihood.T.contiguous()).T
