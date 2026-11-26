"""Identify the smallest noun partition preserving specified decisions about an image."""

import torch

from .part_occlusion_observations import PAIRS


def decision_questions():
    # Each pair of regions can be removed with the other two regions visible,
    # or after withholding one of those other regions.
    context = []
    for left, right in PAIRS:
        for held in range(4):
            if held != left and held != right:
                a = 4 + PAIRS.index(tuple(sorted((left, held))))
                b = 4 + PAIRS.index(tuple(sorted((right, held))))
                context.append((a, b))
    return list(PAIRS), context


def decisions(observations, questions):
    left, right = zip(*questions, strict=True)
    return (observations[:, left] > observations[:, right]).to(torch.int8) - (
        observations[:, left] < observations[:, right]
    ).to(torch.int8)


def partitions(observations):
    single, context = decision_questions()
    simple, fine = [], []
    for obs in observations.split(512):
        code = torch.zeros_like(obs[:, 0], dtype=torch.int32)
        for group, output in ((single, simple), (context, fine)):
            for question in decisions(obs, group).unbind(1):
                code = code * 3 + (1 + question.to(torch.int32))
            output.append(code)
    return torch.cat(simple), torch.cat(fine)


def pooled_distribution(probabilities, codes):
    """For each noun return the full probability of its within-image state."""
    order = codes.argsort(dim=1, stable=True)
    sorted_codes = codes.gather(1, order)
    starts = torch.ones_like(sorted_codes, dtype=torch.bool)
    starts[:, 1:] = sorted_codes[:, 1:] != sorted_codes[:, :-1]
    groups = starts.cumsum(1) - 1
    mass = torch.zeros_like(probabilities).scatter_add_(1, groups, probabilities.gather(1, order))
    return torch.zeros_like(probabilities).scatter_(1, order, mass.gather(1, groups))


def gallery_control(p):
    # Exchange entire partitions between RGB images with the same most probable
    # public noun. Neighboring rows in each stratum have similar posterior norm.
    # Retain any unpaired last member, where a matched donor does not exist.
    head = p.argmax(1)
    radius = p.square().sum(1)
    permutation = torch.arange(len(p), device=p.device)
    for label in head.unique():
        members = (head == label).nonzero().flatten()
        order = members[radius[members].argsort(stable=True)]
        order = order[:2*(len(order)//2)].reshape(-1, 2)
        permutation[order.flatten()] = order.flip(1).flatten()
    return permutation


def collision_reference(q, p, q_state, p_state):
    """Exact joint state masses for one pair; no lifted word-pair features."""
    cells = torch.stack((q_state, p_state), dim=1)
    ids, counts = torch.unique(cells, dim=0, return_inverse=True, return_counts=True)[1:]
    qmass = q.new_zeros(len(counts)).scatter_add_(0, ids, q)
    pmass = p.new_zeros(len(counts)).scatter_add_(0, ids, p)
    expected = ((counts.double() * (counts - 1)).sum() / (len(q)*(len(q)-1))).item()
    dot = q @ p
    return qmass @ pmass, dot + (1 - dot)*expected


def pair_features(p, state, left, right, reverse=False):
    mask = (state[:, left] == state[:, right]).float()
    if reverse:
        left, right = right, left
    value = torch.cat((p[:, left] * mask, p[:, right] * mask), dim=1)
    return value, mask


def collision_matrix(q, p, qstate, pstate, *, count_opportunities=False):
    """Exhaustive FP32 matrix, streamed over unordered noun pairs on the GPU."""
    words = q.size(1)
    pairs = torch.triu_indices(words, words, offset=1, device=q.device)
    dot = q @ p.T
    extra = torch.zeros_like(dot)
    counts = torch.zeros_like(dot) if count_opportunities else None
    for block in pairs.split(1536, dim=1):
        left, right = block
        qfeat, qs = pair_features(q, qstate, left, right)
        pfeat, ps = pair_features(p, pstate, left, right, reverse=True)
        extra.addmm_(qfeat, pfeat.T)
        if count_opportunities:
            counts.addmm_(qs, ps.T)
        del qfeat, pfeat, qs, ps
    expected = dot + (1-dot) * (2/(words*(words-1))) * counts if count_opportunities else None
    return dot + extra, expected


def fixture(device):
    # Equal Dot, equal sizes of the two partitions and of their joint cells.
    # Only membership of the candidate's lexical alternative differs.
    x = torch.tensor([[.93, .01, .01, .01, .01, .01, .01, .01]], device=device)
    y = x.roll(1, dims=1).repeat(2, 1)
    sx = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], device=device)
    sy = torch.tensor([[2, 2, 3, 3, 2, 2, 3, 3], [2, 3, 2, 3, 2, 3, 2, 3]], device=device)
    real, random_identity = collision_matrix(x, y, sx, sy, count_opportunities=True)
    torch.testing.assert_close(random_identity[0, 0], random_identity[0, 1])
    if float(real[0, 0] - real[0, 1]) < .5:
        raise ValueError("Estimator confused cell size and lexical identity")
    for j in range(2):
        ref, chance = collision_reference(x[0], y[j], sx[0], sy[j])
        torch.testing.assert_close(real[0, j], ref)
        torch.testing.assert_close(random_identity[0, j], chance)
    def split_name(prob, fraction):
        return torch.cat((prob[:, :1]*fraction, prob[:, 1:], prob[:, :1]*(1-fraction)), 1)
    clone_x, clone_y = split_name(x, .2), split_name(y, .8)
    cloned, _ = collision_matrix(clone_x, clone_y, torch.cat((sx, sx[:, :1]), 1),
                                 torch.cat((sy, sy[:, :1]), 1))
    torch.testing.assert_close(real, cloned)
    isolated = torch.arange(8, device=device)[None, :]
    direct, _ = collision_matrix(x, y, isolated, sy)
    torch.testing.assert_close(direct, x @ y.T)
    return {"same_opportunity": random_identity.tolist()[0], "quotient_scores": real.tolist()[0],
            "after_domain_specific_name_split": cloned.tolist()[0], "literal": (x @ y.T).tolist()[0]}
