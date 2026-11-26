"""Infer a region correspondence from single deletions and predict double deletions."""

from itertools import permutations
import math

import torch
import torch.nn.functional as F

from .part_occlusion_observations import PAIRS


ARMS = ("heldout", "uniform_correspondence", "fixed_layout", "pooled_double", "additive_double")
GALLERY_TILE = 768


class Correspondences:
    def __init__(self, device):
        maps = list(permutations(range(4)))
        singles = torch.tensor(maps, device=device)
        edges = torch.tensor([[PAIRS.index(tuple(sorted((pi[i], pi[j])))) for i, j in PAIRS]
                              for pi in maps], device=device)
        self.single_assignment = F.one_hot(singles, num_classes=4).flatten(1).T.contiguous().float()
        self.double_assignment = F.one_hot(edges, num_classes=6).flatten(1).T.contiguous().float()
        self.log_count = math.log(len(maps))


def additive_predictions(observed, whole):
    """Null with additive deletion effects on all pairwise word log odds."""
    left, right = zip(*PAIRS, strict=True)
    output = []
    for part, baseline in zip(observed.split(512), whole.split(512), strict=True):
        singles = part[:, :4].clamp_min(1e-30).log()
        logits = singles[:, left] + singles[:, right] - baseline.clamp_min(1e-30).log()[:, None, :]
        output.append(logits.softmax(2))
    return torch.cat(output)


def overlap(left, right):
    n, masks, words = left.shape
    m = len(right)
    values = left.reshape(-1, words) @ right.reshape(-1, words).T
    return values.reshape(n, masks, m, masks).transpose(1, 2).reshape(n, m, masks*masks)


def score_tile(query, photo, additive_query, additive_photo, maps):
    a = overlap(query[:, :4], photo[:, :4]).clamp_min(1e-30).log() @ maps.single_assignment
    log_weights = a.log_softmax(2)
    observed = overlap(query[:, 4:], photo[:, 4:])
    b = observed.clamp_min(1e-30).log() @ maps.double_assignment
    synthetic = overlap(additive_query, additive_photo).clamp_min(1e-30).log() @ maps.double_assignment
    return {
        "heldout": torch.logsumexp(log_weights + b, 2) / 6,
        "uniform_correspondence": (torch.logsumexp(b, 2) - maps.log_count) / 6,
        "fixed_layout": b[:, :, 0] / 6,
        "pooled_double": observed.mean(2).clamp_min(1e-30).log(),
        "additive_double": torch.logsumexp(log_weights + synthetic, 2) / 6,
    }


def scores(query, photo, additive_query, additive_photo, maps):
    output = {arm: query.new_empty((len(query), len(photo))) for arm in ARMS}
    for start in range(0, len(photo), GALLERY_TILE):
        sl = slice(start, min(start + GALLERY_TILE, len(photo)))
        tile = score_tile(query, photo[sl], additive_query, additive_photo[sl], maps)
        for arm, value in tile.items():
            output[arm][:, sl] = value
    return output


def enumerate_pair(q, p):
    # A scalar reference computed in probability space in double precision.
    # 601-way real posteriors and ten independent events stay in float64 range.
    q, p = q.double().cpu(), p.double().cpu()
    numerator, denominator = 0., 0.
    for pi in permutations(range(4)):
        a = math.prod(float(q[i] @ p[j]) for i, j in enumerate(pi))
        b = math.prod(float(q[4 + k] @ p[4 + PAIRS.index(tuple(sorted((pi[i], pi[j]))))])
                      for k, (i, j) in enumerate(PAIRS))
        numerator += a*b
        denominator += a
    return math.log(numerator / denominator) / 6


def fixture(device, maps):
    single = .99*torch.eye(10, device=device) + .001
    x = single[None]
    pi = (1, 2, 3, 0)
    linked = [PAIRS.index(tuple(sorted((pi[i], pi[j])))) for i, j in PAIRS]
    together = torch.cat((x[:, pi], x[:, [4+j for j in linked]]), 1)
    broken = torch.cat((x[:, :4], x[:, [4+j for j in linked]]), 1)
    candidates = torch.cat((x, together, broken))
    result = score_tile(x, candidates, x[:, 4:], candidates[:, 4:], maps)
    true, renamed, inconsistent = result["heldout"][0]
    torch.testing.assert_close(true, renamed)
    torch.testing.assert_close(result["uniform_correspondence"], result["uniform_correspondence"][0, 0].expand(1, 3))
    if not (true - inconsistent > 1):
        raise ValueError("Failed to detect inconsistent identities across the two mask orders")
    for i, value in enumerate(result["heldout"][0]):
        if abs(value.item() - enumerate_pair(x[0], candidates[i])) > 1e-5:
            raise ValueError("Disagreement with exhaustive conditional-probability calculation")
    z, y = x[:, :1].expand(-1, 10, -1), x[:, 1:2].expand(-1, 10, -1)
    constant = score_tile(z, y, z[:, 4:], y[:, 4:], maps)
    base = (z[0, 0] @ y[0, 0]).log()
    for score in constant.values():
        torch.testing.assert_close(score.squeeze(), base)
    return {"consistent_renamed_broken": result["heldout"].tolist()[0],
            "all_three_uniform": result["uniform_correspondence"].tolist()[0],
            "constant_report_log_dot": base.item()}
