import unittest
import torch
from torch.nn import functional as F

from tf_sbir.global_methods import centered
from tf_sbir.phase_agreement import fusion, group_scores, Config, phase_features
from tf_sbir.paired_neighborhood_scale import scale_index


class PhaseAgreementTests(unittest.TestCase):
    def test_polarities_fuse_symmetrically(self):
        g = torch.Generator().manual_seed(332)
        raw = torch.randn(3, 9, generator=g)
        v = torch.randn(3, 12, 9, generator=g)
        swapped = v.clone()
        swapped[:, 4:] = v[:, 4:].unflatten(1, (2, 2, 2)).flip(2).flatten(1, 3)
        first = phase_features(raw, v, torch.eye(9), Config())
        second = phase_features(raw, swapped, torch.eye(9), Config())
        for a, b in zip(first, second, strict=True):
            torch.testing.assert_close(a, b)

    def test_each_query_uses_only_its_features(self):
        g = torch.Generator().manual_seed(212)

        def units(*shape):
            return F.normalize(torch.randn(*shape, generator=g), dim=-1)

        raw, a, p = units(31, 9), units(31, 9), units(31, 9)
        means = a.mean(0), p.mean(0)
        index = scale_index(raw, centered(a, means[0]), centered(p, means[1]), 5)
        q, t = units(3, 5, 9), units(3, 4, 9)
        tail, heads = group_scores(q, t, means, index, ("adaptive", "cross"))
        single, h = group_scores(q[1:2], t[1:2], means, index, ("adaptive", "cross"))
        torch.testing.assert_close(single, tail[1:2])
        for key in heads:
            torch.testing.assert_close(heads[key][1:2], h[key])
        for mode in ("direct", "cross", "adaptive", "rgb", "phase", "rgb-to-phase", "cross75", "peak"):
            w = fusion(mode, torch.zeros(3, 4))
            torch.testing.assert_close(w.sum(1), torch.ones(3))
