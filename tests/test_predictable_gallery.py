import unittest

import torch
from torch.nn import functional as F

from tf_sbir.predictable_gallery import disjoint_anchor_support, predictable_residuals


class PredictableGalleryTests(unittest.TestCase):
    def test_reference_halves_are_disjoint_and_each_candidate_is_excluded(self):
        features = F.normalize(torch.randn(15, 6, generator=torch.Generator().manual_seed(7)), dim=-1)
        first, second = disjoint_anchor_support(features, 5, seed=17)
        self.assertEqual(first.shape, (15, 5))
        for i, (a, b) in enumerate(zip(first.tolist(), second.tolist())):
            self.assertNotIn(i, a + b)
            self.assertFalse(set(a) & set(b))
        swapped_order = disjoint_anchor_support(features, 5, seed=17)
        torch.testing.assert_close(first, swapped_order[0])

    def test_reliability_is_cross_half_error_agreement_and_ignores_the_target_axes(self):
        target = torch.tensor([[1., 2.], [1., 1.], [0., 0.]])
        first = torch.tensor([[0., 2.], [0., 1.], [0., 0.]])
        second = torch.tensor([[0., 2.], [2., 1.], [0., 0.]])
        residual, reliability = predictable_residuals(target, first, second)
        torch.testing.assert_close(reliability[:, 0], torch.tensor([1., 0., 0.]))
        rotation = torch.tensor([[0., -1.], [1., 0.]])
        other, weights = predictable_residuals(target @ rotation, second @ rotation, first @ rotation)
        torch.testing.assert_close(other, residual @ rotation)
        torch.testing.assert_close(weights, reliability)
