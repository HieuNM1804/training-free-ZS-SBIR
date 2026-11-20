import unittest

import torch
from torch.nn import functional as F

from tf_sbir.contour_query_expansion import pagerank_tail
from tf_sbir.visual_paired_graph import image_index, paired_sum, visual_tail


class VisualPairedGraphTests(unittest.TestCase):
    def test_single_bank_reproduces_the_walk_with_duplicate_identical_evidence(self):
        rng = torch.Generator().manual_seed(314)
        photos, target = [F.normalize(torch.randn(25, 9, generator=rng), dim=1) for _ in range(2)]
        gallery, neighbors, _, transition = image_index(photos, target, 5)
        scores = torch.randn(3, 9, generator=rng) @ gallery.T
        prior, tail = visual_tail(scores, neighbors, transition)
        expected = pagerank_tail(scores, scores, prior, (neighbors, neighbors), *transition, 3)
        torch.testing.assert_close(tail, expected)
        for i in range(len(scores)):
            torch.testing.assert_close(tail[i:i + 1], visual_tail(scores[i:i + 1], neighbors, transition)[1])

    def test_weighted_cosine_is_the_sum_of_views_and_preserves_reflections(self):
        rng = torch.Generator().manual_seed(135)
        query = F.normalize(torch.randn(5, 8, 32, generator=rng), dim=-1)
        gallery = F.normalize(torch.randn(18, 32, generator=rng), dim=1)
        weights = torch.randn(5, 8, generator=rng).softmax(1)
        scores = paired_sum(query, weights) @ gallery.T
        torch.testing.assert_close(scores, ((query @ gallery.T) * weights[:, :, None]).sum(dim=1))
        torch.testing.assert_close(scores, paired_sum(query.roll(4, 1), weights.roll(4, 1)) @ gallery.T)
