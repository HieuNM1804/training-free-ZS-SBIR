import unittest

import torch
from torch.nn import functional as F

from tf_sbir.gallery_semantic_balance import compact_union
from tf_sbir.paired_neighborhood_scale import mixed_rows, paired_preservation, radius_weights
from tf_sbir.union_density import candidate_union, grow_density


class PairedNeighborhoodTests(unittest.TestCase):
    def test_calibration_is_invariant_to_relabeling_paired_gallery_and_handles_ties(self):
        rng = torch.Generator().manual_seed(271)
        a, b = [F.normalize(torch.randn(43, 7, generator=rng), dim=1) for _ in range(2)]
        ids = torch.randperm(len(a), generator=rng)
        preserve = paired_preservation(a, b, (3, 8, 13))
        torch.testing.assert_close(preserve[ids], paired_preservation(a[ids], b[ids], (3, 8, 13)))
        torch.testing.assert_close(paired_preservation(a, a, (3, 8, 13)), torch.full((len(a), 3), .5))
        torch.testing.assert_close(preserve.mean(0), torch.full((3,), .5))

    def test_compact_union_keeps_every_greedy_candidate_position_with_exact_score_ties(self):
        rng = torch.Generator().manual_seed(807)
        scores = [torch.randn(5, 41, generator=rng) for _ in range(6)]
        ids, unique = candidate_union(scores, 8)
        narrow, valid = compact_union(ids, unique)
        self.assertLess(narrow.size(1), ids.size(1))
        for i in range(len(ids)):
            torch.testing.assert_close(ids[i][unique[i]], narrow[i][valid[i]])
        graph = (torch.randint(41, (41, 9), generator=rng), torch.rand(41, 9, generator=rng))
        zero = torch.zeros_like(scores[0])
        torch.testing.assert_close(grow_density(zero, zero, ids, unique, graph, 18),
                                   grow_density(zero, zero, narrow, valid, graph, 18))

    def test_mixing_is_query_independent_and_handles_uniform_or_tied_scales(self):
        features = [torch.full((4, 17), float(k)) for k in (1, 3, 8)]
        confidence = torch.tensor([[.1, .2, .9], [.4, .4, .1], [.2, .2, .2], [.7, .1, .3]])
        for t in (0., 8., None):
            weights = radius_weights(confidence, t)
            torch.testing.assert_close(weights.sum(dim=1), torch.ones(4))
            all_rows = torch.stack(list(mixed_rows(features, weights)))
            torch.testing.assert_close(all_rows[1], next(mixed_rows([x[1:2] for x in features], radius_weights(confidence[1:2], t))))
            if t is None:
                torch.testing.assert_close(all_rows[:, 0], torch.tensor([8., 2., 4., 1.]))
