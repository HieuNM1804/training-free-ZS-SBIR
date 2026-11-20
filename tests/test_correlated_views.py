import itertools
import unittest

import torch
from torch.nn import functional as F

from tf_sbir.correlated_views import as_correlation, correlated_weights, error_directions, minimum_error_weights
from tf_sbir.multiview_anchor_transfer import predict_views
from tf_sbir.transfer_reconstruction import nearest_anchors


class CorrelatedViewTests(unittest.TestCase):
    def test_diagonal_and_boundary_optima(self):
        pair = torch.tensor([[[1., 0., 0.], [0., 2., 0.], [0., 0., 4.]],
                             [[1., 1.2, 0.], [1.2, 2., 0.], [0., 0., 1.]]])
        start = torch.ones(2, 3) / 3
        actual = minimum_error_weights(pair, start)
        torch.testing.assert_close(actual, torch.tensor([[4/7, 2/7, 1/7], [.5, 0., .5]]))

    def test_active_set_matches_an_exhaustive_independent_solver(self):
        rng = torch.Generator().manual_seed(947)
        a = torch.randn(7, 5, 3, dtype=torch.float64, generator=rng)
        h = a @ a.mT + .025 * torch.eye(5, dtype=a.dtype)
        initial = torch.rand(7, 5, dtype=a.dtype, generator=rng)
        weights = minimum_error_weights(h, initial)
        for gram, w in zip(h, weights, strict=True):
            candidates = []
            for n in range(1, 6):
                for indices in itertools.combinations(range(5), n):
                    ids = torch.tensor(indices)
                    z = torch.linalg.solve(gram[ids][:, ids], torch.ones(n, dtype=a.dtype))
                    z = z / z.sum()
                    if z.min() >= -1e-12:
                        trial = torch.zeros_like(w)
                        trial[ids] = z
                        candidates.append((float(trial @ gram @ trial), trial))
            expected = min(candidates, key=lambda item: item[0])[1]
            torch.testing.assert_close(w, expected, atol=1e-9, rtol=1e-8)
        torch.testing.assert_close(weights[:1], minimum_error_weights(h[:1], initial[:1]))

    def test_diagonal_limit_and_view_permutation(self):
        g = torch.Generator().manual_seed(16)
        x = torch.randn(3, 8, 6, generator=g)
        c = as_correlation(x @ x.mT)
        base = torch.randn(3, 8, generator=g).softmax(dim=1)
        torch.testing.assert_close(correlated_weights(base, c, 0.), base, rtol=0., atol=0.)
        torch.testing.assert_close(correlated_weights(base, torch.eye(8).expand_as(c), .9), base)
        reorder = torch.randperm(8, generator=g)
        torch.testing.assert_close(correlated_weights(base, c, .75)[:, reorder],
                                   correlated_weights(base[:, reorder], c[:, reorder][:, :, reorder], .75))

    def test_error_gram_diagonal_recovers_the_original_transfer_risks(self):
        g = torch.Generator().manual_seed(5)
        source, target, error = [F.normalize(torch.randn(80, 8, generator=g), dim=-1) for _ in range(3)]
        q = F.normalize(torch.randn(3, 8, 8, generator=g), dim=-1)
        ids = nearest_anchors(q.flatten(0, 1), source, 64)
        predicted, original = predict_views(q, source, target, error, ids, count=64, penalty=1., carry=1., risk_components=True)
        actual, u, e, sr, tr = error_directions(q, source, target, error)
        torch.testing.assert_close(actual, predicted)
        torch.testing.assert_close(original, torch.stack((sr, tr), dim=-1))
        torch.testing.assert_close(u.square().sum(dim=-1), sr)
        torch.testing.assert_close(e.square().sum(dim=-1), tr)
