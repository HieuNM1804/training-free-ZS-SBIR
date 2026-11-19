import unittest

import torch
from torch.nn import functional as F

from tf_sbir.contextual_rank import column_stochastic
from tf_sbir.reciprocal_transfer import ReverseSetting, directional_votes, reciprocal_rows, reverse_fraction, reverse_readout
from tf_sbir.transfer_benchmark import ranked_rows
from tf_sbir.union_density import pack_affinity


class ReciprocalTransferTests(unittest.TestCase):
    def test_reverse_projection_never_uses_the_candidate_itself(self):
        target = torch.tensor([[1., 0.], [.8, .6], [-1., 0.]])
        source = torch.eye(3)
        torch.testing.assert_close(reverse_readout(source, target, 1), source[[1, 0, 1]])

    def test_direction_weights_are_query_local_and_handle_empty_evidence(self):
        f = torch.tensor([[[.1, .3, 0.]], [[0., 0., 0.]]])
        r = torch.tensor([[[0., .3, .1]], [[0., 0., 0.]]])
        c = ReverseSetting("predicted", .25, 2., True)
        a = reverse_fraction(f, r, c)
        torch.testing.assert_close(a, torch.full_like(a, .25))
        torch.testing.assert_close(directional_votes(f, r, a, 1.), torch.minimum(f, r))
        torch.testing.assert_close(directional_votes(f, r, a, .25), directional_votes(r, f, 1 - a, .25))

    def test_forward_control_reconstructs_the_original_rank_and_respects_independent_queries(self):
        rng = torch.Generator().manual_seed(46)
        gallery, vocabulary, reverse = [F.normalize(torch.randn(1300, 13, generator=rng), dim=-1) for _ in range(3)]
        q, mapped, words = [F.normalize(torch.randn(3, 5, 13, generator=rng), dim=-1) for _ in range(3)]
        risk = torch.rand(3, 5, generator=rng) + .1
        ids = torch.arange(1300)
        bank = torch.stack([(ids + j) % len(ids) for j in (1, 3, 7, 8)], dim=1)
        edges = [(ids, bank[:, 0], torch.ones(len(ids)))]
        index = (gallery, vocabulary, (bank, bank.flip(1)), pack_affinity(len(ids), edges), column_stochastic(len(ids), edges))
        forward = ReverseSetting("forward", 0., 0.)
        scores = torch.stack(list(reciprocal_rows(q, mapped, risk, words, index, None, None, forward, 2)))
        original = torch.stack(list(ranked_rows(mapped, risk, words, index, "channel-p0-l0.35", 2)))
        torch.testing.assert_close(scores, original, rtol=1e-5, atol=2e-6)
        cfg = ReverseSetting("predicted", .5, 4., True, .25)
        scores = torch.stack(list(reciprocal_rows(q, mapped, risk, words, index, reverse, bank, cfg, 2)))
        single = torch.stack(list(reciprocal_rows(q[:1], mapped[:1], risk[:1], words[:1], index, reverse, bank, cfg, 1)))
        torch.testing.assert_close(scores[:1], single, rtol=1e-5, atol=2e-6)
