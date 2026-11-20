import unittest

import torch

from tf_sbir.gallery_routing import choose_experts, probe_utilities, routed_tensor


class GalleryRoutingTests(unittest.TestCase):
    def test_rank_utility_uses_the_paired_photo_only_for_the_instance_measure(self):
        identity = torch.tensor([600, 605, 650])
        reference = torch.arange(200).repeat(3, 1)
        first = torch.cat((identity[:1], torch.arange(255)))
        second = torch.cat((torch.arange(100), identity[1:2], torch.arange(100, 255)))
        third = torch.arange(300, 556)
        actual = probe_utilities(torch.stack((first, second, third)), identity, reference)
        expected = torch.tensor([[1., 1., 1.], [1., 1., 1/101], [0., 0., 0.]])
        torch.testing.assert_close(actual, expected)

    def test_one_sketch_has_the_same_decision_alone_or_in_a_batch(self):
        rng = torch.Generator().manual_seed(246)
        utility = torch.rand(213, 2, 3, generator=rng)
        bank = torch.randint(213, (213, 128), generator=rng)
        support = torch.randint(213, (5, 2, 128), generator=rng)
        together, params = choose_experts(utility, bank, support)
        alone, single_params = choose_experts(utility, bank, support[1:2])
        self.assertEqual(params, single_params)
        self.assertEqual(len(together), 57)
        for name in together:
            torch.testing.assert_close(together[name][1:2], alone[name])

    def test_selecting_a_score_row_keeps_its_exact_ranking_and_ties(self):
        a = torch.tensor([[3., 1., 1.], [.7, .1, 2.]])
        b = torch.tensor([[1., 3., 1.], [.1, .1, -.2]])
        selected = routed_tensor(a, b, torch.tensor([False, True]))
        self.assertTrue(torch.equal(selected, torch.stack((a[0], b[1]))))
