import unittest

import torch

from tf_sbir.reflection_transfer import SETTINGS, reflection_seed


class ReflectionTransferTests(unittest.TestCase):
    def test_correctly_paired_evidence_is_invariant_to_reversing_the_orientations(self):
        rng = torch.Generator().manual_seed(64)
        votes = torch.rand(4, 8, 31, generator=rng)
        risk = torch.rand(4, 8, generator=rng) + .1
        for mode, config in SETTINGS.items():
            if config.shuffle:
                continue
            current = reflection_seed(votes, risk, config)
            opposite = reflection_seed(votes.roll(4, dims=1), risk.roll(4, dims=1), config)
            torch.testing.assert_close(current, opposite, msg=mode)
            torch.testing.assert_close(current[:1], reflection_seed(votes[:1], risk[:1], config))

    def test_equal_halves_are_unaffected_by_the_intersection_term(self):
        rng = torch.Generator().manual_seed(66)
        half = torch.rand(3, 4, 13, generator=rng)
        risk = torch.rand(3, 8, generator=rng) + .1
        votes = torch.cat((half, half), dim=1)
        torch.testing.assert_close(reflection_seed(votes, risk, SETTINGS["pairs-agree1"]),
                                   reflection_seed(votes, risk, SETTINGS["pairs-intersect05"]))
