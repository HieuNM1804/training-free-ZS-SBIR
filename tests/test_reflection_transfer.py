import unittest

import torch

from tf_sbir.multiview_anchor_transfer import predict_views
from tf_sbir.reflection_transfer import SETTINGS, matched_controls, reflection_seed, select_view_risk


class ReflectionTransferTests(unittest.TestCase):
    def test_splitting_transfer_risk_preserves_mapping_and_combined_score(self):
        q = torch.tensor([[[.5, .3, .2], [.1, .2, .7]]])
        source = torch.eye(3)
        target = source.roll(1, 1)
        ids = torch.tensor([[0, 1], [2, 1]])
        options = dict(count=2, penalty=1., carry=1.)
        mapped, combined = predict_views(q, source, target, source, ids, **options)
        other, parts = predict_views(q, source, target, source, ids, **options, risk_components=True)
        torch.testing.assert_close(mapped, other)
        torch.testing.assert_close(combined, select_view_risk(parts, matched_controls()["reference"]))
        torch.testing.assert_close(combined, select_view_risk(parts, matched_controls()["source"]) +
                                    select_view_risk(parts, matched_controls()["target"]))

    def test_negative_control_reassigns_risk_within_each_query_and_orientation(self):
        components = torch.arange(32.).reshape(2, 8, 2)
        controls = matched_controls()
        both = select_view_risk(components, controls["reference"])
        shuffled = select_view_risk(components, controls["shuffled"])
        torch.testing.assert_close(shuffled[:, :4], both[:, :4].roll(1, dims=1))
        torch.testing.assert_close(shuffled[:, 4:], both[:, 4:].roll(1, dims=1))

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
