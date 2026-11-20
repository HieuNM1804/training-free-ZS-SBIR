import unittest

import torch
from torch.nn import functional as F

from tf_sbir.crossdomain_benchmark import SPLIT_HASHES, digest
from tf_sbir.data import UNSEEN_CLASSES
from tf_sbir.domain_prompt_gallery import (
    PairConfig,
    local_readout,
    prompt_coordinates,
    rank_block,
    source_descriptors,
)
from tf_sbir.paired_neighborhood_scale import scale_index


class DomainPromptTransportTests(unittest.TestCase):
    def test_reconstruction_is_an_identity_with_identical_domains_and_full_residual(self):
        rng = torch.Generator().manual_seed(603)
        x = F.normalize(torch.randn(20, 12, generator=rng), dim=1)
        text = F.normalize(torch.randn(30, 12, generator=rng), dim=1)
        own, moved, residual = prompt_coordinates(x, text, text, 12)
        torch.testing.assert_close(moved, own)
        torch.testing.assert_close(moved + residual, x)

    def test_dual_descriptor_can_have_a_different_dimension_than_the_target(self):
        rng = torch.Generator().manual_seed(604)
        bank = F.normalize(torch.randn(41, 9, generator=rng), dim=1)
        code = F.normalize(torch.randn(41, 5, generator=rng), dim=1)
        q = F.normalize(torch.randn(5, 6, 9, generator=rng), dim=-1)
        words = F.normalize(torch.randn(5, 6, 5, generator=rng), dim=-1)
        source = source_descriptors(bank, code, 0.4)
        queries = source_descriptors(q, words, 0.4)
        pred, risk = local_readout(queries, q, source, bank, bank, torch.zeros_like(bank), PairConfig(support=8))
        self.assertTrue(torch.isfinite(risk).all())
        torch.testing.assert_close(pred, q, atol=2e-6, rtol=1e-5)
        one = local_readout(queries[:1], q[:1], source, bank, bank, torch.zeros_like(bank), PairConfig(support=8))[
            0
        ]
        torch.testing.assert_close(one, pred[:1])

    def test_full_frame_ranking_has_no_query_cohort_dependency(self):
        rng = torch.Generator().manual_seed(605)

        def unit(*shape):
            return F.normalize(torch.randn(*shape, generator=rng), dim=-1)

        photos, visual, words = unit(40, 12), unit(40, 12), unit(40, 12)
        index = scale_index(photos, visual, words, 5)
        x, q = unit(3, 6, 12), unit(3, 6, 12)
        r = torch.rand(3, 6, generator=rng) + 0.1
        tail, heads = rank_block(x, r, q, index, ["separate", "joint"])
        one, single = rank_block(x[:1], r[:1], q[:1], index, ["separate", "joint"])
        torch.testing.assert_close(one, tail[:1])
        for name in heads:
            torch.testing.assert_close(single[name], heads[name][:1])

    def test_all_reported_splits_use_the_original_kd_class_names_and_order(self):
        for split, value in SPLIT_HASHES.items():
            self.assertEqual(digest(UNSEEN_CLASSES[split]), value)
