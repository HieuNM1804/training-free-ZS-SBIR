import unittest

import torch
from torch.nn import functional as F

from tf_sbir.paired_neighborhood_scale import scale_index
from tf_sbir.structure_benchmark import rank_structure
from tf_sbir.universal_structure import layout, self_descriptor


class StructureTests(unittest.TestCase):
    def test_descriptor_is_invariant_to_an_orthogonal_channel_map_and_per_image_offset(self):
        rng = torch.Generator().manual_seed(744)
        x = torch.randn(3, 49, 15, generator=rng)
        rot, _ = torch.linalg.qr(torch.randn(15, 15, generator=rng))
        bias = torch.randn(3, 1, 15, generator=rng)
        bins = layout(7, x.device)
        a = self_descriptor(x, bins)
        b = self_descriptor(1.7 * (x @ rot) + bias, bins)
        torch.testing.assert_close(a, b, atol=4e-6, rtol=2e-5)
        torch.testing.assert_close(a[:1], self_descriptor(x[:1], bins))
        self.assertTrue(torch.isfinite(self_descriptor(torch.ones_like(x), bins)).all())

    def test_structure_rank_does_not_use_another_query(self):
        rng = torch.Generator().manual_seed(745)

        def unit(*size):
            return F.normalize(torch.randn(*size, generator=rng), dim=-1)

        photos, target, geom = unit(39, 12), unit(39, 12), unit(39, 19)
        q, qs = unit(4, 4, 12), unit(4, 4, 19)
        index = scale_index(photos, target, geom, 6)
        tail, heads = rank_structure(q, qs, index, ["w030", "adaptive"])
        one, h = rank_structure(q[2:3], qs[2:3], index, ["w030", "adaptive"])
        torch.testing.assert_close(one, tail[2:3])
        for m in h:
            torch.testing.assert_close(h[m], heads[m][2:3])
