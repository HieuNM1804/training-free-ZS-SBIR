import unittest

import numpy as np
import torch

from tf_sbir.saliency_transfer import FAMILIES, Setting, grid, refined, view_components
from tf_sbir.saliency_views import field, salient_box, views


class SalientCropTests(unittest.TestCase):
    def test_off_center_object_on_either_background(self):
        for bg, fg in ((225, 20), (20, 225)):
            image = np.full((90, 150, 3), bg, dtype=np.uint8)
            image[23:59, 11:46] = fg
            x, y, w = salient_box(field(image))
            self.assertLess(abs(x - 28.5), 2)
            self.assertLess(abs(y - 41), 2)
            self.assertTrue(31 < w < 44)
            result = views(image)
            self.assertEqual(result.shape, (10, 3, 224, 224))
            self.assertTrue(torch.equal(result[0].flip(-1), result[1]))
            self.assertTrue(torch.equal(255 - result[6], result[8]))
            middle = views(image, at_center=True)
            self.assertEqual(middle.shape, result.shape)
            self.assertFalse(torch.equal(middle[0], result[0]))

    def test_flat_rectangular_input_has_a_finite_symmetric_fallback(self):
        image = np.full((24, 120, 3), [120, 100, 200], np.uint8)
        self.assertEqual(salient_box(field(image)), (60.0, 12.0, 120))
        result = views(image)
        self.assertTrue(torch.equal(result[0], result[2]))
        torch.testing.assert_close(result, views(image, at_center=True))

    def test_view_components_are_independent_across_examples(self):
        g = torch.Generator().manual_seed(13)
        a = torch.nn.functional.normalize(torch.randn(4, 8, generator=g), dim=-1)
        b = torch.randn(4, 10, 8, generator=g)
        for name, n in (("full", 1), ("core", 2), ("all", 7)):
            first = view_components(a, b, name)
            single = view_components(a[2:3], b[2:3], name)
            for x, y in zip(first, single, strict=True):
                self.assertEqual(x.shape, (4, n, 8))
                torch.testing.assert_close(x[2:3], y)

    def test_refinement_is_disjoint_and_does_not_add_a_shuffled_candidate(self):
        initial = grid()
        winners = {f: (Setting(f), "joint") for f in FAMILIES}
        fine = refined(winners)
        self.assertTrue(fine)
        self.assertTrue(set(fine).isdisjoint(initial))
        self.assertTrue(all(c.paired for c, _ in fine.values()))
