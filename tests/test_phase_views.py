import unittest
import torch

from tf_sbir.phase_views import filters, phase_strength, square_box


class PhaseTests(unittest.TestCase):
    def test_phase_survives_contrast_change_and_polarity_inversion(self):
        x = torch.zeros(2, 64, 64)
        x[:, 12:48, 10:40] = 1
        x[1, 22:31, 10:40] = 0.2
        f, a = filters(64, x.device)
        p = phase_strength(x, f, a)
        torch.testing.assert_close(p, phase_strength(1 - 0.6 * x, f, a), atol=2e-4, rtol=2e-4)
        self.assertGreater(p.amax().item(), 0.15)
        torch.testing.assert_close(p[:1], phase_strength(x[:1], f, a))
        self.assertTrue(torch.isfinite(p).all())

    def test_blank_frames_and_region_coordinates(self):
        blank = torch.zeros(2, 48, 48)
        f, a = filters(48, blank.device)
        self.assertTrue(torch.equal(phase_strength(blank, f, a), blank))
        torch.testing.assert_close(square_box(blank), torch.tensor([[24.0, 24.0, 48.0], [24.0, 24.0, 48.0]]))
        blank[0, 6:16, 10:30] = 1
        torch.testing.assert_close(square_box(blank)[0], torch.tensor([20.0, 11.0, 20.0]))
