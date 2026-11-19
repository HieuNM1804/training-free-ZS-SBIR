import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from tf_sbir.contextual_rank import column_stochastic
from tf_sbir.sketch_canvas_fusion import CanvasSketches
from tf_sbir.transfer_benchmark import AnchorSketchViews, ranked_rows
from tf_sbir.transfer_reconstruction import transferred_ranking
from tf_sbir.union_density import pack_affinity
from tf_sbir.view_hypotheses import RegionalSketches


class TransferStreamingTests(unittest.TestCase):
    def test_compact_loader_matches_reference_crop_and_canvases_pixel_for_pixel(self):
        with TemporaryDirectory() as dest:
            path = Path(dest) / "sketch.png"
            drawing = Image.new("RGB", (117, 83), "white")
            ImageDraw.Draw(drawing).line([(9, 36), (47, 9), (98, 71)], fill="black", width=3)
            drawing.save(path)
            combined = AnchorSketchViews([path])[0]
            torch.testing.assert_close(combined[0], RegionalSketches([path])[0][0])
            torch.testing.assert_close(combined[1:], CanvasSketches([path])[0])

    def test_bounded_scores_match_reference_and_queries_can_be_scored_independently(self):
        rng = torch.Generator().manual_seed(43)
        size = 1300
        gallery, vocabulary = [F.normalize(torch.randn(size, 23, generator=rng), dim=-1) for _ in range(2)]
        ids = torch.arange(size)
        bank = torch.stack([(ids + offset) % size for offset in (1, 2, 4, 5)], dim=1)
        banks = (bank, bank.flip(1))
        edges = [(ids, bank[:, 0], torch.ones(size))]
        graph = pack_affinity(size, edges)
        walk = column_stochastic(size, edges)
        index = (gallery, vocabulary, banks, graph, walk)
        vectors, concepts = [F.normalize(torch.randn(3, 5, 23, generator=rng), dim=-1) for _ in range(2)]
        risks = torch.rand(3, 5, generator=rng) + .1
        all_at_once = transferred_ranking(torch.cat((vectors[:, 0], vectors[:, 1])), gallery,
                                          tuple(concepts[:, i] @ vocabulary.T for i in (0, 1)), banks, graph, walk, 3)
        reference = torch.stack(list(all_at_once()))
        actual = torch.stack(list(ranked_rows(vectors, risks, concepts, index, "two", 2)))
        torch.testing.assert_close(actual, reference, atol=2e-6, rtol=1e-5)
        batched = torch.stack(list(ranked_rows(vectors, risks, concepts, index, "risk4", 2)))
        alone = torch.stack(list(ranked_rows(vectors[:1], risks[:1], concepts[:1], index, "risk4", 1)))
        torch.testing.assert_close(batched[:1], alone, atol=2e-6, rtol=1e-5)
