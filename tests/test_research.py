import unittest

import torch
from torch.nn import functional as F

from tf_sbir.manifold import build_graph, fused_features
from tf_sbir.semantic import gallery_memory, top_codes


class GalleryManifoldTests(unittest.TestCase):
    def test_transition_has_no_self_edges_and_preserves_constants(self):
        examples = F.normalize(torch.tensor([[1., 0.], [1., .5], [.1, 1.], [-1., .1]]), dim=-1)
        graph = build_graph(examples, examples, k=2, shape_weight=.2, reciprocal=True)
        self.assertFalse(graph.indices.eq(torch.arange(len(examples)).unsqueeze(1)).any())
        torch.testing.assert_close(graph.weights.sum(dim=1), torch.ones(len(examples)))
        constants = torch.ones((len(examples), 3))
        torch.testing.assert_close(graph.diffuse(constants, restart=.1, steps=4), constants)

    def test_concatenated_features_equal_weighted_cosine(self):
        generator = torch.Generator().manual_seed(5)
        a = F.normalize(torch.randn(3, 4, generator=generator))
        b = F.normalize(torch.randn(6, 4, generator=generator))
        c = F.normalize(torch.randn(3, 8, generator=generator))
        d = F.normalize(torch.randn(6, 8, generator=generator))
        q, p = fused_features(a, b, c, d, 0.25)
        torch.testing.assert_close(q @ p.T, 0.75 * (a @ b.T) + 0.25 * (c @ d.T))


class PublicVocabularyTests(unittest.TestCase):
    def test_gallery_memory_is_normalized_and_tracks_support(self):
        generator = torch.Generator().manual_seed(8)
        p = F.normalize(torch.randn(45, 16, generator=generator))
        terms = F.normalize(torch.randn(30, 16, generator=generator))
        memory = gallery_memory(p, terms)
        torch.testing.assert_close(memory.gallery_words.norm(dim=-1), torch.ones(len(p)))
        torch.testing.assert_close(memory.centroids.norm(dim=-1), torch.ones(len(terms)))
        torch.testing.assert_close(memory.support.sum(), torch.tensor(float(len(p))))
        _, weights = top_codes(p @ terms.T, words_per_sample=12)
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(len(p)))


if __name__ == "__main__":
    unittest.main()
