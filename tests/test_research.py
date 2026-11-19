import unittest

import torch
from torch.nn import functional as F

from tf_sbir.asif import sparse_relative_features
from tf_sbir.cluster_mine import majority_concepts
from tf_sbir.manifold import build_graph, fused_features, propagate_scores
from tf_sbir.mta import robust_modes
from tf_sbir.score_calibration import dynamic_activation, normalized_score_rows, sinkhorn_target_hubness
from tf_sbir.semantic import gallery_memory, guided_term_scores, neighborhood_term_posteriors, top_codes
from tf_sbir.semantic_transport import conditional_similarity, dustbin_plan, partial_similarity


class GalleryManifoldTests(unittest.TestCase):
    def test_transition_has_no_self_edges_and_preserves_constants(self):
        examples = F.normalize(torch.tensor([[1., 0.], [1., .5], [.1, 1.], [-1., .1]]), dim=-1)
        graph = build_graph(examples, examples, k=2, shape_weight=.2, reciprocal=True)
        self.assertFalse(graph.indices.eq(torch.arange(len(examples)).unsqueeze(1)).any())
        torch.testing.assert_close(graph.weights.sum(dim=1), torch.ones(len(examples)))
        constants = torch.ones((len(examples), 3))
        torch.testing.assert_close(graph.diffuse(constants, restart=.1, steps=4), constants)

    def test_personalized_walk_preserves_probability_and_query_independence(self):
        gallery = F.normalize(torch.randn(9, 4, generator=torch.Generator().manual_seed(8)))
        graph = build_graph(gallery, gallery, k=3)
        scores = gallery[:2] @ gallery.T
        mass = propagate_scores(scores, graph.transition(), seed_count=3)
        torch.testing.assert_close(mass.sum(dim=1), torch.ones(2))
        one = propagate_scores(scores[1:2], graph.transition(), seed_count=3)
        torch.testing.assert_close(mass[1:2], one)

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
        mass = neighborhood_term_posteriors(p[:3] @ p.T, memory, neighbors=12)
        torch.testing.assert_close(mass.sum(dim=1), torch.ones(3))
        _, weights = top_codes(p @ terms.T, words_per_sample=12)
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(len(p)))

    def test_other_queries_do_not_change_lexical_scores(self):
        generator = torch.Generator().manual_seed(8)
        p = F.normalize(torch.randn(45, 16, generator=generator))
        terms = F.normalize(torch.randn(30, 16, generator=generator))
        q = F.normalize(torch.randn(3, 16, generator=generator))
        memory = gallery_memory(p, terms)
        independent = guided_term_scores(q[1:2], p, terms, terms, memory, .3)
        batched = guided_term_scores(q, p, terms, terms, memory, .3)
        torch.testing.assert_close(independent, batched[1:2])

    def test_cluster_vote_ignores_empty_clusters(self):
        group = torch.tensor([0, 0, 0, 2, 2])
        concepts = torch.tensor([2, 2, 1, 3, 3])
        torch.testing.assert_close(majority_concepts(group, concepts, 4, 5), torch.tensor([2, 3]))


class InferenceCalibrationTests(unittest.TestCase):
    def test_dynamic_qb_norm_corrects_only_activated_queries(self):
        activation = dynamic_activation(torch.tensor([[.7, .1], [.6, .4]]), 1)
        query = gallery = torch.eye(2)
        offset = torch.tensor([.2, -.2])
        results = torch.stack(list(normalized_score_rows(query, gallery, offset, 2, activation)))
        torch.testing.assert_close(results[1], query[1])
        torch.testing.assert_close(results[0], torch.tensor([.8, .2]) / 1.2)

    def test_sinkhorn_balances_probe_marginals(self):
        scores = torch.tensor([[.1, .3, .2], [.2, .05, .15], [.1, .15, .1], [.2, .25, .05]])
        temperature = .15
        penalty = sinkhorn_target_hubness(scores, temperature, iterations=40)
        balanced = ((scores - penalty) / temperature).softmax(dim=1)
        torch.testing.assert_close(balanced.mean(dim=0), torch.ones(3) / 3)

    def test_mta_descriptors_depend_on_one_sketch(self):
        rng = torch.Generator().manual_seed(0)
        views = F.normalize(torch.randn(3, 8, 4, generator=rng), dim=-1)
        words = F.normalize(torch.randn(20, 4, generator=rng), dim=-1)
        all_modes = robust_modes(views, words, 100.)
        torch.testing.assert_close(all_modes.norm(dim=-1), torch.ones(len(views)))
        torch.testing.assert_close(robust_modes(views[1:2], words, 100.), all_modes[1:2])

    def test_transport_is_symmetric_and_reduces_to_single_cosine(self):
        similarities = torch.tensor([[[.1, .4, .6], [.3, .6, .1]]])
        result = conditional_similarity(similarities)
        torch.testing.assert_close(result, conditional_similarity(similarities.mT))
        torch.testing.assert_close(conditional_similarity(similarities[:, :1, :1]), torch.tensor([.1]))

    def test_partial_assignment_balances_marginals_and_rejects_noisy_views(self):
        similar = torch.tensor([[.65, .12, .04], [.1, .68, .03]])
        plan = dustbin_plan(similar, .35, iterations=80)
        torch.testing.assert_close(plan.sum(dim=-1), torch.tensor([1., 1., 3.]))
        torch.testing.assert_close(plan.sum(dim=-2), torch.tensor([1., 1., 1., 2.]))
        self.assertGreater(plan[-1, 2], .9)
        self.assertGreater(partial_similarity(similar, .35), partial_similarity(similar - .3, .35))

    def test_asif_retains_only_top_anchor_coordinates(self):
        anchors = torch.eye(3)
        sample = F.normalize(torch.tensor([[2., 1., 0.5]]))
        representation = sparse_relative_features(sample, anchors, 2, 2)
        torch.testing.assert_close(representation, F.normalize(torch.tensor([[4., 1., 0.]])))


if __name__ == "__main__":
    unittest.main()
