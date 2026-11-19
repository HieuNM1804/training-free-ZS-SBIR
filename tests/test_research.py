import unittest

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

from tf_sbir.adaptive_fusion import curve_reliability
from tf_sbir.anchor_context import gallery_kl_parameters, negative_gallery_kl
from tf_sbir.asif import sparse_relative_features
from tf_sbir.bimodal_graph import bimodal_transition
from tf_sbir.cluster_mine import majority_concepts
from tf_sbir.cluster_prior import cache_votes, cluster_first_rows, cluster_mass, first_neighbor_groups
from tf_sbir.contextual_rank import column_stochastic, jaccard_seed, personalized_walk
from tf_sbir.contour_render import xdog
from tf_sbir.coregistration import leading_basis
from tf_sbir.data import normal_transform
from tf_sbir.dense_views import tokencut_mask
from tf_sbir.gallery_diffusion import mutual_transition, regional_cosines, solve_diffusion
from tf_sbir.isoclip import output_space_maps
from tf_sbir.isolated_regions import isolated_views, token_boxes
from tf_sbir.manifold import build_graph, fused_features, propagate_scores
from tf_sbir.manifold_benchmark import covariance_roots
from tf_sbir.mta import normalized_crops, pixel_lookup, robust_modes
from tf_sbir.multiscale import merge_tile_patches, split_tiles
from tf_sbir.relevance import refine_relevance
from tf_sbir.score_calibration import dynamic_activation, normalized_score_rows, sinkhorn_target_hubness
from tf_sbir.semantic import gallery_memory, guided_term_scores, neighborhood_term_posteriors, top_codes
from tf_sbir.semantic_transport import conditional_similarity, dustbin_plan, maximal_marginal_relevance, partial_similarity
from tf_sbir.signed_feedback import feedback_vectors
from tf_sbir.union_density import candidate_union, density_rrf_rows, grow_density, pack_affinity


class GalleryManifoldTests(unittest.TestCase):
    def test_cluster_cache_votes_average_tip_adapter_affinities(self):
        cosine = torch.tensor([[1., .5, 0.], [.2, .8, .4]])
        assignment = torch.tensor([0, 0, 1])
        actual = cache_votes(cosine, assignment, torch.tensor([2, 1]), 3.)
        affinities = torch.exp(3. * (cosine - 1))
        expected = torch.stack((affinities[:, :2].mean(1), affinities[:, 2]), dim=1)
        torch.testing.assert_close(actual, expected)

    def test_finch_shared_neighbors_and_cluster_prior_conserve_mass(self):
        assignments = first_neighbor_groups(torch.tensor([1, 2, 1, 4, 3, 4]))
        self.assertEqual(assignments.tolist(), [0, 0, 0, 1, 1, 1])
        probability = torch.tensor([[.15, .5, .1, .1, .1, .05], [.1, .15, 0., .4, .25, .1]])
        expected = torch.tensor([[.25, .25, .25, 1/12, 1/12, 1/12], [1/12, 1/12, 1/12, .25, .25, .25]])
        actual = cluster_mass(probability, assignments, torch.tensor([3, 3]))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(1), probability.sum(1))

    def test_cluster_first_preserves_evidence_order_within_each_group(self):
        evidence = torch.tensor([[1., 2., 4., 3.]])
        assignment = torch.tensor([0, 1, 0, 1])
        row, = cluster_first_rows(evidence, assignment, torch.tensor([2, 2]), 1)
        self.assertEqual(row.argsort(descending=True).tolist(), [2, 0, 3, 1])

    def test_rocchio_feedback_uses_complementary_weights_within_each_shortlist(self):
        evidence = torch.tensor([[2., 0., 1.], [0., 3., 1.]])
        shortlist = torch.tensor([[0, 1], [1, 2]])
        gallery = torch.eye(3)
        positive, negative = feedback_vectors(evidence, shortlist, gallery, .5)
        weights = torch.tensor([4., 0.]).softmax(0)
        torch.testing.assert_close(positive, torch.stack((torch.cat((weights, torch.zeros(1))), torch.cat((torch.zeros(1), weights)))))
        torch.testing.assert_close(positive + negative, gallery[shortlist].sum(1))
        solo = feedback_vectors(evidence[1:], shortlist[1:], gallery, .5)
        torch.testing.assert_close(solo[0], positive[1:])

    def test_density_rrf_keeps_unselected_photos_in_initial_order(self):
        initial = 1 / (60 + torch.arange(1, 17101, dtype=torch.float32))[None, :]
        selected = torch.tensor([[1, 3, 0]])
        result = next(density_rrf_rows(initial, selected, 1, 1.))
        evaluated = ((result + 1) * .5).clamp(min=torch.finfo(result.dtype).eps, max=1.)
        order = evaluated.argsort(descending=True)
        self.assertEqual(order[:4].tolist(), [1, 0, 3, 2])
        self.assertTrue(torch.equal(order[4:], torch.arange(4, 17100)))

    def test_binary_relevance_matches_lame_softmax_and_keeps_queries_independent(self):
        graph = torch.tensor([[0., .55, 0.], [.55, 0., .3], [0., .3, 0.]])
        prior = torch.tensor([[-.5, .6, -.2], [.8, -.2, .9]])
        unary = torch.stack((torch.zeros_like(prior), prior), dim=-1).log_softmax(dim=-1)
        for squared in (False, True):
            transition = graph @ graph if squared else graph
            posterior = unary.exp()
            for _ in range(8):
                posterior = (unary + 1.3 * (transition @ posterior)).softmax(dim=-1)
            logits = refine_relevance(prior, graph.to_sparse_csr(), 1.3, squared=squared)
            torch.testing.assert_close(logits.sigmoid(), posterior[:, :, 1])
            solo = refine_relevance(prior[:1], graph.to_sparse_csr(), 1.3, squared=squared)
            torch.testing.assert_close(logits[:1], solo)

    def test_union_density_grows_by_marginal_edge_weight_without_duplicates(self):
        first = torch.tensor([[.8, .6, .2, .1]])
        second = torch.tensor([[.6, .1, .8, .2]])
        candidates, valid = candidate_union((first, second), 2)
        self.assertEqual(candidates[valid].tolist(), [0, 1, 2])
        source = torch.tensor([0, 2, 0, 3])
        target = torch.tensor([2, 0, 3, 0])
        weights = torch.tensor([.8, .8, 5., 5.])
        graph = pack_affinity(4, ((source, target, weights),))
        self.assertEqual(grow_density(first, first, candidates, valid, graph, 3).tolist(), [[0, 2, 1]])

    def test_contextual_rank_uses_jaccard_and_preserves_probability(self):
        scores = torch.tensor([[.6, .4, .2]])
        neighbors = torch.tensor([[1, 2], [0, 2], [0, 1]])
        torch.testing.assert_close(jaccard_seed(scores, neighbors), torch.tensor([[1/3, 1/3, 0.]]))
        a, b, edge = torch.tensor([0, 1]), torch.tensor([1, 0]), torch.tensor([.4, .4])
        transition, dangling = column_stochastic(3, ((a, b, edge),))
        seeds = torch.eye(3)[[0, 2]]
        stationary = personalized_walk(transition, dangling, seeds, .8, iterations=80)
        torch.testing.assert_close(stationary, torch.tensor([[1/1.8, .8/1.8, 0.], [0., 0., 1.]]))

    def test_coregularized_subspace_matches_dense_eigensolver(self):
        transition = torch.tensor([[0., .3, .2], [.3, 0., .1], [.2, .1, 0.]])
        partner = F.normalize(torch.tensor([[.3], [.2], [.9]]), dim=0)
        modified = transition + .25 * (partner @ partner.T)
        _, expected = torch.linalg.eigh(modified)
        actual = leading_basis(transition.to_sparse_csr(), torch.eye(3), (partner,), .25)
        torch.testing.assert_close(actual[:, 0].abs(), expected[:, -1].abs())

    def test_photo_text_graph_only_connects_text_to_photos(self):
        photos = F.normalize(torch.tensor([[1., .2], [.2, 1.], [1., 1.]]))
        transition, active = bimodal_transition(photos, torch.eye(2), 1, 1, 3)
        dense = transition.to_dense()
        self.assertEqual(active.tolist(), [0, 1])
        self.assertEqual(float(dense[3:, 3:].abs().sum()), 0.)
        self.assertTrue(torch.all(dense[:3, 3:].sum(1) > 0))
        torch.testing.assert_close(dense, dense.T)

    def test_mutual_diffusion_matches_dense_solution_and_query_independence(self):
        features = F.normalize(torch.tensor([[1., 0.], [1., .2], [1., .4], [0., 1.]]))
        transition = mutual_transition(features, 2)
        dense = transition.to_dense()
        torch.testing.assert_close(dense, dense.T)
        self.assertEqual(float(dense.diag().abs().sum()), 0.)
        seeds = torch.eye(4)[:2]
        targets = torch.linalg.solve(torch.eye(4) - .7 * dense, .3 * seeds.T).T
        result = solve_diffusion(transition, seeds, .7, iterations=60)
        torch.testing.assert_close(result, targets)
        torch.testing.assert_close(result[:1], solve_diffusion(transition, seeds[:1], .7, iterations=60))

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

    def test_coral_aligns_covariances_without_regularization(self):
        source = torch.tensor([[3., 0.], [-3., 0.], [0., 1.], [0., -1.]])
        target = source @ torch.tensor([[0.5, 0.3], [0.1, 2.]])
        transform = covariance_roots(source, target, regularization=0)
        torch.testing.assert_close((source @ transform).T.cov(), target.T.cov())


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
    def test_isolated_views_preserve_pixels_inside_the_proposal(self):
        mask = torch.tensor([[False, True, True, False, True, False, False, False, False]])
        self.assertEqual(token_boxes(mask).tolist(), [[1, 0, 3, 2]])
        pixels = np.full((224, 224, 3), 255, dtype=np.uint8)
        pixels[64:160, 64:160] = 0
        cropped, blurred = isolated_views(Image.fromarray(pixels), (64, 64, 160, 160))
        self.assertTrue(torch.all(cropped == 0))
        self.assertTrue(torch.all(blurred[:, 64:160, 64:160] == 0))
        self.assertLess(blurred[0, 63, 112].item(), 255)

    def test_isoclip_output_map_matches_projection_before_clip_head(self):
        rng = torch.Generator().manual_seed(31)
        raw = torch.randn(5, 4, generator=rng)
        visual = torch.randn(4, 3, generator=rng)
        text = torch.randn(3, 3, generator=rng)
        vectors, _, right = torch.linalg.svd(visual @ text.T, full_matrices=False)
        middle = vectors[:, 1:2]
        expected = raw @ (middle @ middle.T) @ visual
        actual = raw @ visual @ output_space_maps(visual, text, ((1, 1),))[(1, 1)]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        lexical = raw[:, :3]
        projection = right[1:2].T @ right[1:2]
        expected_text = lexical @ projection @ text
        image_map, text_map = output_space_maps(visual, text, ((1, 1),), paired=True)[(1, 1)]
        torch.testing.assert_close(raw @ visual @ image_map, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(lexical @ text @ text_map, expected_text, rtol=1e-5, atol=1e-5)

    def test_anchor_scores_equal_the_negative_direct_kl_divergence(self):
        p = torch.tensor([[.4, .6], [.75, .25]])
        q = torch.tensor([[.3, .7], [.9, .1]])
        expected = -(p[None, :] * (p[None, :].log() - q[:, None].log())).sum(dim=2)
        score = negative_gallery_kl(q.log(), *gallery_kl_parameters(p.log()))
        torch.testing.assert_close(score, expected)

    def test_xdog_turns_a_flat_surface_white_and_retains_a_step_edge(self):
        levels = np.full((31, 31), 0.2, dtype=np.float32)
        self.assertTrue(np.all(xdog(levels, 0.8) == 255))
        levels[:, 15:] = 0.8
        rendered = xdog(levels, 0.8)
        self.assertLess(rendered[:, 13:17].min(), 128)
        self.assertTrue(np.all(rendered[:, :8] == 255))

    def test_multiscale_tile_order_recovers_the_correct_spatial_grid(self):
        pixels = torch.arange(72, dtype=torch.float32).view(2, 1, 6, 6)
        tiles = split_tiles(pixels, 3).flatten(2).transpose(1, 2)
        expected = F.avg_pool2d(pixels, 2).flatten(2).transpose(1, 2)
        torch.testing.assert_close(merge_tile_patches(tiles), expected)

    def test_curve_weight_prefers_a_sharp_head_after_reference_subtraction(self):
        fractions = torch.linspace(0, 1, 5)
        tail = 1 - fractions
        base = tail.repeat(10, 1)
        scores = torch.stack((tail + torch.tensor([1., 0., 0., 0., 0.]),
                              tail + torch.tensor([1., .8, .6, .4, .2])))
        strengths = curve_reliability(scores, base, fractions, 2)
        self.assertGreater(strengths[0], strengths[1])
        torch.testing.assert_close(strengths[0], curve_reliability(scores[:1], base, fractions, 2)[0])

    def test_crop_uint8_lookup_matches_reference_normalization(self):
        pixels = np.arange(224 * 224 * 3, dtype=np.int32).astype(np.uint8).reshape(224, 224, 3)
        image = Image.fromarray(pixels)
        expected = normal_transform()(image).unsqueeze(0)
        encoded = pil_to_tensor(image).unsqueeze(0)
        cases = [(torch.device("cpu"), torch.float32)]
        if torch.cuda.is_available():
            cases.append((torch.device("cuda"), torch.float16))
        for device, dtype in cases:
            output = normalized_crops(encoded, pixel_lookup(device, dtype)).cpu()
            self.assertTrue(torch.equal(output, expected.to(dtype=dtype)))

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

    def test_full_gallery_transport_matches_explicit_pairs(self):
        rng = torch.Generator().manual_seed(4)
        concepts = F.normalize(torch.randn(7, 5, 25, generator=rng), dim=-1)
        gallery = F.normalize(torch.randn(9, 3, 25, generator=rng), dim=-1)
        affinities = torch.matmul(concepts[:, None], gallery.mT)
        torch.testing.assert_close(regional_cosines(concepts, gallery, 2), conditional_similarity(affinities))

    def test_mmr_picks_a_distinct_second_concept(self):
        relevance = torch.tensor([[.7, .69, .68]])
        descriptors = torch.tensor([[[1., 0.], [1., 0.], [0., 1.]]])
        plain = maximal_marginal_relevance(relevance, descriptors, 2, 1.0)
        diverse = maximal_marginal_relevance(relevance, descriptors, 2, 0.95)
        self.assertEqual(plain.tolist(), [[0, 1]])
        self.assertEqual(diverse.tolist(), [[0, 2]])

    def test_partial_assignment_balances_marginals_and_rejects_noisy_views(self):
        similar = torch.tensor([[.65, .12, .04], [.1, .68, .03]])
        plan = dustbin_plan(similar, .35, iterations=80)
        torch.testing.assert_close(plan.sum(dim=-1), torch.tensor([1., 1., 3.]))
        torch.testing.assert_close(plan.sum(dim=-2), torch.tensor([1., 1., 1., 2.]))
        self.assertGreater(plan[-1, 2], .9)
        self.assertGreater(partial_similarity(similar, .35), partial_similarity(similar - .3, .35))

    def test_tokencut_selects_the_connected_salient_partition(self):
        patches = torch.tensor([[[1., 0.]] * 2 + [[0., 1.]] * 7])
        expected = torch.tensor([[True, True, False, False, False, False, False, False, False]])
        self.assertTrue(torch.equal(tokencut_mask(patches, .4), expected))

    def test_asif_retains_only_top_anchor_coordinates(self):
        anchors = torch.eye(3)
        sample = F.normalize(torch.tensor([[2., 1., 0.5]]))
        representation = sparse_relative_features(sample, anchors, 2, 2)
        torch.testing.assert_close(representation, F.normalize(torch.tensor([[4., 1., 0.]])))


if __name__ == "__main__":
    unittest.main()
