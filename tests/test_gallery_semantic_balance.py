import unittest

import torch
from torch.nn import functional as F

from tf_sbir.contextual_rank import column_stochastic
from tf_sbir.gallery_semantic_balance import edge_weights, query_evidence
from tf_sbir.reflection_transfer import matched_controls, reflection_block, select_view_risk
from tf_sbir.union_density import pack_affinity


class GallerySemanticBalanceTests(unittest.TestCase):
    def test_equal_weight_keeps_original_graph_and_zero_removes_language_edges(self):
        ids = torch.arange(4)
        graphs = tuple((ids, ids.roll(i + 1), torch.ones(4)) for i in range(3))
        self.assertIs(edge_weights(graphs, 2), graphs)
        for sixths in (0, 1, 3, 4):
            edges = edge_weights(graphs, sixths)
            torch.testing.assert_close(sum(e[2] for e in edges), torch.full((4,), 3.))
            torch.testing.assert_close(edges[1][2], torch.full((4,), sixths / 2.))
        zero_graph, _ = column_stochastic(4, edge_weights(graphs, 0))
        image_graph, _ = column_stochastic(4, (graphs[0], graphs[2]))
        torch.testing.assert_close(zero_graph.to_dense(), image_graph.to_dense())

    def test_separate_and_joint_query_votes_reproduce_the_two_reference_configurations(self):
        rng = torch.Generator().manual_seed(421)
        n, dim = 1300, 12
        def features(*shape):
            return F.normalize(torch.randn(*shape, generator=rng), dim=-1)
        ids = torch.arange(n)
        edges = [(ids, ids.roll(i + 1), torch.rand(n, generator=rng)) for i in range(3)]
        banks = tuple(torch.randint(n, (n, 10), generator=rng).sort(dim=1).values for _ in range(2))
        index = (features(n, dim), features(n, dim), banks, pack_affinity(n, edges), column_stochastic(n, edges))
        mapped, words, risk = features(2, 8, dim), features(2, 8, dim), torch.rand(2, 8, 2, generator=rng) + .1
        tail, heads = query_evidence(mapped, risk, words, index)
        for option, reference in (("text35", "risk4"), ("sharp", "sharp-source")):
            config = matched_controls()[reference]
            compare, ranks = reflection_block(mapped, select_view_risk(risk, config), words, index, config)
            torch.testing.assert_close(tail, compare)
            torch.testing.assert_close(heads[option], ranks)
