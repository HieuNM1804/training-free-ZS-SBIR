import unittest

import torch
from torchmetrics.functional.retrieval import retrieval_average_precision, retrieval_precision

from tf_sbir.data import UNSEEN_CLASSES
from tf_sbir.metrics import cosine_rows, evaluate_rows
from tf_sbir.reranking import RankingConfig, reranked_rows, score_shortlist


class ProtocolTests(unittest.TestCase):
    def test_metrics_follow_torchmetrics_with_score_ties(self):
        rng = torch.Generator().manual_seed(42)
        queries = torch.randn(4, 16, generator=rng)
        photos = torch.randn(273, 16, generator=rng)
        photos[1] = photos[0]
        query_labels = torch.arange(4)
        photo_labels = torch.arange(len(photos)) % 4
        cosine = list(cosine_rows(queries, photos))
        for split, map_k, p_k in (("sketchy_1", 0, 100), ("sketchy_2", 200, 200)):
            with self.subTest(split=split):
                scores = [((x + 1) * 0.5).clamp(min=torch.finfo(x.dtype).eps, max=1.0) for x in cosine]
                positives = [photo_labels.eq(y) for y in query_labels]
                expected_ap = torch.stack([
                    retrieval_average_precision(score, target, top_k=map_k or None)
                    for score, target in zip(scores, positives)
                ]).mean()
                expected_p = torch.stack([
                    retrieval_precision(score, target, top_k=p_k)
                    for score, target in zip(scores, positives)
                ]).mean()
                self.assertEqual(
                    evaluate_rows(iter(cosine), query_labels, photo_labels, split),
                    (expected_ap, expected_p, map_k, p_k),
                )

    def test_split_sizes(self):
        self.assertEqual(len(set(UNSEEN_CLASSES["sketchy_1"])), 25)
        self.assertEqual(len(set(UNSEEN_CLASSES["sketchy_2"])), 21)


class RerankingTests(unittest.TestCase):
    def test_consensus_requires_the_same_photo_patch(self):
        query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        sketch = {
            "primary": query,
            "secondary": query,
            "valid": torch.tensor([[True, False]]),
        }
        gallery = {
            "primary": torch.cat([query, query]),
            "secondary": torch.cat([query.flip([1]), query]),
            "valid": torch.ones((2, 2), dtype=torch.bool),
        }
        scores = score_shortlist(
            sketch, gallery, torch.tensor([[0, 1]]),
            {"patch_independent", "patch_consensus", "patch_stable"},
        )
        self.assertEqual(scores["patch_independent"].tolist(), [[1.0, 1.0]])
        self.assertEqual(scores["patch_consensus"].tolist(), [[0.0, 1.0]])
        self.assertTrue(torch.equal(scores["patch_stable"], scores["patch_consensus"]))

    def test_score_permutation_preserves_the_gallery_tail(self):
        query = torch.tensor([[1.0, 0.0]])
        gallery = torch.tensor([[0.9, 0.0], [0.8, 0.0], [0.7, 0.0]])
        row = next(reranked_rows(
            query, gallery, torch.tensor([[0, 1]]), torch.tensor([[0.0, 1.0]]),
            RankingConfig("patch_ink", 2, 2.0),
        ))
        self.assertTrue(torch.equal(row, torch.tensor([0.8, 0.9, 0.7])))


if __name__ == "__main__":
    unittest.main()
