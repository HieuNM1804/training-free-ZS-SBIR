# Negative association and gallery diffusion

Test whether evidence that two photos express different concepts can protect an independent cross-domain query from erroneous propagation. For each photo pair use four disjoint panels of 1,024 gallery photos. Ignore panels containing either endpoint. Require disjoint top-eight public-word readings and negative recognition covariance on every remaining panel, both within one reference photograph and over its eight visual neighbors in both directions. Remove the selected transitions from the local and dense RGB diffusion matrices, renormalize, and propagate the same two-view query evidence as the baseline. A row losing all its effective mass uses the original transition. No query pooling or gradient is involved.

| Row | Ext-1 mAP@all | P@100 | Ext-2 mAP@200 | P@200 |
|---|---:|---:|---:|---:|
| Frozen two-view reference | 85.29 | 84.65 | 77.55 | 75.45 |
| Exclude pairs | 85.36 | 84.67 | 77.48 | 75.43 |
| Attenuate an equal amount of mass per row and affinity bin | 85.34 | 84.67 | 77.50 | 75.42 |
| Exclude only from local transitions | 85.30 | 84.65 | 77.54 | 75.45 |
| Exclude only from dense transitions | 85.36 | 84.67 | 77.49 | 75.43 |
| Exclude all pairs with disjoint top-eight words | 85.21 | 84.66 | 77.82 | 75.43 |

The initial scan checked one query in every batch against KD-SBIR and scored every query. The baseline, full exclusion and disjoint-words control were then replayed query by query using the original KD-SBIR metric function (pinned source checksum). The two JSON files retain both results. Percentages before rounding are available there. The class-bootstrap intervals for full exclusion versus the equal-mass control include zero on both splits. TU-Berlin was not used for selection or evaluated in this experiment.

Over all vetoed pairs, 99.96% and 99.94% join different classes. For local transitions, weighting by the actual amount transmitted lowers this to 77.33% and 78.82%. The mask removes only 10.07% and 7.82% of the baseline local mass flowing between classes. Among dense transitions it captures 61.40% and 48.44%. These are observed on the two Sketchy splits and concern the kernels before query evidence is diffused. They do not isolate the complete cause of a given query's rank error. Labels were used post-ranking to assign these categories.

The statistical motivation comes from measuring lexical association against an independence expectation, using signed covariance here. The inherited propagation is the single-view specialization of Wang et al. Both are separate from the proposed automatic discovery and cross-panel validation of negative edges. Prior work on constrained clustering already formalized explicit pairwise exclusions. [Church and Hanks, ACL 1989](https://aclanthology.org/P89-1010/) · [Wang et al., CVPR 2012](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/cvpr12.pdf) · [Wang and Davidson, KDD 2010](https://www.cs.ucdavis.edu/~davidson/Publications/rp058d-wang.pdf)

Run on Colab L4 using `python -m tf_sbir.exclusion_diffusion` at `7919c6d` for the original four controls and `python -m tf_sbir.exclusion_diffusion --official --audit` at `520bd92` for the post-ranking mass attribution. The short-lived label-oracle diagnostic from an intermediate revision was interrupted and did not supply any retrieval metric reported here.
