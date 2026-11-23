# Does foreground provide information after controlling for noun identity and position?

The [paired-name experiment](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/3b73bfc2ca41502956829846f75686de8c53453e/docs/results/foreground-word-role/README.md) compared two gallery nominees. This experiment tests a stronger claim: a separate, calibrated contribution of the semantic role can improve the whole ranking when applied to every photo expressing the nominated word.

For each nominated noun pair, project gallery photo embeddings and the two-view sketch onto three fixed text contrasts: isolated noun identity, mention position, and counterbalanced foreground role. Unlabeled gallery photos are assigned to a word by their largest frozen CLIP word reading. Fit a two-population, equal-covariance Gaussian model to these automatically defined populations using least squares and pooled within-group variance. The residual of the role contrast after regression on noun and position yields the conditional Gaussian log likelihood ratio. This is an analytic estimate, with no model updates, gradient, or pooling of sketch queries.

Add this evidence to the native log odds between the two groups. Preserve their total retrieval mass and the order within each group; photos outside both groups retain their original scores. We compare identity alone, marginal role, reversed roles of position and foreground in the regression, and random group assignments of the same sizes. Early or late mention is counterbalanced by encoding both descriptions with the same two nouns in both orders.

| 768 queries each, complete native galleries | Ext-1 mAP@all | P@100 | Ext-2 mAP@200 | P@200 |
|---|---:|---:|---:|---:|
| Native words and RGB diffusion | 83.00 | 82.85 | 74.07 | 72.78 |
| Isolated noun likelihood | 80.00 | 82.92 | 73.60 | 71.61 |
| Marginal foreground likelihood | 81.56 | 82.92 | 73.70 | 71.83 |
| Foreground conditional on noun and position | 82.87 | 82.97 | 73.99 | 72.51 |
| Permuted-group conditional likelihood | 83.02 | 82.97 | 73.98 | 72.69 |

The conditional role likelihood corrected 8 / 15 wrong top-1 decisions and damaged 3 / 3 correct ones on Ext-1 / Ext-2. These head corrections do not deliver a consistent mAP gain. The name groups supplied sufficient observations in 711 / 665 queries. The 99th percentile of the absolute conditional log odds was 45.5 / 178.1. A score on this scale can change the ordering of entire groups substantially.

For noun pairs with at least six donor photos per name, group members were split into separate fitting and evaluation sets. Adding the conditional role to noun and position changed equal-group held-photo log loss from 0.1515 to 0.1570 and from 0.1431 to 0.1544. This test measures held-photo role grounding against automatic word identities, not semantic class annotations. Even on that narrower task the extra likelihood did not improve calibration.

Run `python -m tf_sbir.conditional_foreground_evidence` on Colab T4, revision `d5f6fd43ae8876957b16be42357d757fc68a837d`. All reported ranking metrics replay the original KD-SBIR evaluator. The fixed CLIP ViT-B/32 and the public vocabulary are identical to the parent experiment. [Abbasi et al., CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Abbasi_CLIP_Under_the_Microscope_A_Fine-Grained_Analysis_of_Multi-Object_Representation_CVPR_2025_paper.pdf) · [Thrush et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Thrush_Winoground_Probing_Vision_and_Language_Models_for_Visio-Linguistic_Compositionality_CVPR_2022_paper.pdf)
