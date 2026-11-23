# Why can the same sketch change its preference between the same two photos?

Two native gallery runs share 2,231 sketches and 2,205 photos in four categories. We selected 15,149 adjacent relevant–irrelevant pairs in the top 200 of the direct word ranking among the shared photos. The reference diffuser reverses 1,489 of those comparisons (9.83%) between galleries. Joined observations can repeat sketches and photos. This is a conditional stress test on four overlapping classes. Each retrieval still uses only its own gallery.

## Predicting instability from one gallery

Build each native diffusion graph again on eight seeded random half partitions. For an image pair use draws containing both endpoints, and require at least two such draws. Use the root mean squared change of its margin relative to the absolute full-gallery margin as the instability estimate. The opposite native gallery supplies the actual reversal outcome after scoring. No gallery or sketch from the other split is used to estimate risk.

| Direction | Eligible pairs | Observed reversals | AUROC, small native margin | AUROC, variation divided by native margin | AUROC within class and margin decile, small margin / proposed |
|---|---:|---:|---:|---:|---:|
| Ext-1 predicts Ext-2 | 14,621 | 1,428 | 0.801 | 0.807 | 0.546 / 0.655 |
| Ext-2 predicts Ext-1 | 14,606 | 1,432 | 0.819 | 0.836 | 0.549 / 0.695 |

The within-stratum AUROC gains are +0.109 and +0.146. Paired linearized 95% multiplier intervals clustering repeated sketches of the same source object are [0.067, 0.151] and [0.106, 0.185]. The overall AUROC gains have intervals spanning zero in both directions. Correlated gallery objects remain shared across queries. These intervals are conditional on the observed galleries and classes.

Uncertainty alone provides weak evidence about the desired decision. Among direct/diffused disagreements select the highest-risk quarter per sketch, rounding up. Substituting the direct order fixes 684 pairs and damages 1,385 in Ext-1, or fixes 704 and damages 1,359 in Ext-2. Requiring agreement from whole-image visual CLIP with the proposed direct order yields 347 / 847 and 334 / 822 on the same high-risk subsets. The proposed confidence signal predicts some reversals between galleries, but these fallback decisions harm more correctly ordered pairs than they recover.

## Which gallery contributions drive the reversal?

Write a candidate's positive pre-log score as its contribution from endpoints common to both galleries plus its contribution from other endpoints. The first term additionally factors into total probability mass sent to common endpoints and their conditional mean query evidence. All intermediate graph nodes stay in the original native gallery. Swap one scalar component at a time across the two already-computed native scores to isolate where the change occurs. These retrospective component exchanges are explanatory interventions. No deployed ranker can observe the other gallery to make these exchanges.

The precise replay fixes the original 15,149 pair identities, evaluates the linear operator at full fp32 matmul precision, and resolves all 1,489 original sign changes. The largest absolute reconstruction residuals were 1.1e-7 and 3.7e-7. Mean total probability on the same common photo endpoints changed from 80.37% to 86.72%.

| Replaced score component | Ext-1 → Ext-2, sign changes reproduced / 1,489 | Ext-2 → Ext-1 |
|---|---:|---:|
| Complete common-endpoint contribution | 806 | 771 |
| All other endpoints' contribution | 718 | 683 |
| Only total mass on common endpoints inside the common term | 718 | 729 |
| Only relative mixture within the common endpoints | 189 | 192 |

Counts from different rows overlap. The one-component swaps need not preserve the sum of both components' probability masses. They localize score terms and should not be interpreted as normalized alternative kernels or claims of cross-domain causal invariance. In the first direction, 713 changes occur with only the common-term swap, 625 with only the other-term swap, 93 with either swap, and 58 need both. Photographs common to two native galleries are known to the retrospective measurement through file identity. These experiments verify dataset-context sensitivity within sketch-to-photo retrieval.

Reproduce the fixed fixture with `python -m tf_sbir.gallery_context_reversals`, then run `python -m tf_sbir.gallery_context_mediation` at `fd303e6607793219930f0387ca7b75415eb4c8b5`. Run `python -m tf_sbir.gallery_context_prediction` at `e20e1e6bd588a6421d169b83fda98c4aaf5f0c01` for reversal forecasting and the directional controls. All experiment runs used Colab L4. [Context effects in choice, ICML 2019](https://proceedings.mlr.press/v97/seshadri19a)
