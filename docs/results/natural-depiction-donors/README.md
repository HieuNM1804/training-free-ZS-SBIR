# Which leg of the gallery route fails?

All rankings were computed for 768 uniformly sampled unseen sketches per split against the complete original galleries of 17,100 and 12,552 photos. The two splits overlap. Selection on Sketchy is exploratory. Values are percentages in the original KD-SBIR evaluator. Ext-1 reports mAP@all / P@100 and Ext-2 reports mAP@200 / P@200. Encoders are frozen OpenAI CLIP ViT-B/32. The public vocabulary contains the same 20,931 Open Images words in both splits. No other sketch participates in an individual ranking. All inference and evaluation ran on Google Colab Tesla T4. Retrieval revision `f25347a0045ba26061800b17837cfcc5d71fbd25`. Diagnostic revision `4d72a368b266d4b44a2a1824e744d77bb3c2d0e9`. Each gallery image receives a depiction log odds from two frozen prompt banks with the same public nouns and equal priors. Within a split, “upper” means the half with higher source-versus-target log odds. Both halves are defined by the model. These are relative responses without human style annotations. Intermediate-domain adaptation motivates testing naturally occurring intermediaries. This experiment uses existing images and no fitted translator. [DLOW, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Gong_DLOW_Domain_Flow_for_Adaptation_and_Generalization_CVPR_2019_paper.html).

## Which photos are recovered?

Among 96 / 137 sketches where the strong reference has zero correct photos in its top 200, visual cosine returns a correct image for 64 / 124. The mean per-query fraction of correct cosine returns in the upper half is 81.40% / 86.22%. For those same classes, the corresponding gallery frequencies are 49.04% / 42.39%. The excess also survives averaging equally across represented classes, +30.55 points (bootstrap 95% interval 16.70–42.54, 20 classes) and +39.96 points (32.19–47.49, 17 classes). The two Sketchy splits overlap, so these are checks on related populations. Strong retrieval-miss conditioning limits this observation to that subset.

## Can a naturally source-like image bridge to the other half?

Retrospectively identify the 61 / 124 strong misses with at least one correct upper-half photo inside visual top 200. The highest-ranked upper-half photo belongs to the correct class for just 3 / 61 and 17 / 124 sketches. Use the selected donor to retrieve only from the opposite half, with 8,550 / 6,276 eligible endpoint photos. Label-informed alternatives below are diagnostic oracles; all normal donor and endpoint rankings use no labels.

| Endpoint ranking on these recoverable misses | Ext-1 mean fraction correct in top 200 | Ext-2 |
|---|---:|---:|
| Direct sketch cosine | 13.33% | 21.01% |
| First upper donor, photo cosine | 7.15% | 19.74% |
| First upper donor, photo words | 6.34% | 16.06% |
| Earliest correct upper donor, photo cosine (oracle) | 50.93% | 63.97% |
| Earliest correct upper donor, photo words (oracle) | 69.06% | 91.58% |
| Random correct upper donor, photo words (oracle) | 82.65% | 91.79% |

For photo cosine, the macro class gain from selecting the earliest correct donor over the actual first donor is +42.65 points (bootstrap 95% interval 29.62–56.00, 19 classes) / +40.39 (27.01–54.45, 17 classes). On this selected subset, the ranking of candidate donors loses much of the available class signal. Correct donors can lead to many matching photos in the opposite half. A candidate verifier must identify which donor is correct using a single sketch and an unlabeled gallery before these gains are actionable. The target-half diagnostic is separate from official whole-gallery mAP.

## Official paired retrieval controls

Normalize the contribution of source-like versus target-like RGB neighbors by the prevalence of each depiction. Exclude the endpoint itself from every donor graph. Swap the weights, ignore style altogether or shuffle style within the target noun to test whether style-specific evidence drives any gain.

| Readout | Ext-1 | Ext-2 |
|---|---:|---:|
| strong | 85.964 / 85.579 | 75.744 / 74.598 |
| query/0.03125 | 85.942 / 85.573 | 75.908 / 74.653 |
| all/0.03125 | 85.934 / 85.464 | 76.013 / 74.615 |
| reverse/0.03125 | 85.938 / 85.469 | 75.992 / 74.635 |

Reversing or removing depiction-dependent weights yields larger Ext-2 gains. The discovery about which photos are recoverable does not justify favoring all source-like donors in a final scoring method.

[Retrieval pilot and controls](sketchy-pilot-t4.json) · [Two-leg diagnostic and class intervals](path-diagnostic.json).
