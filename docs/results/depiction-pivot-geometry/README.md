# Can geometry verify a cross-domain image intermediary?

The complete held-out TU-Berlin experiment includes all 2,130 sketches, all 30 unseen classes and 27,262 gallery images. Sketchy development used 768 uniformly sampled sketches on each original split with the complete respective galleries. The splits overlap. All runs used Colab Tesla T4, frozen OpenAI CLIP ViT-B/32, independent sketches, the public Open Images vocabulary and KD-SBIR's original evaluator. The same full-gallery strong baseline was reconstructed on the T4. Comparisons in this report are paired on the same queries and device.

## Mechanism under test

A gallery intermediary has two jobs. Its appearance must carry the query's content across depiction, and its photo-side word distribution must retrieve the correct target photographs. The previous [diagnostic](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/1dd06b4ee53e4f52a018434fc5927c795b1bc386/docs/results/natural-depiction-donors/README.md) found that ranking the intermediaries was a bottleneck. This study verifies candidates by asking whether patch pairs matched by frozen CLIP preserve their relative distances. With the 49 native patches in each image, form the joint distribution by averaging the row and column conditional matches. Draw two correspondences independently under this joint distribution. Score each candidate by Pearson correlation between the two patch distances in the sketch and in the candidate. The product of marginal expectations gives the coincidence correction. No fitted transport, contour operation, detector or gradient enters this calculation.

This is a frozen-affinity check of correspondence, motivated by spatial matching and probabilistic comparison of distance matrices. Patch correspondence is established prior art. Its role here is to localize and verify intermediaries in cross-domain retrieval. [Cho et al., CVPR 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Cho_Unsupervised_Object_Discovery_2015_CVPR_paper.html). [Peyré et al., ICML 2016](https://proceedings.mlr.press/v48/peyre16.html).

## How often did the verifier select the correct intermediary?

Take candidates in the visual top 200 and the gallery half with higher CLIP source-depiction evidence. Every automatic selector uses the same candidate set. Score its top candidate against unseen labels only after selecting it. The hard subset has zero correct strong-base top-200 images and at least one correct candidate in that set. The blank control preserves the original grid and mean query color while removing the image pattern. The coordinate-shuffle control keeps the same content affinities but randomizes the candidate grid's distance assignment.

| Correct choices in hard subset | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Hard subset size | 61 | 124 | 184 |
| Nearest by whole-image cosine | 3 | 17 | 26 |
| Geometric evidence | 6 | 31 | 55 |
| Shuffled photo patch coordinates | 3 | 12 | 24 |
| Blank query canvas | 2 | 2 | 3 |
| Direct sketch words | 6 | 25 | 37 |
| Existing strong score restricted to the nominees | 4 | 3 | 10 |

On TU, macro-averaging over 27 hard-case classes, the improvement in donor accuracy against shuffled positions is 16.73 percentage points (95% class bootstrap interval 6.72 to 27.57). Against plain words the interval includes zero. Ext-1 also has wide intervals. Across all 2,130 TU queries the strong score selects correct donors more often (80.75%) than geometry (62.30%). On naturally occurring strong misses, geometry has information which the already-failed score often lacks.

The depiction enrichment also transfers. Of the 240 complete strong misses on TU, 188 have a relevant visual top-200 image. Their mean fraction of correct visual returns in the upper style half is 83.11%, versus 51.18% in their class-matched gallery. The macro class excess is 31.37 points, with a 95% class bootstrap interval of 23.84 to 38.83. Neither split nor image-level labels are used by the scoring code to define the style partition. “Source-like” is relative to frozen domain prompts, without independent human annotation.

## Effect when attaching just one intermediary

| mAP / precision in percent | Sketchy Ext-1 pilot | Sketchy Ext-2 pilot | TU-Berlin full |
|---|---:|---:|---:|
| Strong paired base | 85.96 / 85.58 | 75.74 / 74.60 | 80.70 / 81.49 |
| Geometry | 85.82 / 85.56 | 75.98 / 74.79 | 80.92 / 81.62 |
| Shuffled geometry | 85.81 / 85.40 | 75.77 / 74.55 | 80.66 / 81.38 |
| Direct words choose the intermediary | 85.82 / 85.50 | 75.86 / 74.69 | 81.02 / 81.72 |
| Strong score chooses the intermediary | 86.01 / 85.54 | 76.29 / 75.11 | 81.01 / 81.38 |

Ext-1 and TU use mAP@all and P@100. Ext-2 uses mAP@200 and P@200. A historical L4 full TU reference gave 80.59 / 81.32. All TU deltas here use the T4 reconstruction. Changing geometry to a perfect retrospective on/off switch for exactly the same correction could at most gain 0.165 / 0.623 / 0.392 mAP points on the respective populations. The shape mechanism has support for a conditional donor-identification error. Its current end-to-end gain is small relative to the competing semantic controls. The two datasets measure one domain pair.

[Pre-TU freeze](transfer-lock.md) · [Sketchy rerun](locked-pilot.json) · [TU transfer](locked-tu.json) · [Paired differences and headroom](failure-localization.json) · [Per-query tensors](paired-t4.zip).
