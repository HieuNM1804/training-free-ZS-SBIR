# Can the response to gallery expansion predict the direction of a rank change?

The locked two-view public-word and RGB-diffusion scorer assigns each photo a log score. Our hypothesis is that this score has a gallery-size-dependent bias, and that a change measured by thinning the gallery forecasts the direction in which the same pair's margin will move. The method uses a single frozen CLIP ViT-B/32 and builds everything from the unlabeled native gallery and one sketch. The test uses the gallery's class labels only after assigning scores.

Construct eight random nested partitions into quarters and halves. A score for a photo always comes from a graph containing that photo. Every subset rebuilds the same native RGB graph. Let Q, H, F denote log-scores in a quarter, a half, and the full gallery. Under a first-order 1/n bias, F−H is predicted by (H−Q)/2 and the extrapolated score is 2F−H. This rate is a falsifiable hypothesis for fixed-neighbor diffusion. Multiscale nearest-neighbor extrapolation has an antecedent in supervised classification. A transfer of its convergence guarantees to this method is unproved. [Okuno and Shimodaira, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f9028faec74be6ec9b852b0a542e2f39-Abstract.html) · [Wang et al., CVPR 2012](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/cvpr12.pdf)

For the half-to-full directional test, estimate H−Q on four draws and compare it with F−H on the other four. Both members of a pair must coexist in the quarters or halves used to measure their margin. Swap the roles of the two groups for a second measurement. This avoids subtracting the same noisy half-gallery observation on both sides. Overlapping objects and queries are still present.

| Forecast | Ext-1 | Ext-2 |
|---|---:|---:|
| Fraction of correctly forecast margin-change signs on random expansion, two assignments | 71.5% / 70.0% | 68.4% / 69.4% |
| Mean squared forecast error relative to no change, two assignments | 0.839 / 0.874 | 0.839 / 0.877 |
| Fraction with correct direction when the other real gallery replaces the native one | 72.0% | 31.1% |
| Relative squared error on the actual gallery swap, original coefficient | 1.083 | 1.847 |

Even where the direction was frequently right, the correction magnitude and category relevance were different questions. Of queries' 991 fixed-pair switches under Ext-1 extrapolation, 429 helped the category order and 562 hurt. Ext-2 had 349 helped versus 698 hurt. These are pair counts, with repeated queries and photos. Extrapolated rankings on both native galleries reverse 1,506 pairs on the shared T4 fixture versus 1,488 with native diffusion.

| Original KD evaluator, 768 uniform queries per split and full original gallery | Ext-1 mAP@all | P@100 | Ext-2 mAP@200 | P@200 |
|---|---:|---:|---:|---:|
| Word posterior plus RGB diffusion | 83.00 | 82.85 | 74.07 | 72.78 |
| Extrapolated log score | 82.28 | 82.22 | 73.84 | 72.37 |
| Mean log score on half galleries | 83.48 | 83.38 | 74.01 | 72.92 |
| Add a candidate-permuted change | 82.64 | 82.93 | 73.55 | 72.53 |

The comparison uses the standalone word-plus-diffusion component of the strong reference. The file `sketchy-t4.json` includes both mAP and precision per method, class-bootstrap intervals, and both directions of the fixed-pair audit. No TU selection or evaluation was performed. Code `c5b82220fad553924fd44c61fe4a379c82a9c6d4`, Tesla T4, `python -m tf_sbir.finite_gallery_response`. The [pinned T4 pair fixture](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/d7cacfdc94a69ff1968f8c354f814d9455e87236/docs/results/public-context-standardization/t4-fixed-boundaries.zip) can be restored to `/content/sbir/results/gallery-context-reversals`.
