# Shrinking a pairwise gallery correction

A gallery can help a query by changing the order given by direct word evidence. Earlier experiments found that deleting some surrounding photos forecasts which pair orders will flip between two real galleries. We tested whether that signal can also estimate the appropriate amount of gallery correction inside a single independent sketch retrieval.

For an image pair, the observed correction is its log-score margin under native diffusion minus the direct lexical margin. Eight seeded random gallery halvings estimate the mean squared change of the diffused margin in contexts retaining both photos. A centered normal prior on contextual correction and a heteroscedastic observation model give weight T/(T+V), where V is that pair's perturbation variance proxy and T is the nonnegative second moment of correction less the average estimated V. Missing perturbations use the same sketch's mean. The scale model is an assumption. Sampling only from the native gallery need not reproduce the changes in a new gallery.

Each candidate compares with the same 64 uniformly chosen gallery anchors. The mean of the corrected pair margins plus one common anchor offset produces its retrieval score. The strong frozen reference replaces its diffused lexical contribution by this score. The two controls use the same mean trust for every pair, or drop the paired cross term from the variance estimate. Gallery labels never enter this calculation.

Each split used a uniform sample of 768 queries with the same pre-specified seed and its full gallery. All rankings were replayed with the source-pinned KD-SBIR evaluator on Colab L4. The numerical result is exploratory.

| Method | Ext-1 mAP@all | P@100 | Ext-2 mAP@200 | P@200 |
|---|---:|---:|---:|---:|
| Native reference on the sample | 85.96 | 85.57 | 75.73 | 74.59 |
| Joint margin shrinkage | 85.94 | 85.60 | 75.55 | 74.59 |
| Same mean trust | 85.98 | 85.64 | 75.69 | 74.61 |
| Remove paired covariance | 85.94 | 85.59 | 75.54 | 74.59 |

Average trust in the native correction was 0.973 / 0.961. The proposed method fixed 1 / 2 wrong first results and damaged 5 / 6 correct first results on Ext-1 / Ext-2. The class-balanced Ext-2 mAP difference versus the native reference was −0.182 points, with a paired class-bootstrap interval of [−0.360, −0.028]. There is no evidence here for promoting adaptive shrinkage above its simple control. These pilot values must be compared with the reference on the same queries. The matched-context diagnostic and this scoring intervention answer different questions.

The normal hierarchy and method-of-moments shrinkage are standard statistical constructions. This method substitutes photo-pair perturbation under gallery thinning for measurement noise, which the cited paper does not establish. The diffusion operator follows Wang et al. The controlled order-reversal question draws on context-dependent choice. [Xie et al., JASA 2012](https://www.tandfonline.com/doi/full/10.1080/01621459.2012.728154) · [Wang et al., CVPR 2012](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/cvpr12.pdf) · [Seshadri et al., ICML 2019](https://proceedings.mlr.press/v97/seshadri19a)

Run `python -m tf_sbir.gallery_context_shrinkage`. Scoring revision `9e8c6495c599d2d1ff5b3addeccb77dcadee6638`.
