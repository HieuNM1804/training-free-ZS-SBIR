# Does the direction of gallery propagation cause unstable cross-domain retrieval decisions?

A word reading of one independent sketch supplies positive evidence at every photo. The frozen RGB diffusion graph averages that evidence with a directed endpoint kernel K. We asked how much the direction of information flow, after accounting for the gallery's stationary sampling weights, contributes to instability. We measured the exact stationary distribution pi and formed the time-reversed kernel K*(i,j) = pi(j) K(j,i) / pi(i). The additive reversible kernel (K+K*)/2 preserves pi and removes the stationary current. Controls reverse or average after assigning every initial photo equal mass. All alternatives see the same gallery, evidence and scoring backbone. This decomposition adapts an established idea from directed-graph learning. [Zhou et al., ICML 2005](https://icml.cc/Conferences/2005/proceedings/papers/131_LabeledUnlabeledGraph_ZhouEtAl.pdf) · [Wang et al., CVPR 2012](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/cvpr12.pdf)

| Treatment | Ext-1 mAP@all / P@100 | Ext-2 mAP@200 / P@200 | Same-pair reversals across galleries | Correct fixed pairs, Ext-1 / Ext-2 |
|---|---:|---:|---:|---:|
| Native directed kernel | 83.00 / 82.85 | 74.07 / 72.78 | 1,488 | 11,191 / 10,973 |
| Reversible average | 82.99 / 82.82 | 74.06 / 72.79 | 1,461 | 11,235 / 11,014 |
| Stationary time reversal | 82.96 / 82.77 | 74.09 / 72.75 | 1,440 | 11,287 / 11,049 |
| Transpose normalized under a uniform prior | 82.66 / 82.35 | 74.08 / 72.50 | 1,527 | 10,958 / 10,631 |
| Average with the uniform-prior transpose | 82.85 / 82.57 | 74.20 / 72.69 | 1,491 | 11,098 / 10,815 |

The stationary joint-flow total variation distance between forward and reverse was only 0.0212 and 0.0238. The maximum relative stationarity residual was below 0.000003 on both splits. Reversing the stationary current improves the selected pair counts on both sides, but it produces no consistent full-gallery mAP gain. The root mean square across-gallery change in margin is 0.278 log units. The change in the forward-versus-reversible score correction is 0.022 log units RMS. These two quantities may correlate and should not be read as an additive variance decomposition.

Metrics use 768 uniformly selected queries per native split (seed 87243) and replay the original KD-SBIR evaluator. They concern the standalone two-view word plus RGB-diffusion baseline. The separate conditional ranking diagnostic uses the [archived 15,129 T4 pairs](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/d7cacfdc94a69ff1968f8c354f814d9455e87236/docs/results/public-context-standardization/t4-fixed-boundaries.zip) on four overlapping classes. No labels or other sketch enter the kernel or its time reversal. We did not select a configuration on TU-Berlin. Run `python -m tf_sbir.gallery_flow_direction` at `aaac39245320f8548bffca320f6beb07934eaf7a` on Colab T4.
