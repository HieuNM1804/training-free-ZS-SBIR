# Is a high diffused score supported by representative neighboring photos?

For each sketch and gallery photo let L be its positive two-view public-word agreement. The ordinary gallery contribution for candidate i is log E[L] under the endpoints of the normalized RGB diffusion kernel starting at i. The alternative uses log E[L^t] / t. The limit at t=0 is E[log L]. The Jensen gap log E[L] − E[log L] measures the difference between the two forms of pooling. A larger value means that some favorable endpoints can support the arithmetic mean despite low log evidence at others. All variants use the same evidence, encoder and graph.

The hypothesis predicts that the wrong photo has a larger gap than its relevant neighbor when immediately above it in the original strong top 200. On Ext-1 this held for 56.96% of 11,266 wrong–right boundaries. On Ext-2 it held for only 42.69% of 33,055. The average wrong-minus-right gaps were +0.1205 and −0.0128 log units, respectively. These pooled observations can share queries and photos. Their sign does not identify a universal correction. The geometric mean and the half-power mean both reduced mAP on Ext-1. For Ext-2 the half-power change is close to its control that preserves the per-sketch score distribution but keeps candidates in their original diffusion order.

| RGB aggregation within strong reference | Ext-1 mAP@all | P@100 | Ext-2 mAP@200 | P@200 |
|---|---:|---:|---:|---:|
| Original arithmetic mean | 85.29 | 84.65 | 77.55 | 75.45 |
| Geometric mean | 84.59 | 84.58 | 77.54 | 75.45 |
| Power 0.5 | 85.05 | 84.67 | 77.58 | 75.47 |
| Power 2 | 85.30 | 84.51 | 77.42 | 75.38 |

The full query-by-query original KD-SBIR replay is in `sketchy-official.json`. It includes the standalone word-plus-RGB variants, distribution-matched controls and class-bootstrap intervals. Finite-precision GPU propagation produced a minimum nominal Jensen gap of approximately −0.003 log units on Ext-2. The original exploratory run is also included. This branch did not justify tuning a different power for each dataset.

The exponential certainty equivalent has been used in risk-sensitive reinforcement learning. Here it is applied once over the endpoints of a fixed visual retrieval kernel. [Fei et al., NeurIPS 2021](https://papers.neurips.cc/paper_files/paper/2021/hash/ab6439fa2daf0246f92eea433bca5ac4-Abstract.html) · [Wang et al., CVPR 2012](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/cvpr12.pdf)

Run `python -m tf_sbir.gallery_diffusion_risk --official` on Colab L4. Code `4a2bea0d59308e028e92bebc40bfc00b1165c380`.
