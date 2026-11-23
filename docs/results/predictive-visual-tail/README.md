# Can other gallery images validate a high visual rank?

All rankings were computed for 768 uniformly sampled unseen sketches per split against the complete original galleries of 17,100 and 12,552 photos. The two splits overlap. Selection on Sketchy is exploratory. Values are percentages in the original KD-SBIR evaluator. Ext-1 reports mAP@all / P@100 and Ext-2 reports mAP@200 / P@200. Encoders are frozen OpenAI CLIP ViT-B/32. The public vocabulary contains the same 20,931 Open Images words in both splits. No other sketch participates in an individual ranking. All inference and evaluation ran on Google Colab Tesla T4. Scoring revision `0a7c80aa7b4dd8aff6fa270beb76a93e04228c83`. The reference here is two-view word matching and RGB gallery diffusion. Other branches also report a stronger regional reference. The two references answer different comparisons.

For each candidate ask whether its other gallery neighbors see unusually many high visual ranks from the current sketch. Compare against photos with a similar starting semantic score. Integrate over an unknown rank threshold and tail probability to form a likelihood ratio. Remove the candidate from all neighbor sets. Replace semantic neighbors by RGB neighbors or shuffled neighbors to test specificity. Other controls keep increment magnitudes and assign them in the order of the visual ranks. Spatial/graph scan statistics and gallery diffusion inspired the experiment. The conditional rank likelihood and the gallery controls are specific to this adaptation. [Chen and Neill, KDD 2014](https://www.cs.cmu.edu/~neill/papers/kdd2014.pdf). [Wang et al., CVPR 2012](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/cvpr12.pdf).

| Readout | Ext-1 | Ext-2 |
|---|---:|---:|
| native | 83.003 / 82.850 | 74.069 / 72.778 |
| semantic_conditional/0.125 | 82.850 / 82.724 | 74.007 / 72.885 |
| semantic_conditional/1 | 79.025 / 79.120 | 70.896 / 69.923 |
| shuffled_conditional/0.125 | 82.989 / 82.837 | 74.034 / 72.788 |
| matched_conditional_rank/0.125 | 82.768 / 82.879 | 73.770 / 72.842 |

At unit strength, 44 / 85 sketches gain their first correct top-200 result, while 4 / 2 lose it. Both splits lose several mAP points. At the smallest tested nonzero strength the conditional rank method still trails the reference in mAP on both splits. This provides evidence for a coverage-density tradeoff: verification by neighbors can rescue an otherwise excluded class but changes the order for many more sketches. This version did not advance to TU.

[Full tables](sketchy-pilot-t4.json).
