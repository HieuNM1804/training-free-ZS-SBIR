# Do the names that distinguish visual groups also improve retrieval from sketches?

A reference task partitions unlabeled gallery photos into anonymous groups. A disjoint sample from those groups supplies CLIP observations. Each group receives a distinct public noun. A proposed noun replaces a current one if the replacement improves the balanced log probability of identifying the correct group across all alternatives. One replacement changes the normalization of every group's prediction. A sketch answers each of 16 such tasks using the nouns in its codebook. Candidate photos receive the response for their visual group. The codebooks were indexed before receiving any query. [Implementation](../../../tf_sbir/reference_lexicon.py)

| Full unseen mAP / precision (%) | Ext-1 | Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Direct semantic baseline | 75.35 / 77.10 | 72.66 / 71.07 | 73.46 / 73.98 |
| Joint code design | 77.58 / 79.24 | 70.25 / 68.38 | 70.50 / 72.40 |
| Independent utilities with one-to-one matching | 75.26 / 76.03 | 72.26 / 70.16 | 67.92 / 70.56 |
| Tuned full word distribution of each visual group | 80.07 / 80.03 | 75.78 / 73.43 | 77.90 / 76.78 |

Two full unseen Sketchy splits selected 36 groups for the joint code and 192 for the full-distribution control. TU used these locked choices. Ext-2 uses mAP@200 and P@200. Ext-1 and TU use mAP@all and P@100. The replay used KD-SBIR's original evaluator, OpenAI CLIP ViT-B/32, seed 42 and NVIDIA L4.

The jointly chosen names improved balanced identification accuracy on held-out photos under the source prompt. Against distinct names selected with separate utilities, accuracy rose from 65.53% to 66.60%, from 56.13% to 57.76%, and from 65.41% to 66.53%. These gains measure recognition of the artificial visual groups on photos. The complete word-distribution control outperformed joint code design on all three retrieval evaluations. [Frozen selection](freeze.json) · [Per-query results](paired-L4.zip)

Referential communication and context-dependent interpretation have precedents in learned agents and studies of human drawing. Here the representation being optimized is a finite code over public CLIP nouns for one unlabeled gallery task. The experiments give insufficient evidence for advancing this code as a primary SBIR method. [Emergent Communication of Generalizations, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/9597353e41e6957b5e7aa79214fcb256-Paper.pdf) · [Fan et al., Computational Brain & Behavior 2019](https://cogtoolslab.github.io/pdf/fan_cbb_2019.pdf)
