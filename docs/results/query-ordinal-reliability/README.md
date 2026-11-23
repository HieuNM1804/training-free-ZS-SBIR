# Can one sketch identify a reliable way to compare photos?

For each sketch the frozen strong ranker supplies its first 256 photos. Split their rank positions into two disjoint sets of 128 using a fixed seed and swap roles. Four readers rank pairs of photos using the same frozen CLIP ViT-B/32 and approved vocabulary. They use whole-image visual similarity, direct word posterior agreement, mean visual similarity across two poses and 13 crops, and gallery-diffused word agreement.

On one panel estimate the six observed pairwise agreements of four readers. Under a symmetric independent-error model given a latent pair orientation, each off-diagonal agreement factors as the product of the readers' signed accuracies. Fit the four factors by closed-form least squares of the log agreements. Return equal weights if a positive solution with factors below one is unavailable. Each reader's log-odds weights its percentile rank. The individual-reader selection diagnostic breaks weight ties by index and picks the first visual reader on fallback. The proper weighted ensemble uses equal weights in these cases. Each held photo gets weights from the opposite panel. No sketch shares information with another.

The major assumption is that the latent ordering tracked by agreeing readers is the one that places relevant photos first. On the subset of held panels containing both positive and negative photos, ranking the readers by their estimated reliability has average within-sketch rank concordance −0.061 on Ext-1 and −0.008 on Ext-2. A direct word score wins 84.68% / 66.17% of positive-negative pairs in these mixed panels, compared with 52.74% / 64.10% for the incumbent on identical pairs. Only 4,382 / 5,265 queries have any mixed held panel. This weights queries equally among those with such a pair and excludes uninformative panels. It need not predict class-level or full ranking gains. For example, moving word matches to the top repaired 77 and damaged 184 previously correct first results on Ext-1.

| Original KD evaluator, full queries and gallery | Ext-1 mAP@all | P@100 | Ext-2 mAP@200 | P@200 |
|---|---:|---:|---:|---:|
| Frozen reference | 85.29 | 84.65 | 77.55 | 75.45 |
| Order the 256 candidates by direct words | 85.26 | 84.58 | 76.92 | 75.49 |
| Diffused words | 85.26 | 84.49 | 77.34 | 75.38 |
| Equal-weight percentile ranks | 85.26 | 84.58 | 76.41 | 75.26 |
| Cross-panel moment weights | 85.26 | 84.57 | 76.38 | 75.27 |

Class-bootstrap intervals for moment weights versus equal weights on held-panel pair accuracy include zero on both splits. The held-panel accuracy and the full retrieval replay do not establish a usable per-sketch reliability estimator. This does not distinguish conditional dependence of the readers from nuisance orderings shared by them.

The rank factorization adapts unlabeled classifier agreement to choices over pairs of candidate photos inside a single query. Common origins in a single CLIP encoder make independence questionable. [Jaffe et al., AISTATS 2015](https://proceedings.mlr.press/v38/jaffe15.html) · [Jaffe et al., AISTATS 2016](https://proceedings.mlr.press/v51/jaffe16.html)

Code `552feb00f54b7920ab40f2e84826f940ffbe1afc`, Colab L4, `python -m tf_sbir.query_ordinal_reliability`. Full counts, score vectors, feasible-fit counts, data sizes, and intervals are in `sketchy.json`. TU-Berlin was not used in this study.
