# Does the identity of the displaced image help interpret the sketch?

In each opposite gallery fold, an eight-photo draw has two winners, one under the real query's patch geometry and one under the blank query's geometry. Estimate noun probabilities from which winner replaced which. No benchmark labels enter either choice. Under shared draws the winners differed for 70.6% and 70.1% of decisions, versus 98.0% with independent draws on the two splits. The calculated joint probability exactly reproduced exhaustive enumeration on a small tied example. The first pilot kept the 512 highest query words distinct, grouping the remaining probability into a residual category. This retained only 65.9% / 64.8% as individually identified words. The follow-up evaluated all 20,931 words and exact winner pairs. Both used the same 768 queries per split, all gallery photos and the original KD-SBIR metrics on Colab T4.

| All words, mAP / precision | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Strong base | 85.96 / 85.58 | 75.74 / 74.60 |
| Previous calibrated geometry | 86.00 / 85.15 | 77.44 / 76.06 |
| Matched choice likelihood | 86.15 / 85.40 | 77.11 / 75.63 |
| Independent winners with identical marginals | 86.17 / 85.69 | 76.76 / 75.41 |
| Same winners with reversed orientation | 80.85 / 79.31 | 69.26 / 68.04 |
| Signed orientation term only | 84.63 / 83.23 | 77.77 / 75.65 |
| Symmetric penalty only | 83.86 / 83.54 | 72.79 / 71.80 |
| Ratio of marginal word means | 86.30 / 85.76 | 77.27 / 75.79 |

On the grouped pilot the matched likelihood predicted held-fold choice orientations more accurately than the previous estimator. Its mean log score relative to a fair choice was 0.105 / 0.089 versus 0.065 / 0.055. On complete-vocabulary retrieval, matching the two winners changes Ext-1 mAP by −0.01 point versus independent winners and improves Ext-2 by 0.35. The preceding method still has higher Ext-2 mAP. The simple marginal ratio gives higher mAP than the matched method on both pilots. Choosing the matched method as a publication candidate solely on the basis of the improved orientation prediction is unsupported.

A direction-versus-reversal comparison has a stronger limitation. Its signed log odds depend only on the two marginal distributions. The distinctive contribution of pairing is a symmetric penalty for the difference in strength between the two winners' word preferences. This identity holds for the full vocabulary. A fused T4 reduction allows exact aggregation without sampling or word truncation and agreed with a float64 reference on the numerical check to within 0.000001. This experiment remains exploratory on Sketchy.

[Scorer and controls](protocol.md) · [Derivation](likelihood-decomposition.md) · [All-noun results](all-words-t4.json) · [Grouped pilot and held-fold scores](pilot-t4.json) · [Grouped per-query results](paired-t4.zip) · [Complete per-query results](full-vocabulary-t4.zip).
