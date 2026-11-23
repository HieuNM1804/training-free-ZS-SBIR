# What breaks the semantic channel at equal perturbed mass?

Within the identical support of 7,398 public word IDs, permute query-word probabilities either between reciprocal neighbors in the text space or between adjacent ranks in the gallery frequency of words. For each query interpolate both permutations toward the original just enough to equate total variation. Both corrupt the same allowed IDs and move the same total probability per query. The affected IDs can differ in the amount each contributes. On 1,536 seed-42 queries from each split the mean moved probability was 15.14% and 14.31%, respectively.

| mAP / precision (%) | Ext-1 direct | Ext-2 direct | Ext-1 diffused | Ext-2 diffused |
|---|---:|---:|---:|---:|
| Original | 75.86 / 79.79 | 70.46 / 69.49 | 82.32 / 82.19 | 73.25 / 71.60 |
| Exchange nearby words | 75.27 / 78.98 | 70.11 / 69.23 | 81.98 / 81.39 | 73.15 / 71.52 |
| Exchange frequency-matched words | 64.85 / 69.21 | 66.94 / 65.68 | 74.81 / 73.79 | 72.13 / 70.09 |

Text cosine averaged 0.940 for nearby pairs and 0.676 for frequency pairs. Broad semantic identity has a larger effect than choosing between these nearby names on both splits. The effect size varies with the gallery. Neighboring names may denote related, distinct objects, so this evidence concerns the measured text space. The original remains the strongest among the tested direct scores.

[Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Per-query records](paired-L4.zip)
