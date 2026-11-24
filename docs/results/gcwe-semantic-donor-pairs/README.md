# Semantic dissimilarity in GCWE donor pairs

| GCWE sampling law | Ext-1 mAP@all / P@100 | Ext-2 mAP@200 / P@200 |
|---|---:|---:|
| Original | 84.306 / 84.269 | 78.574 / 76.429 |
| Four semantic determinant pairs | 84.274 / 84.212 | 78.552 / 76.414 |
| Four uniform distinct pairs | 84.306 / 84.266 | 78.573 / 76.430 |
| Independent draws with semantic endpoint marginals | 84.282 / 84.229 | 78.553 / 76.419 |
| Shuffled semantic pairing | 84.323 / 84.274 | 78.598 / 76.456 |

On 1,536 queries per split, changing only the donor proposal lowered mAP by 0.032 and 0.022 points against native GCWE. Both matched controls also outscored the proposed sampling law on both splits. This fails the prespecified promotion criterion. No TU run was triggered.

[Mechanism, prior art and locked comparisons](protocol.md) · [Complete Ext-1 results](sketchy_1.json) · [Complete Ext-2 results](sketchy_2.json) · [Per-query data](paired-T4.zip)
