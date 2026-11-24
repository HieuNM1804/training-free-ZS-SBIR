# Do two distinct correspondences provide better GCWE feedback?

| Verifier | Ext-1 mAP@all / P@100 | Ext-2 mAP@200 / P@200 |
|---|---:|---:|
| Native GCWE | 84.373 / 83.981 | 78.488 / 76.183 |
| Conditional agreement given disjoint endpoints | 84.282 / 83.711 | 78.339 / 75.832 |
| Joint probability of disjoint endpoints and agreement | 84.282 / 83.778 | 78.416 / 75.843 |
| Raw offset concentration | 84.472 / 83.952 | 78.666 / 76.157 |
| Centered within-part concentration | 84.541 / 84.204 | 78.344 / 76.274 |
| Centered whole-grid concentration | 84.553 / 84.174 | 78.447 / 76.254 |
| Centered whole-grid with scrambled coordinates | 84.653 / 84.193 | 78.889 / 76.310 |

On 1,152 queries per split, excluding duplicate correspondences and conditioning on distinct endpoints reduced mAP by 0.091 and 0.149 against native GCWE. It also underperformed raw concentration. The exact conditional probability passed exhaustive numerical checks on three nondegenerate small couplings and a degenerate point mass. Improving the statistical definition of spatial corroboration did not improve these retrieval rankings.

Removing the even/odd partition produces higher point estimates than within-part concentration on this sample, but scrambling photo coordinates improves the whole-grid result on both splits. This does not support a physical-offset explanation for the whole-grid variant. Centered within-part concentration underperforms native GCWE on this particular Ext-2 subset despite its positive full-split estimate. No promotion or additional TU evaluation. [Frozen method and controls](protocol.md) · [All Ext-1 rows](sketchy_1.json) · [All Ext-2 rows](sketchy_2.json) · [Per-query data](paired-T4.zip)
