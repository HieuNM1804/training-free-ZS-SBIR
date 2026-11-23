# Testing shared learned position labels in sketch–photo matching

All conditions keep the CLIP ViT-B/32 parameters, the original image pixels, and the physical 7×7 patch grid fixed. Each image was encoded four times while reflecting only its learned position table. The independent estimator averages all 16 possible pairings of query and photo position assignments in the affinity before softmax. Its matched-frame counterpart uses four pairings in which the assignments agree. Their analytically isolated shared component is different from the ordinary physical coordinates used to measure geometric consistency. The direct 16-pair and 12-pair checks agreed with the accelerated identities to within the logged numerical tolerance.

| Geometry used for word feedback, original mAP / P in % | Sketchy Ext-1 pilot | Sketchy Ext-2 pilot |
|---|---:|---:|
| Strong base without geometry word update | 85.96 / 85.58 | 75.74 / 74.60 |
| Ordinary native position table | 86.00 / 85.15 | 77.44 / 76.06 |
| Four same-frame comparisons | 85.82 / 85.26 | 76.72 / 75.19 |
| Independent query and photo frame assignments | 85.58 / 85.13 | 76.28 / 74.58 |
| Independent assignments with scale matched to same-frame logits | 85.58 / 85.12 | 76.28 / 74.58 |
| Only mismatched frame assignments | 85.56 / 85.13 | 76.56 / 74.66 |
| Only the shared frame-sensitive component | 86.02 / 84.97 | 77.44 / 75.66 |

Compared with the four same-frame scores, independent assignments brought the mean blank-query diagonal patch mass down from 3.69% to 3.20% on Ext-1 and from 3.89% to 3.31% on Ext-2. Chance uniform mass would be 2.04% on a 49×49 match matrix. On Ext-2 the fraction of correct top-choice nominees fell from 45.18% with matching frames to 38.41% with independent frames, and native geometry achieved 55.47%. The diagnostic suppresses a measurable position effect, but its removal hurts selection of semantically relevant images. Frame dependence is therefore not sufficient evidence of a harmful artifact for this task. Canonical position or orientation may carry useful structure. This inference does not resolve which part of the remaining signal generalizes to an entirely different domain pair.

Each development split uses 768 sketches and the full gallery. Hardware was Colab Tesla T4. No benchmark label affected the estimator or the frame choices. The independent variant did not advance to TU after trailing the ordinary native readout on both pilots. [Design and source paper](protocol.md) · [Results and geometric checks](pilot-t4.json) · [Per-query results](paired-t4.zip).
