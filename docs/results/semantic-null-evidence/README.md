# Analytic Gaussian background correction removes useful CLIP evidence

Score only the global semantic component, on the same 768 queries per split with their full galleries. Results below follow the ordinary RGB diffusion.

| Query word logits | Ext-1 mAP@all / P@100 | Ext-2 mAP@200 / P@200 |
|---|---:|---:|
| Native | 82.978 / 82.810 | 74.024 / 72.220 |
| Query-specific Gaussian log moment subtracted | 74.416 / 73.883 | 60.190 / 57.947 |
| Pooled-gallery log moment subtracted | 76.771 / 76.454 | 62.359 / 60.727 |
| Query-specific penalties assigned to shuffled words | 81.569 / 81.288 | 72.388 / 70.479 |
| Query-specific minus pooled moment subtracted | 80.215 / 80.893 | 55.602 / 53.893 |
| Native word order with primary entropy | 83.079 / 82.902 | 74.086 / 72.258 |

The primary loses 8.562 and 13.834 mAP points. Matching the corrected entropy while preserving the native word order gives small positive point estimates relative to native. Incorrectly assigning the penalties is substantially less damaging than applying the analytic correction to its intended words. This falsifies the prediction that these strong responses can simply be discounted as nuisance under the proposed background. The importance of native image–word alignment in the earlier randomized test does not establish that subtracting its Gaussian log moment improves retrieval. Stop this intervention before TU. [Frozen hypothesis and equations](protocol.md) · [All raw and diffused scores, Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Per-query data](paired-T4.zip)
