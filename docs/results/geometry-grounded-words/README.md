# Conditioning the nominee's public words on the spatially supporting region

The native geometric correlation can be decomposed into contributions from locations on the gallery image. Center these contributions, subtract the location contributions measured on the blank query, and compare them with the text responses of the same image patches. This changes each nominee's word posterior before the existing evidence pooling step. The exact reconstruction error of the whole-image geometric correlation was below 0.000002 on both splits. The position-conditioned update produced a mean KL divergence from the original image word posterior of 0.00133 / 0.00081 nats at the pretrained CLIP scale.

| mAP / precision in percent | Sketchy Ext-1 pilot | Sketchy Ext-2 pilot |
|---|---:|---:|
| Prior calibrated whole-image geometry | 85.999 / 85.151 | 77.435 / 76.059 |
| Use the actual patch-to-word alignment | 85.991 / 85.178 | 77.352 / 76.023 |
| Change both selected and reference word estimates | 86.003 / 85.168 | 77.418 / 76.047 |
| Keep geometry, shuffle patches supplying noun evidence | 85.998 / 85.141 | 77.448 / 76.064 |
| Reverse the logit adjustment | 86.005 / 85.139 | 77.542 / 76.090 |
| Four times the actual patch adjustment | 85.973 / 85.212 | 77.303 / 75.881 |
| Four times the shuffled adjustment | 85.995 / 85.098 | 77.376 / 76.002 |

Both pilots used 768 same-query comparisons on the full galleries, frozen CLIP ViT-B/32 and original KD-SBIR scoring on Colab T4. In the primary aligned variant mAP decreased on both. The fourfold adjustment amplified the loss and the reversed adjustment slightly improved Ext-2. This evidence does not support directing more noun probability toward the locations with positive contributions under the proposed covariance measurement. The experiment distinguishes this particular readout from the independent question of whether all forms of patch grounding improve retrieval. TU was reserved for methods supported by the exploratory comparisons.

[Definition and controls](protocol.md) · [Metrics, geometric checks and nominee accuracy](pilot-t4.json) · [Per-query tensors](paired-t4.zip).
