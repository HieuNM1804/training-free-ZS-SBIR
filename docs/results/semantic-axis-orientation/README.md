# How much of word retrieval survives a covariance-matched image representation?

Each entry is mAP / precision (%) on the same 1,536 sketch queries per split, with the complete gallery. This isolates Cross-Domain Semantic Retrieval before fusion with the other two modules.

| Transformation of the image embeddings used to read words | Ext-1 direct | Ext-1 diffused | Ext-2 direct | Ext-2 diffused |
|---|---:|---:|---:|---:|
| Native CLIP image and word axes | 76.14 / 79.55 | 82.36 / 81.66 | 72.32 / 71.22 | 74.52 / 73.23 |
| Shared covariance-preserving reflection 1 | 25.94 / 35.83 | 44.90 / 45.62 | 26.06 / 22.57 | 33.66 / 31.46 |
| Shared covariance-preserving reflection 2 | 23.21 / 30.92 | 38.03 / 30.59 | 28.13 / 26.64 | 34.16 / 33.16 |
| Shared covariance-preserving reflection 3 | 24.78 / 36.98 | 43.10 / 44.27 | 27.70 / 25.73 | 36.54 / 36.13 |
| Independent query and gallery reflections | 5.68 / 4.11 | 9.12 / 4.78 | 5.98 / 4.53 | 4.90 / 4.09 |
| Shared reflection 1 and per-input entropy restoration | 16.87 / 21.48 | 41.93 / 40.72 | 19.45 / 15.95 | 28.11 / 25.73 |

The native word channel exceeds all three shared reflections by at least 37.45 mAP points on Ext-1 and 37.98 on Ext-2 after diffusion. All image distances are preserved by each shared reflection around the gallery mean. The algebraic relative error for gallery covariance is below 3.7×10⁻⁸. These moment properties hold **before** applying the optional per-image entropy correction, which changes logit scale. The entropy-matched comparison restores the original entropy to within 2.6×10⁻⁵ nats per image and changes mAP by −40.43 and −46.42. The corresponding class-bootstrap intervals for native minus this control are [32.76, 49.59] and [35.28, 57.10] points (class-equal average). Two-view visual cosine alone scores 31.16 / 42.78 and 36.31 / 33.95.

The alignment of the image variation with the fixed CLIP text banks is important beyond gallery means, covariances and visual distances. This experiment cannot separate human-interpretable word identity from meaningful higher-order structure in CLIP, and it does not propose an accuracy improvement. It gives an empirical target for interventions in the global word score. [Frozen design and prior art](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Query-level results](paired-T4.zip)
