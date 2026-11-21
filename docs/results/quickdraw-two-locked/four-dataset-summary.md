# Locked configurations across four datasets

Both configurations were scored with the KD-SBIR retrieval metric implementation. The QuickDraw run used the existing locked parameters without tuning. Scores are percentages.

| Dataset | Metric cutoffs | Public semantic translation mAP / precision | Cross diffusion map mAP / precision |
|---|---|---:|---:|
| Sketchy Ext-1 | mAP@all, P@100 | 75.96 / 78.87 | 81.86 / 81.33 |
| Sketchy Ext-2 | mAP@200, P@200 | 71.60 / 70.66 | 74.20 / 72.68 |
| TU-Berlin, 30 classes | mAP@all, P@100 | 73.68 / 77.00 | 79.16 / 80.24 |
| QuickDraw, 30 classes | mAP@all, P@200 | 36.62 / 42.36 | 45.69 / 46.31 |

## Locked configurations

- **Public semantic translation:** nearest public-word support, mass 0.5, local bandwidth, no shuffle.
- **Cross diffusion map:** 16 RGB neighbors, bandwidth 0.36, two iterations, normalized mass, no semantic shuffle.

QuickDraw evaluation covered 91,491 queries and 56,407 gallery images. Both methods completed every query. The independent KD-SBIR metric check differed by at most `2.3842e-7` for mAP and `5.9605e-8` for precision. The run used an NVIDIA L4 and checkpoint SHA-256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.

The QuickDraw scores and full precision are in [`scores.json`](scores.json). The Sketchy and TU-Berlin scores are from the existing locked evaluation records in this branch.
