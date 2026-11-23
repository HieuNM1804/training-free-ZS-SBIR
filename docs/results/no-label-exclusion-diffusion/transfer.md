# Locked transfer on TU-Berlin-Ext

The same three scorers ran on all 2,130 sketches and the full 27,262-photo gallery, with all 30 unseen classes present on each side. Code and protocol were frozen at `6b4842584b1f22877daa6161bc745803dcd5705e` before TU was evaluated. The Colab L4 replayed the original KD-SBIR evaluator for every query in each row. Images and sketches use frozen CLIP ViT-B/32 and the approved 20,931-word public dictionary.

| Frozen row | mAP@all | P@100 |
|---|---:|---:|
| Strong two-view reference | 80.59 | 81.32 |
| Reproducible negative association | 80.83 | 81.32 |
| Disjoint top-eight word lists | 81.40 | 80.93 |

Negative association improves the query-averaged mAP by 0.24 points with a P@100 change of −0.001 points. The paired mean across classes is +0.24 mAP points (class bootstrap 95% interval −0.08 to +0.62). Different classes account for 99.95% of the vetoed pairs by count. The simpler disjoint-words control records the highest mAP and gives up 0.39 P@100 points versus the reference. The full results are in `tuberlin-official.json`. TU measures transfer between datasets in the same sketch-to-photo direction.
