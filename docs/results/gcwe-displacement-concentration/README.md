# Confirmation on 1,536 queries per split

| GCWE donor statistic | Ext-1 mAP@all / P@100 | Ext-2 mAP@200 / P@200 |
|---|---:|---:|
| Original | 84.962 / 84.525 | 77.516 / 75.190 |
| Within-part displacement concentration | 85.190 / 84.840 | 78.246 / 75.999 |
| Scrambled photo coordinates | 84.926 / 84.566 | 78.182 / 75.542 |
| Before grouping by displacement | 85.030 / 84.458 | 78.103 / 75.336 |
| Raw displacement concentration | 85.228 / 84.723 | 78.330 / 76.004 |
| Without GCWE | 84.690 / 84.357 | 77.455 / 75.408 |

The primary exceeded the four locked retrieval controls on both splits. Its paired class-bootstrap mAP intervals relative to original GCWE span zero on both. The raw concentration exceeded the primary by 0.038 and 0.084 mAP points. The test supports taking the concentration statistic to full evaluation and provides no positive evidence for subtracting independent-endpoint occupancy. The confirmation excludes the prior seed-permutation block within each split. The full report also excludes the union of selection IDs across splits.

[Protocol](protocol.md) · [All Ext-1 comparisons](sketchy_1.json) · [All Ext-2 comparisons](sketchy_2.json) · [Query-level outputs](paired-T4.zip)
