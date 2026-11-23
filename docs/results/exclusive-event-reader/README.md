# Reading sketch evidence on independent reference photos

This diagnostic distinguishes selecting a competing photo from deciding whether that photo fits the sketch better. It inherits the eight-rival shortlist in gallery-concept-exclusion. Panel 1 selects the candidate with the largest minimum of the within-photo and neighbor co-mention deficits. Panel 2 supplies the independently measured reference events for the selected pair.

For each reference photo take the inner product of its public-word posterior and the two-view sketch posterior. Compare the mean score on reference photos that recognize the competitor with the mean for photos recognizing the incumbent. The exclusive variant drops references recognizing both. Each conditional mean receives one observation at the unconditional mean. A proposed switch must exceed the incumbent's original log-score advantage. The two rules use the same photo pair for each query.

| Fixed / damaged top-1 decisions | Ext-1 (7,085 eligible) | Ext-2 (7,932 eligible) |
|---|---:|---:|
| All eligible, marginal evidence | 55 / 86 | 137 / 92 |
| All eligible, exclusive evidence | 59 / 95 | 149 / 107 |
| Positive deficits on panel 2, marginal | 33 / 69 | 105 / 61 |
| Positive deficits on panel 2, exclusive | 34 / 70 | 107 / 63 |

Removing overlapping events produced no consistent gain in net corrected decisions. The original margins are from the frozen reference ranker. The shortlist excludes 4,096 reference photos, so this diagnostic cannot be read as a standard full-gallery metric. All word events are CLIP readings. Labels are used for evaluation, and all sketch decisions are independent.

Code at scoring time was `e69485d9522edf25def3e9a6e383b9dfc26e6be7`. Run on Colab L4 using `python -m tf_sbir.gallery_concept_exclusion` on this branch.
