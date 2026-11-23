# Locked development choice for full Sketchy and TU

The two exploratory 768-query Sketchy runs used scorer `90a6f5c3a28c37dc79e9fe5a2f55cd3426a4bfaf`. Among the tested updates to the strong reference, using excess spatial agreement over the same query's mean-color canvas at m=8 gave the highest equal-split mean mAP. It gave 85.999 / 85.151 on Ext-1 and 77.435 / 76.059 on Ext-2. The original corresponding strong scores were 85.964 / 85.579 and 75.744 / 74.598. Only the Sketchy pilot chose the configuration. Previous TU experiments tested a one-donor method; this new word-distribution update has not been run on TU at the time of this lock.

For the full replications retain visual top 200, 49 original final-layer ViT-B/32 projected patch vectors, original sketch and per-channel mean-color canvas, the fixed two-fold gallery partition seed 69313, frozen CLIP logit scale, score(real sketch geometry) minus score(blank geometry), the maximum of m=8 independent rank draws, one gallery-prior photo's equivalent of smoothing, the query's original two-view public-word probability, and the original 0.65 semantic-channel coefficient. No source-style partition enters the scorer. Reconstruct all other strong reference terms. The same configuration is the primary for both full Sketchy splits and TU as a held-out dataset. Keep it even if the full results differ from the pilot. All 30 TU unseen classes are required.

Controls set the numerator to raw patch geometry, permute candidate patch locations in both the real and the blank readout, preserve the real geometry but subtract a different nominated image's blank score, or order donors by visual cosine, direct words, or the existing strong score. Shuffle indices only within the eligible donor fold for the mismatched-blank control. These are controls for interpretation and cannot change the locked primary. Report both metrics, paired differences and uncertainty across classes. Use original KD-SBIR, full galleries and independent queries throughout. Query IDs and original baselines are checked against each cached shard. Full runs save intermediate per-query statistics to survive interruption.

| Exploratory comparison, mAP / precision in % | Sketchy Ext-1 | Sketchy Ext-2 |
|---|---:|---:|
| Strong paired base | 85.964 / 85.579 | 75.744 / 74.598 |
| Corrected word evidence, real minus blank, 8 draws | 85.999 / 85.151 | 77.435 / 76.059 |
| Same, 2 draws | 86.157 / 85.581 | 76.469 / 75.145 |
| Real geometry, 8 draws | 85.694 / 84.941 | 76.873 / 75.313 |
| Shuffled geometry, 8 draws | 85.448 / 84.823 | 76.600 / 75.510 |
| Visual cosine chooses the donors | 85.318 / 84.932 | 75.915 / 74.565 |
| Strong score chooses the donors | 85.917 / 85.208 | 76.281 / 74.865 |

Numbers in these two columns are exploratory, with overlapping underlying Sketchy data. Comparing with the strongest donor control and correctly paired sham on full data is material to an algorithmic novelty claim.
