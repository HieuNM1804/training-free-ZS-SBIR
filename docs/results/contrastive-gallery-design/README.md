# Designed comparisons increased candidate availability, but did not improve retrieval

Each method reads up to 200 distinct gallery photographs per query from a common pool of 1,200 closest visual nominees across two folds. The information design chooses pairs whose predicted winner would most distinguish plausible query nouns. It uses the same geometry verifier as the frozen reference. Fixed-cost controls compare random pairs, highest outcome entropy, nearest pairs, information under a gallery prior, and a simpler update using word frequencies on the selected images. The two exploratory panels each have 768 unseen sketches scored with the full gallery and KD-SBIR's original evaluator on Colab Tesla T4.

| Readout | Sketchy Ext-1 mAP@all / P@100 | Sketchy Ext-2 mAP@200 / P@200 |
|---|---:|---:|
| Earlier locked calibrated-geometry method | 86.00 / 85.15 | 77.44 / 76.06 |
| Information-designed pairs, integrated orientation likelihood | 85.39 / 84.23 | 75.48 / 74.17 |
| Random pairs, same likelihood | 85.59 / 85.15 | 76.31 / 75.06 |
| Nearest pairs, same likelihood | 85.42 / 84.64 | 76.53 / 75.15 |
| Designed pairs, original frequency readout | 86.20 / 85.67 | 76.20 / 74.62 |
| Random pairs, original frequency readout | 86.11 / 85.64 | 76.94 / 75.56 |

For the 96 / 137 strong-base complete top-200 misses, the information design included a correct photo on 80 / 133 queries. A random design from the same wider nominee pool included one on 81 / 134. The corresponding original visual top 200 contained a correct photo for 64 / 124. Wider proposal range explains most of the additional availability. The designed panels did raise the *count* of correct photos on these queries to 17.91 / 31.66 versus 14.34 / 25.25 with random panels. Designing for the information model still failed to translate this advantage into a better mAP.

Held-out pair orientations were easier to predict using the Bayesian update on these designed comparisons. Against the unchanged noun prior, its mean log score relative to chance rose from 0.0239 to 0.0684 on Ext-1 and from 0.0191 to 0.0513 on Ext-2. These results concern which photo the frozen spatial verifier prefers. They do not establish that this preference is correct for the underlying category. Average inferred reliability was 0.568 / 0.548; this is a model parameter and is not an empirical classification accuracy. Class-macro mAP differences relative to the locked method were −0.70 (95% class bootstrap −1.31 to −0.14) and −2.07 (−4.16 to −0.07). The new likelihood did not advance to TU. The next scientific question concerns what the spatial verifier measures about the original visual content.

[Design and sources](protocol.md) · [Candidate audit](nominee-audit-t4.json) · [All paired comparisons](pilot-t4.json) · [Per-query results](paired-t4.zip).
