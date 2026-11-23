# Order constraints from competing public nouns

The two-noun reader sometimes selects a better first result, but its margins are poorly calibrated as posterior odds. This pilot instead treats the judgments as purely ordinal constraints. From the frozen word-plus-RGB-diffusion top 200, take the first 2 or at most 8 distinct public names and their highest-ranking photos. For every pair ask which concept the sketch places in the foreground under both possible mention orders. Isolated nouns, mention position and the orthogonalized role give matched controls.

Within this small complete comparison graph, rank the hypotheses by their net number of pairwise wins, breaking equal totals by native rank. Weighted isotonic regression projects the native group-peak log scores onto that order with the minimum squared displacement. Weights equal the original positive retrieval mass of each named group. Apply the resulting log shifts to group members, then restore total selected mass. Native ordering within each group and scores outside the selected groups are preserved up to numerical ties. A control applies the procedure only to individual representative photos. The program checks mass conservation and the projected order. No hard contour or trained prompt is used.

| Reading and propagation | Ext-1 mAP@all / P@100 | Ext-2 mAP@200 / P@200 | Top-1 fixed / damaged on Ext-1 and Ext-2 |
|---|---:|---:|---:|
| Frozen word-plus-diffusion reference | 83.00 / 82.85 | 74.07 / 72.78 | 0 / 0 and 0 / 0 |
| Two groups, isolated nouns | 83.02 / 82.89 | 74.08 / 72.81 | 9 / 5 and 18 / 9 |
| Two groups, counterbalanced role | 83.01 / 82.88 | 74.09 / 72.78 | 7 / 4 and 16 / 3 |
| Eight groups, isolated nouns | 83.04 / 82.96 | 73.76 / 72.67 | 12 / 9 and 29 / 9 |
| Eight groups, counterbalanced role | 83.01 / 82.87 | 73.93 / 72.67 | 13 / 12 and 28 / 8 |
| Eight leaders only, counterbalanced role | 83.00 / 82.82 | 74.00 / 72.78 | 3 / 2 and 7 / 3 |

Projecting eight group peaks affected groups containing an average of 352 / 322 photos on queries whose hypothesis order changed. Improved first results did not imply an improved full ranking. Compared with isolated nouns, the eight-group role readout moved mAP in opposite directions on the two pilots; its paired class-bootstrap intervals include zero on both splits. The two-group intervention produces mAP changes under 0.02 points versus native. This experiment does not establish that noun-role votes warrant reordering semantic neighborhoods.

The pilot used 768 uniform sketches per split with seed 87243, full original galleries and original KD-SBIR replay for all 11 rows. The two sets of selected names are public CLIP readings inferred from unlabeled photos. No benchmark label participates in the projection, selection or vote. Run `python -m tf_sbir.ordinal_foreground_ranking` at `3c9ed447b66bc6ae8fa96eab35a889c9fd6b6138` on Colab T4. No tuning on TU-Berlin.

The linguistic controls draw on minimal word-order pairs and CLIP's first-mention bias. Isotonic regression is an established ranking tool; the untrained projection tested here preserves retrieval mass and uses the counterbalanced comparisons as its proposed order. [Thrush et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Thrush_Winoground_Probing_Vision_and_Language_Models_for_Visio-Linguistic_Compositionality_CVPR_2022_paper.pdf) · [Abbasi et al., CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Abbasi_CLIP_Under_the_Microscope_A_Fine-Grained_Analysis_of_Multi-Object_Representation_CVPR_2025_paper.pdf) · [Blondel et al., ICML 2020](https://proceedings.mlr.press/v119/blondel20a.html)
