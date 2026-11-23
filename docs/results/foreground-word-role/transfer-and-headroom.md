# Transferring the frozen foreground readout and measuring its mAP ceiling

Before viewing TU retrieval, commit `f8632868ffc83da4ca842f8365140df18b541e2f` locked all readouts and controls. The original KD-SBIR function scored every one of the 2,130 sketches against all 27,262 photos from all 30 unseen TU classes on Colab Tesla T4. The same two-view public-word and RGB-diffusion baseline was used on all datasets.

| Dataset | Wrong first results fixed / correct ones lost by role | By isolated words | By mention position alone | Native mAP / P | With role mAP / P |
|---|---:|---:|---:|---:|---:|
| Sketchy Ext-1 | 131 / 59 | 103 / 52 | 90 / 58 | 82.6970 / 82.2704 | 82.6972 / 82.2704 |
| Sketchy Ext-2 | 160 / 86 | 214 / 160 | 166 / 143 | 75.1522 / 73.5939 | 75.1252 / 73.5939 |
| TU-Berlin | 21 / 4 | 19 / 6 | 30 / 10 | 80.1124 / 81.3822 | 80.1141 / 81.3822 |

Only 71 TU queries nominated a pair in which exactly one candidate was relevant. Foreground role achieved 34 correct choices versus 30 with isolated nouns and 37 with position alone. Class-balanced precision of the two possible choices averages to 0.577 with role and 0.484 with position on this conditional test (balanced accuracy over the two orientations). Requiring both photos and the sketch to justify the role left just six informative TU decisions. The name-position control prevents attributing the net TU top-1 gain uniquely to foreground semantics.

## An upper bound on this intervention

After generating both allowed rankings independently of labels, replay each with the original evaluator. Taking the larger of their two per-query APs gives a retrospective bound on any selector confined to that candidate pair. No label enters the ranking algorithm. These calculations were repeated at `f56d90ecfc15636be25eb4833f46d4f3f8124114`; small T4 replay differences can affect near ties. Both choices always keep the same membership for top 200 and in the measured runs also gave identical precision per query.

| Dataset | Native top-1 errors | Those with any correct photo in top 200 | Correct alternative among first two distinct named representatives | Highest attainable mAP improvement, percentage points |
|---|---:|---:|---:|---:|
| Ext-1 | 2,676 | 683 | 205 | +0.00143 |
| Ext-2 | 3,732 | 1,556 | 411 | +0.07163 |
| TU-Berlin | 431 | 175 | 54 | +0.00705 |

This explains why better pair preferences cannot close a multi-point mAP gap by exchanging only the two first named nominees. Adding representatives expands the opportunities, but on Ext-2 even 32 first named representatives recover just 1,426 of the 3,732 native top-1 errors. The remaining source of error includes relevant photos absent from the top 200. See `attainable-kd-results-t4.json` for the signed gains and losses and each candidate budget. The earlier locked TU transfer scores are in `tuberlin-full-t4.json`. TU tests transfer between datasets within sketch-to-photo retrieval.
