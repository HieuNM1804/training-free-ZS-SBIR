# Leave-one-out of the highest confirmed four-component score

All masks use the locked parameters from the geometry-conditioned word study at 8edd3b8. The common input is the frozen OpenAI CLIP ViT-B/32, its original and mirrored views, the same 20,931 public nouns and the same full photo gallery. Each query is scored independently. Labels enter only the original KD-SBIR evaluator and the paired report.

| Removal | Intervention |
|---|---|
| RGB diffusion | Use the identity on global word-match scores before and after the geometry update. Rebuild the gallery weights for the 64 regional words with the same prior, heads and visual prototypes, and skip the RGB graph there as well. Retain gallery clustering for the independent hierarchical reader. |
| Regional readout | Zero the 0.35 contribution from candidate-specific regional word evidence. Retain the 0.65 global word coefficient and all views supplied to the separate hierarchical reader. |
| Hierarchical readout | Set its 0.4 additive gallery-cell contribution to zero. Retain both global and regional word scores and the geometry update. |
| Geometry-conditioned words | Set the cross-fold, blank-corrected spatial word update to zero. This also replays the published strong reference. |

The hierarchical contribution uses the original per-query standard deviation of the surviving pre-geometry score, exactly as in the frozen model. Thus an upstream removal propagates through this scale. The full reference and the geometry removal are compared with previously saved per-query AP and precision. The gallery index rebuilt before diffusion is checked by putting diffusion back and comparing with the original. Results use every query on Ext-1, Ext-2 and all 30 TU unseen classes. We report removal minus full in percentage points. The class bootstrap resamples classes equally, unlike the query-weighted official metrics.

Run `python -m tf_sbir.geometry_word_leave_one_out sketchy` then `python -m tf_sbir.geometry_word_leave_one_out tuberlin` in the project's Colab environment with the locked reference archives. All four removals are scored without parameter selection.
