# Minimal S/R/G without RGB diffusion

The primary score is 0.65 S + 0.35 R + 0.65 G. Freeze the recipe before evaluation and retain the previous weights. Use the original KD-SBIR ranking metrics, independent queries, frozen OpenAI CLIP ViT-B/32, the existing domain-specific text encodings of 20,931 public Open Images words, and seed 42.

- **S** is the log dot product of whole-image word posteriors. The sketch posterior averages native and reflected whole-image predictions, as in the archived direct semantic control. The gallery supplies the native whole-photo reading.
- **R** is the log dot product of the uniform means of the 49 native CLIP patch word posteriors on each side. Apply the word softmax to each patch before averaging. All images have the same weight for each patch. No crops, gallery prior, visual prototypes, candidate-specific selection, or gallery calibration enter R.
- **G** is the log difference between the direct whole-image word overlap after and before the locked GCWE word update. Keep displacement concentration, both canvas readings, the frozen donor weighting, donor normalization, and opposite-fold assignment from the archived method.

Use direct overlap in all three branches. Compare the primary row with S+G, S+R and S using the same weights. Reuse the archived full scorer and the archived scorer with RGB diffusion off at all three sites. Replay S against its historical direct score to check feature and evaluator consistency. Archive per-query metrics, paired mean differences, and the bootstrap interval over classes.

Evaluate the complete unseen Sketchy Ext-1 and Ext-2 splits and all 30 unseen TU-Berlin classes on Colab. No parameter selection on TU-Berlin.
