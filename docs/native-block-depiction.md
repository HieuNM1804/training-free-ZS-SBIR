# A held-out test of shared channel coordinates across depiction

This diagnostic uses the native OpenAI ViT-B/32 in inference mode. It measures a necessary empirical premise of using the input of one block as a coordinate reference for its output. No retrieval method or tuned parameter enters the diagnostic.

## Fixed cohort and computation

Use the two Sketchy split manifests from commit `c7ff4b9`. They identify 32 source photos per category and two actual human drawings per photo, sampled with seed 42. Within a split, include every subject in those manifests. Use the original unreflected 224-pixel RGB view for every image. Let X denote the 50 tokens before the sixth visual block and D denote its update, B(X) − X. The residual subtraction prevents the identity shortcut through the block from counting as evidence. The transformer has access to the entire image. The patches held out below are held out from estimation of the channel map.

Use one fixed seed-42 partition of the 49 patch positions into 24 and 25 positions. Exchange the two halves in a second fold. In each fold, the fitting rows include CLS and the fitting patch positions. The held-out rows contain only spatial tokens. Estimate the identifiable partial Procrustes map from the fitting rows of the sketch to those of the paired photo. To test it, map the sketch updates at held-out positions into the photo coordinates. Restrict the photo updates to the same identifiable target subspace. Subtract each image's mean update across its own held-out patch positions and calculate a single Frobenius cosine. This measures the spatially varying response. Report the uncentered value as a sensitivity check.

Compare three ways of reading the same updates. The aligned readout uses the actual fitting-row correspondence. The identity control simply reads the sketch in the photo's identifiable target subspace. A second control permutes fitting-patch correspondence in one fixed seeded cycle, leaving CLS fixed. This preserves all fitting rows, their norms, and the two fitted subspaces. Average the two folds and both drawings within a source. Never use these pair-specific maps for actual gallery retrieval.

## Predetermined decision

1. For the actual source photo, require a positive aligned-minus-identity difference and a positive aligned-minus-permuted difference on the held-out centered response, on both splits.
2. Verify category specificity on different subjects. The positive is another photo from the same class, offset by 16 in the fixed 32-subject order. Contrast this with the mean of four photos of other classes per subject, also offset by 16. Cycle the four negative class offsets over subject indices to cover every other class. Require the aligned same-class margin over these negatives to exceed the identity margin on both splits. This protects against rotation rescuing only the exact drawing-to-source task.
3. For each required difference use its lower endpoint from a two-sided 95% paired bootstrap over query classes (10,000 draws, seed 42), conditional on the fixed gallery. If either split fails any of these comparisons, report which premise lacks support and stop before implementing the ranker. Uncentered updates, within-class source discrimination, and the analysis excluding the four shared classes are descriptive. No layer, patch partition or parameter sweep is used.

Successful diagnosis would authorize a separate test of a fixed common gallery frame, which has less information than the paired diagnostic. Semantic Dot and KD-SBIR full-gallery evaluation are required for that retrieval test. TU-Berlin stays outside the diagnostic.
