# Does the physical size of a depiction cause class-retrieval errors?

This is a diagnostic of two existing frozen readers, native OpenAI CLIP ViT-B/32 and the published-in-repo Semantic Dot 601 configuration. The classifier, gallery, text prompts and similarity definitions remain fixed. Each full unseen Sketchy split is scored with its original KD-SBIR evaluator. Using the other drawings and the known white drawing surface enables a retrospective experiment, and would be privileged information in deployment.

## Intervention

Take the native 224-pixel input. Define the center of ink by its first moment and the radius by its RMS distance to that center. Use continuous grayscale darkness on the known white canvas. All nonwhite pixels contribute. For each source photograph find the smallest observed radius among its nonempty drawings. For each drawing isotropically shrink the entire canvas to the ratio of that smallest radius to its own radius, rounded to the nearest whole raster size. Every pixel participates in resampling.

Score three conditions. Original uses the cached native views. Small places the reduced image on a 224-pixel white canvas so its center of ink stays at its original position within rounding. Restored expands the same reduced raster back to 224 pixels. Small and restored see exactly the same reduced-raster information before display. The smallest drawing supplies an identity control. A blank image, if encountered, receives an identity intervention and is excluded only from the source-level comparison of extents.

For each source let L and S be the drawings with largest and smallest original radius. The natural difference in average precision, AP(L,original) − AP(S,original), equals the sum of three signed effects:

1. AP(L,original) − AP(L,restored) captures the net effect of the downsampling and re-expansion loop.
2. AP(L,restored) − AP(L,small) compares display extent at identical reduced-raster information. Interpolation into the final grid is part of the display operation.
3. AP(L,small) − AP(S,original) compares different drawings at matched intended radius, with their own positions, geometry, stroke style and discretization.

Use all subjects with at least two distinct nonzero radii for this paired calculation. Average subjects within class and classes equally for effects and seed-42 class bootstrap intervals. Also report ordinary query-weighted mAP and precision over **all unseen queries and the complete gallery**. Report the changes in the existing, preselected rhinoceros example on its fixed two photographs separately. The selected example does not decide the population outcome.

If the two split intervals for a reader both support a positive display effect, conclude that this controlled reduction in extent changes actual retrieval for that reader. Compare the magnitude with the remaining drawing difference and with the resampling control. This experiment does not establish a gallery-side remedy or the value of a scale rule at inference. Scaling and centering were already used in the original Sketchy paper. [Sangkloy et al., SIGGRAPH 2016](https://faculty.cc.gatech.edu/~hays/tmp/sketchy-database.pdf)
