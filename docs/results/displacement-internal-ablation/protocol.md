# Internal ablations of the frozen displacement-consensus ranker

All new rows start from the same locked scorer, with additive weights 0.65, 0.35 and 0.65 and no hierarchy. One intervention changes one operation unless the row name explicitly combines changes. No retuning. Use frozen OpenAI CLIP ViT-B/32, the existing Open Images word bank, independent queries, label-free galleries, and the original KD-SBIR evaluator. Run full unseen Sketchy-Ext-1, Sketchy-Ext-2 and TU-Berlin-Ext (30 classes), seed 42, on Colab.

| Operation | Intervention |
|---|---|
| Gallery RGB diffusion | Identity in S alone, in the compact concept bank of R alone, in both pre-update and post-update GCWE scores alone, or in all three locations. Keep the regional concept support fixed. |
| Regional gallery visual prototypes | Use the original CLIP-to-text response with the same prior, active words and diffusion. |
| Regional gallery word prior weights | Keep the same 64 concepts, remove their log-prior offsets from the photo likelihood, and recompute the downstream prototype and bank. |
| Candidate-specific region selection | Average local likelihoods using the fixed scale prior, without conditioning the weights on the photo. |
| Regional gallery calibration | Infer regional weights from the raw photo likelihood, without dividing by the regional gallery mean. |
| Horizontal mirror in S and R | Read the native whole sketch and native crops. Keep the existing GCWE, including its two-view nominations. |
| Regional scales | Keep only the 9 small regions or only the 4 larger regions with equal prior weight within the surviving scale. |
| GCWE blank reference | Rank the same donors on displacement energy of the sketch itself, without subtracting mean-color canvas energy. |
| GCWE donor pseudocount | Remove the opposite-fold prior pseudocount from both counts. For a word of zero donor support the likelihood ratio remains neutral. |
| GCWE correction for uniform donor evidence | Keep the observed weighted evidence and pseudocount. Remove the denominator based on uniformly weighted nominees. |
| GCWE best-of-eight weighting | Give the whole donor mass to maximum-energy donors, sharing it evenly across ties. |
| GCWE gallery cross-fitting | Use all the nominated photos for each target, with the same best-of-eight rule and a full-gallery prior. Targets may also be their own donors. |

Reuse the archived term factorial and the archived displacement, coordinate and centering controls compared with the same full scorer. Rescore the locked full scorer for a paired reconstruction check. Save per-query metrics and report both KD-SBIR metrics, paired full-minus-intervention differences, and the class-bootstrap interval for mAP.
