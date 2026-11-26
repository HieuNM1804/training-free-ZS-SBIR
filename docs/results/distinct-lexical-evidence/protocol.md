# Distinct lexical matches across sketch views

## Motivation

Use findings from the full Ext-1 run of the preceding S-only audit to formulate this exploratory experiment. On hard errors of direct S, a single word accounted for a median of 83.7% of the overlap with the strongest incorrect photo and 60.0% with the strongest correct photo. Restricting the query to 64 words discarded more overlap from the correct photo and reduced mAP. The proportion of overlap disputed when the photo prompt changed did not usefully identify those incorrect matches. Test instead whether two different word matches can corroborate a sketch-to-photo association.

## Score

Let (q_0,q_1) be the two existing sketch posteriors before averaging, and (p) the native photo posterior. Let (x_w=q_{0w}p_w) and (y_w=q_{1w}p_w). The degree-two score is the square root of

[
(sum_w x_w)(sum_w y_w)-sum_g(sum_{win g}x_w)(sum_{win g}y_w).
]

The expression is the mass of obtaining a lexical match in each of two categorical draws and assigning the two matches to different word groups. Independence refers to sampling conditional on the computed posteriors. Existing views and related words can still share CLIP errors. No independence of actual visual content is assumed. An equivalent expression subtracts within-group cross-products. With singleton groups this is a two-view ANOVA interaction. Identical views reduce to the square of ordinary overlap minus the second moment of word contributions. Use the square root to return to the units of single-view overlap.

Use (g) induced solely by word embeddings. Pair two words if they are each other's first neighbor in both existing domain-specific text banks. Keep all remaining words as singletons. These pairs indicate strongly similar CLIP embeddings and need not be synonyms. Aggregation within a group is used only to remove repeated evidence. It never lets two differently named words create a first-order collision. Disjoint pairs give an exact evaluation using ordinary batch matrix products, with one additional gallery feature for each pair. Verify against direct FP64 multiplication.

## Controls and transfer

Compare the same ten fixed variants on both complete unseen Sketchy splits. Use either the original photo distribution or the already existing gallery-only MLE concept prior. On each photo distribution compare arithmetic pooling (S), the product across views with no exclusion, the product allowing confirmations only from distinct words, lexically grouped confirmations, and a seed-42 permutation of the partners preserving which words are paired and the group sizes. The arithmetic and ungrouped product controls separate the effect of original view pooling, requiring two views, and requiring distinct evidence. Real versus permuted pairs probes whether lexical structure is useful.

Choose the best of the distinct-word and distinct-group variants by equally weighted full Sketchy mAP and then precision. Freeze the choice before evaluating TU-Berlin with all 30 classes. All variants are specified before TU-Berlin. Use the full original KD-SBIR evaluator on Colab T4 with frozen OpenAI CLIP ViT-B/32 and seed 42. No diffusion, G, R, extra image transformation, class-label list, or cross-query information enters scoring. The comparison of two unseen datasets does not introduce other domain pairs.

The established feature-interaction algebra comes from the ANOVA kernel used in classification and regression by [Blondel et al., ICML 2016](https://proceedings.mlr.press/v48/blondel16.html). Reciprocal neighbors have been used to refine person re-identification rankings by [Zhong et al., CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.html). The proposed transfer uses two CLIP readings to corroborate a vocabulary-space match, with a separate test that pairs of names should count as one evidence source. The vocabulary bridge itself is inherited.
