# Certifying a rank against deleting one latent sketch word

The supporting-photo model predicts a query-dependent value for each word and reads it a second time. Blanking out a term only in the final dot product would miss the change in all other words' gallery responses. Here an intervention sets the sketch likelihood of a single public word to zero. It changes neither the gallery profiles nor the input pixels. Keep the initial scale of the sketch likelihood across interventions. Positive rescaling has no effect on the ranking of any fixed intervention.

Write $c=\sum_i B_i$, $M=\operatorname{diag}(c)^{-1}B^\top B$ and $D_{iw}=B_{iw}^2/(c_w-B_{iw})$. The raw score for candidate $i$ under distinct photo identities is $S_i(l)=l^\top A_i l$ with $A_i=\operatorname{diag}(B_i+D_i)M-B_iD_i^\top$, where row profiles are interpreted as column vectors in the outer product. The generative expression in the feedback protocol shows that every coefficient is nonnegative, even though this compact representation contains a subtraction.

For any coordinate $k$, its deletion decreases a fixed photo pair's margin by
$$
l_k[(A_u-A_v)l+(A_u-A_v)^\top l]_k-l_k^2(A_u-A_v)_{kk}.
$$
The diagonal removes the contribution counted in both directions. Computing every coordinate at once only needs products with the existing gallery index, its transpose and the squared column norms. The formula identifies the exact most damaging single word for a specified pair in real arithmetic. The word with greatest standalone sketch likelihood and a direct-path-only attribution are empirical controls.

There is also a global bound. Let $u$ and $v$ be the original first and second photos. If deleting $k$ reduces the score of $u$ by less than $S_u-S_v$, no photo can overtake $u$, since all other scores can only decrease. Recompute every remaining candidate deletion against the **entire** gallery. This certifies invariance of the original photo, or finds every single-term counterexample, for this intervention. Switching between photos of the same class is a different result and can be counted after the rankings are produced.

## Predeclared checks

Evaluate the top-two pair on every sketch. Actually delete the selected term for every attribution rule, recompute the full gallery response and measure the real margin, rank change, and maximum numerical discrepancy. A normalized tolerance of $2\times10^{-5}$ of the original winner's score identifies numerically ambiguous pair margins and makes global pruning conservative. For the all-gallery/all-word test take 4 sketches per class using seed 79307. Every word surviving the bound is rescored, including for queries with a very large number of survivors. For one predetermined, real, full-gallery query per split, independently evaluate all 20,931 word deletions including those excluded by the bound. Test both numerical precision and whether a screened-out word ever changed the first photo. The sample size constrains the inference about robustness frequency across an entire benchmark.

The scientific claim concerns exact evidence-path accounting and a certified screening bound for this specific nonlinear word–image retrieval model. Query reconstruction, textual explanations and counterfactual rank changes are established research topics. Kim et al. ground visual classifier counterfactuals in textual concept space; *CREDENCE* identifies adversarial document edits in a ranking model. These are related works, not extra components. [Kim et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_Grounding_Counterfactual_Explanation_of_Image_Classifiers_to_Textual_Concept_Space_CVPR_2023_paper.pdf) · [CREDENCE](https://arxiv.org/abs/2302.04983) · [Model being audited](GALLERY_LEXICAL_FEEDBACK_PROTOCOL.md)
