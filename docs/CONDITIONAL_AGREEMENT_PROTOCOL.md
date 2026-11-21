# Agreement of concept explanations under gallery diffusion

## Question

The frozen baseline reads a sketch as likelihoods on 20,931 Open Images terms. For a gallery photo $j$, CLIP gives a probability vector $p_j$ on the same terms using photo prompts and a uniform prior. The seed is the dot product of these two vectors. The row-normalized RGB operator $T$ averages these scalar seeds. It has 16 neighbors, Gaussian bandwidth coefficient 0.36 and two updates. All experiments use the same frozen OpenAI CLIP ViT-B/32 checkpoint, individual queries and unlabeled photo gallery.

Can the graph support an incorrect category because photos supporting a sketch do so through different concepts? An archived Ext-2 windmill query lost its correct first result to a tree when diffusion was turned on. A seagull query improved under the same operator. Neither case identifies its cause. A query-conditioned semantic graph is already a plausible alternative, with query-weighted reranking in prior work and in this repository.

## Proposed experiment

Let $l$ denote the nonnegative sketch likelihood vector and $e_j=l^T p_j$. A supporting photo assigns conditional responsibility $r_j(w)=l(w)p_j(w)/e_j$ to word $w$. At candidate $i$, draw two photos independently with replacement with weights proportional to $T_{ij}\sqrt{e_j}$. Define (R_i$l$) as the expected Bhattacharyya affinity of the two responsibilities. Both draws use the same row of the same operator. Distinct image content and statistical independence of the CLIP errors are not assumptions.

For implementation set $U=T\sqrt P$, with one photo per row of $P$. All powers below are elementwise. The numerator $N=(U^2)l$ and denominator $D=(T\sqrt e)^2$ give $R=N/D$. This follows by expanding the squared norm of the weighted mean of the square roots of the responsibilities. The triangle inequality gives $0\le N\le D\le Te$. A sketch activating a single word gives $R=1$, even when its seed magnitudes vary. If all gallery donors have identical conditional responsibilities, $R=1$. A flat query gives context agreement $R_0=U^2\mathbf 1$. These statements apply to nonzero evidence.

Compare two candidates, conditional agreement $s=(Te)R$ and agreement relative to the flat-query reference $s=(Te)R/R_0$. Each uses one additional scalar signal through the original operator at query time. Controls are direct lexical score, original diffusion, root averaging of the scalar score $D$, root averaging of the lexical distributions $N/R_0$, and the gallery bias $(Te)/R_0$. A diagnostic independently changes **only** the names of the word coordinates in one half of the donor gallery after conditioning on the sketch. It preserves every scalar seed, graph weight, conditional entropy and the values delivered by ordinary diffusion. The half is selected by a fixed random permutation. The full formula and control use all terms.

Use all queries and the full galleries on Sketchy Ext-1 and Ext-2. Select between the two specified candidates by mean official mAP across the two splits, then mean precision. Lock the decision before measuring either on TU-Berlin. Score TU-Berlin in all 30 classes to test transfer even if the method fails to improve Sketchy. Report paired differences, class-cluster bootstrap intervals, top-1 losses and rescues, and the agreement at a correct and an incorrect candidate near the top of the same ranking. Record the reference rerun and full KD-SBIR metric replay on identical rankings. Tuning on Sketchy unseen is exploratory. A gain replicated by scalar pooling, by a query-independent correction, or by shuffling word identity weakens the proposed explanation. A higher ranking for a single counterexample is not sufficient evidence.

## Prior work

The baseline uses FreeDom's vocabulary bridge, the shared-concept interpretation motivated by Bayesian Sets, and Wang et al.'s gallery diffusion. The conditional agreement statistic is a deduction from this model. The Bhattacharyya kernel and generalized mean pooling are established operations. The prospective contribution is the separation of disagreement about *which concept* supports a sketch from variation in *how much* support is present, with an identity intervention that holds all original retrieval scores fixed. The venue target is an aspiration pending evidence and comparison with equivalent methods.

- ICFRR uses test-time gallery reranking on these benchmarks. Its authors also evaluate domains beyond sketch and photo. [Hudson and Smith](https://arxiv.org/abs/2303.17703)
- Wang et al. derive the cross-diffusion update for metric fusion. Here the selected operator uses one gallery view. [CVPR 2012](https://pages.ucsd.edu/~ztu/publication/cvpr12_CrDP.pdf)
- FreeDom maps a query through a discrete vocabulary. [WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Efthymiadis_Composed_Image_Retrieval_for_Training-Free_Domain_Conversion_WACV_2025_paper.html)
- Probability product kernels include the square-root overlap and generalized mean pooling subsumes the numerator's offline pooling. Both are relevant alternatives when assessing originality. [Jebara et al.](https://www.jmlr.org/papers/volume5/jebara04a/jebara04a.pdf) · [Radenović et al.](https://arxiv.org/abs/1711.02512)
- STiTch uses semantic transition and transport for composed image retrieval. [CVPR 2026](https://arxiv.org/abs/2605.21261)
