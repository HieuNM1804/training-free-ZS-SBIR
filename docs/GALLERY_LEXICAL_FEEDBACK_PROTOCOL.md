# Query evidence reconstructed from a word–gallery index

For a candidate photo $i$, let $B_i$ be its public-word distribution after applying the original RGB diffusion, $B=TP$. Normalize each row to remove rounding drift. Let $l$ be the max-scaled likelihood from a single sketch, $m=\operatorname{mean}_i B_i$ the background word distribution, and $e=Bl$ the original retrieval scores. This factorization lets us ask what word distribution the initial results predict. No other sketch enters the model.

## Two hypotheses from the same prediction

With a uniform prior on photos, compute the first image posterior $a^{(1)}=e/\sum_i e_i$. Its predicted word distribution is $p=B^\top a^{(1)}$. The query-compatible distribution under the background is $r\propto m\odot l$. The question is whether a word overrepresented in $p$ should be discounted because the current answer already explains it, or upweighted because other photos support that interpretation.

**Inverse inference.** Find a mixture of gallery profiles approximating $r$ by minimizing $D_\mathrm{KL}(r\|B^\top a)$ over photo weights in the simplex. The usual multiplicative mixture update gives $a'_i=a_i\sum_w B_{iw}r_w/(B^\top a)_w$. At the uniform prior the denominator is exactly $m$, so the first update gives the baseline ranking, up to floating point. The second and fourth iterates are prospective choices. Freezing the predicted distribution at the background for all updates only raises each original score to a power and preserves every strict ordering. Stronger reconstruction by itself does not imply stronger retrieval.

**A second supporting photo.** Let $h_w=p_w/m_w$. Up to a factor common to a given sketch, $h_w$ is the expected original score of a photo sampled in proportion to its support for word $w$. An image can receive support for word $w$ if that word is plausible for the sketch and a *different* gallery image supporting it also fits the sketch. With the candidate excluded from this second draw, rank by
$$
s_i=\sum_w B_{iw}l_w
\frac{\sum_{j\ne i}B_{jw} e_j}{\sum_{j\ne i}B_{jw}} .
$$
Its global sketch scale is immaterial. Unlike a list of proxy documents, this construction conditions the witness population on the particular word explaining the candidate. The two photo identities differ; their diffused profiles can share original neighbors. The version admitting the same identity is $B(l\odot h)$. Independent word draws from the candidate instead give $(Bl)\odot(Bh)$. This is a matched control for reusing the sketch twice. The two models can give different rankings because the expectation of a product depends on the covariance within each candidate's word distribution.

## Promotion and falsification

Use every query and the original KD-SBIR cutoffs. Freeze the highest mean Sketchy unseen mAP among inverse step 2, inverse step 4 and the distinct supporting photo; break ties by mean Sketchy precision. Report all on TU-Berlin after the freeze. Also evaluate the original propagation, its factorization, simple co-occurrence translation $Bh$, sharpened sketch $B(l^2)$, the two independent word draws, a shuffle of word identities in $h$, and the inverse residual alone. Report class-bootstrap paired intervals, top-1 repairs and losses, and the change in lexical KL. Test the exclusion formula against an explicit, independent FP64 calculation that drops each tested photo from both sums. Small TF32 discrepancies can change the ordering of near-tied gallery entries.

Use the score-matched controls to decide what deserves a novelty claim. A benefit reproduced by translation or independent word draws cannot be attributed to requiring the same word. A negligible difference with/without exclusion limits the claim about self-influence. A lower KL without a retrieval gain is a negative result for the inverse model's target. The same RGB backbone, both domain text banks and the original gallery operator are frozen throughout.

## Related research

- The word bridge and the gallery kernel come from [FreeDom, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Efthymiadis_Composed_Image_Retrieval_for_Training-Free_Domain_Conversion_WACV_2025_paper.html) and [Wang et al., CVPR 2012](https://pages.ucsd.edu/~ztu/publication/cvpr12_CrDP.pdf).
- The inverse hypothesis uses ordinary mixture inference with fixed components, as in folding in a new document. The observation $r$, equality at the first step and use of the diffused image index as the observation model are the proposed adaptations. [Hofmann, SIGIR 1999](https://doi.org/10.1145/312624.312649)
- The supporting-photo hypothesis derives its sampling interpretation from relevance language models. Its precise candidate-conditioned score and counterfactual controls are the proposal here. The existing repository study uses proxy-image word distributions to construct a relevance model and ranks by cross-entropy. [Lavrenko and Croft, SIGIR 2001](https://www.inf.ed.ac.uk/publications/report/1190.html) · [Existing SBIR proxy experiment](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/7bce042/docs/CROSS_LANGUAGE_RELEVANCE_VI.md)
- Gallery-only reranking for sketch and non-sketch cross-domain retrieval and early work on co-occurrence in query expansion constrain any claim of originality. [Hudson and Smith, ICFRR](https://arxiv.org/abs/2303.17703) · [Xu and Croft, SIGIR 1996](https://my.eng.utah.edu/~cs7961/papers/XuCroft-SIGIR96.pdf)
