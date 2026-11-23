# Standardizing the concept exposure of gallery diffusion

The same candidate photo can receive a different fraction of its score from the same group of photo endpoints when the surrounding gallery changes. A measured experiment on the two native Sketchy galleries localizes many pair reversals to this change in total exposure. [Fixed-pair experiment](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/f5ff47e03d77a898d4d8c79926baa8405f35d0b5/docs/results/gallery-context-reversals/context-attribution.md)

We test whether a gallery-independent reading of each candidate's public words provides an appropriate target exposure in cross-domain retrieval. The graph follows the existing two-view RGB diffuser. Each query runs independently and CLIP ViT-B/32 is frozen. No example from another gallery enters the method. The dictionary is the approved set of 20,931 words. KD-SBIR supplies the evaluator.

Let K(i,j) be the row-normalized endpoint distribution of the existing RGB diffusion operator, p(j,c) the fixed public-word posterior of photo j, and L(q,j) its two-view lexical agreement with the sketch. The native expected exposure of candidate i to word c is m(i,c) = sum_j K(i,j) p(j,c). The proposed soft estimator gives (j,c) a joint weight K(i,j) p(j,c) p(i,c)/m(i,c). Thus the latent word marginal is p(i,c), and the conditional distribution of endpoints given (i,c) is preserved. After summing over c the resulting photo weights act on the original evidence L(q,j). Exposure in this latent-variable model is distinct from total weight on a class of real photos. The latter requires an independent check.

The hard version groups donors by their best public word and falls back to the candidate's direct evidence for the remaining target mass. Only 59.47% and 63.11% of the average candidate posterior fell on active donor names. The soft version uses all names. The entropy control changes the exponent of each native row to match the entropy after reweighting while preserving the native donor order. All control rows matched within 0.001 nats in both soft runs. The within-name permutation uses another photo's target posterior in the same top-word group. The no-RGB control replaces the conditional RGB expectation by its uniform-gallery counterpart.

| Method | Ext-1 mAP@all | P@100 | Ext-2 mAP@200 | P@200 |
|---|---:|---:|---:|---:|
| Frozen reference on the pilot | 85.96 | 85.57 | 75.73 | 74.59 |
| Hard public-word groups | 84.44 | 85.19 | 75.48 | 74.40 |
| Soft, all words | 85.41 | 85.27 | 75.60 | 74.44 |
| Soft, swap target within the same top-word group | 85.47 | 85.26 | 75.74 | 74.40 |
| Match row entropy only | 85.46 | 85.25 | 75.68 | 74.29 |
| Soft, uniform donor distribution before standardization | 85.53 | 85.43 | 76.09 | 74.91 |

Each run used 768 uniformly sampled queries per split (seed 87243) and the full native gallery, with source-checked KD-SBIR replay for each sampled query. The soft method recovered 1 and damaged 5 correct first results on Ext-1, and recovered 20 and damaged 15 on Ext-2. Its mAP was lower than the incumbent on both pilots. The target posterior assigned to its own image also failed to outperform the within-top-word permutation. No full-split or TU score is claimed for this method.

The statistical premise is covariate-shift correction with a stable conditional response. Here the proposed target distribution is supplied by a frozen photo's own lexical reading, which is a research assumption. There is no external calibration guaranteeing that public-word exposure equals class-relevant exposure. The existing diffusion operator is a single-view specialization of Wang et al. [Sugiyama et al., NeurIPS 2007](https://papers.nips.cc/paper_files/paper/2007/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract.html) · [Wang et al., CVPR 2012](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/cvpr12.pdf)

Run on Colab L4 with `python -m tf_sbir.public_context_standardization` at `0faa3637fa580fa84b6d6ec903049201c73a77ee` for hard strata, or `python -m tf_sbir.public_context_standardization --soft` at `b2088c495995375dd2461ba60e97a74a71ef2536` for the full posterior. The two result files include all paired controls. The additional `standardized_context_audit` program requires the exact `fixed-boundaries.pt` created by the fixed-pair experiment.
