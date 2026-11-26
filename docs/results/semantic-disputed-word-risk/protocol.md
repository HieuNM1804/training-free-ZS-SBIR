# Robust semantic overlap under prompt-dependent lexical ambiguity

## Hypotheses

The original S gives each query and photo a distribution on the same 20,931 words. Its overlap is the expected query support of a word sampled according to the photo. A large vocabulary alone does not establish a large contribution from the tail. Measure top-64 mass and overlap, top-word concentration, and hard baseline errors. Label-based diagnosis is computed after scoring and is never an input to the scorer.

An additional hypothesis is that a photo with high support for a word under its target prompt bank, and lower support for the same word under the source prompt bank, can cause a fragile match if this excess dominates the current query's score. The source-prompt reading of a photo is a controlled change of text description. It is an uncertainty probe whose relation to category relevance must be tested. This is not a label for visual absence.

## Training-free score

Keep the original two-view query posterior (q), native photo posterior (p), and two text banks. On the identical photo, read the source-domain bank to obtain (a). Define (d_w=(p_w-a_w)_+), (D=sum_w d_w) and (b=gamma D). Compute

[
S_{mathrm{robust}}(q,p)=
min_{0le r_wle d_w, sum_w r_w=b}
rac{sum_w q_w(p_w-r_w)}{1-b}.
]

For each photo, the adversary may remove at most the unsupported excess from each coordinate. A fixed budget is allocated in decreasing order of the current query's support (q_w). This is a fractional knapsack argument and reduces to ordinary one-sided trimmed expectation when (d=p) and the budget is specified independently. Zero disputed mass recovers ordinary overlap. The inference score is the lower endpoint from sorting the first 128 query words exactly and bounding the remaining removable contribution by both its remaining dot product and the largest tail word times the remaining budget. This bound is computational and retains evidence from all words in the ordinary overlap. Audit the interval against a full sort on four queries and 64 photos on each split. Measure the interval width relative to the ordinary score for the top 100 baseline photos.

The new component is using the excess mass under a change of domain prompt as a coordinatewise constraint on a query-dependent robust lexical overlap. Simple CVaR and trimming are established. The cap cannot remove a word fully supported under both readers. Estimates and scores use no dataset labels, query pool, RGB diffusion, regions, geometry, gallery neighbors or model updates. The same source and target prompt banks of the original S are used on every split.

## Controls and decision

- Reproduce direct S. Compare ordinary cosine on the softmax vectors, a query-only top-64 mask with the same photo distributions, and the gallery-only MLE prior applied once to photo posteriors. The prior is the existing Bayesian shared-concept baseline applied to exactly the same two-view S.
- Compare averaging the two photo readings with fixed equal weights.
- For each candidate (gamma), compare removing (gamma d) from the photo before reading the query. This removes the same total mass in the same permissible coordinates with fixed proportions. Compare also removing the same total mass adversarially from any words of the photo, including words supported by both readers.
- Evaluate (gammain{1/16,1/4,1/2}) on every query of both unseen Sketchy splits, select by equally weighted mAP then precision, and freeze before TU-Berlin. Sketchy selection is exploratory. Apply the locked fraction to all 30 unseen TU-Berlin classes with fresh unlabelled gallery data. Compare selected row with its matched controls and all simple baselines. A selected robust row must improve over both matched controls to support its specific mechanism.
- The original KD-SBIR function evaluates every ranked query. Record full unseen counts, per-query AP and precision, paired differences and class bootstrap intervals. Use one frozen OpenAI CLIP ViT-B/32 checkpoint on Colab, seed 42. Use no extra datasets. TU verifies a different dataset for the same sketch-to-photo direction.

Direct references are discrete vocabulary transfer in [FreeDom, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Efthymiadis_Composed_Image_Retrieval_for_Training-Free_Domain_Conversion_WACV_2025_paper.html) and distributionally robust trimming for contaminated supervised distributions in [DORO, ICML 2021](https://proceedings.mlr.press/v139/zhai21a.html). Prompt-disagreement constraints and their use at retrieval time are tested here. Gallery prior and cosine are controls.
