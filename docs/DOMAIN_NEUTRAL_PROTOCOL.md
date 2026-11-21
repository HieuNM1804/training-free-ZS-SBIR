# Word-conditioned retrieval without domain names

## Problem and fixed method

Given a single source image and an unlabeled gallery of images, rank gallery items by shared content. The scorer receives image vectors, a matrix of public-word vectors and the frozen CLIP scale. It receives no domain name, collection name, class name, split, query cohort or tuned bandwidth. Every public term is encoded literally with the same frozen text encoder for either role. The external Open Images vocabulary remains explicit.

The primary condition is fixed at this commit. Each gallery image emits a categorical word distribution from its CLIP logits. Let each image draw one other gallery image, over the entire gallery, with probability proportional to $\exp(\tau g_i^\top g_j)$ for $j\ne i$. Here $\tau$ is the frozen checkpoint's logit scale. Average word distributions under this transition to obtain $B_i$. There is no neighbor count, fitted bandwidth, interpolation weight or iteration search. Using an image–text temperature for an image–image transition is a modeling assumption. The study measures its implications, including the effective number of gallery neighbors.

A query induces a vector of likelihoods $l_w$ on the **same** word directions. Let $e=Bl$. To judge candidate $i$, draw $w$ from $B_i$, require that the query supports $w$, then draw another gallery item from those supporting $w$ and measure that item's evidence for the query. The score is
$$
S_i=\sum_w B_{iw}l_w
  \frac{\sum_{j\ne i} B_{jw} e_j}{\sum_{j\ne i} B_{jw}}.
$$
The stationarity that matters is within the shared concept space. A source and target observation referring to the same content must retain some overlapping lexical evidence, and gallery images supporting that content must provide informative witnesses. These assumptions can fail under a new pair of domains. They do not mention strokes, sketches, photos, visual contours or a collection of unseen class names. As an abstract inference rule the same derivation applies to any nonnegative gallery profiles and independent query likelihoods.

Holding the candidate and the distribution over its words fixed gives an exact comparator. Draw *independent* words for the direct query check and for selecting the second image. If $h_{iw}$ denotes the distinct-witness mean, the scores differ by $\operatorname{Cov}_{w\sim B_i}(l_w,h_{iw})$. This is an exact identity with the same leave-one-out convention on both sides. The experiment tests the sign and value of this dependence for ranking. Marginal co-occurrence, softmax diffusion and the covariance identity each have established precedents. The research proposal is the explicit shared latent cause for a candidate's match and for selecting its witnesses, with a deterministic counterfactual audit.

## Calling the scorer

Pass the frozen gallery embeddings, an embedding for each literal public word and the positive checkpoint scale. The scorer accepts one vector or a batch of independent queries. The witness score has one row per query and one column per gallery image.

```python
from tf_sbir.domain_neutral_witness import DomainNeutralIndex

index = DomainNeutralIndex(gallery_vectors, word_vectors, clip_logit_scale)
scores = index.score(query_vector)["witness"]
ranking = scores.argsort(dim=1, descending=True)
first, second = ranking[0, :2].tolist()
explanation = index.explain_pair(query_vector, first, second)
```

The same public interface computes the score margin after every possible removal of one latent query word. It streams over the gallery twice per candidate in FP64. A separate check physically removes all 20,931 terms for two predetermined real queries and 72 gallery images per split and compares every predicted margin. These are implementation fixtures. Benchmark metrics use the full galleries.

## Experimental evidence and interpretation

Run the frozen primary, with independent-word and linear controls, on every permitted benchmark query. Four controls isolate which assumptions matter: literal words and no visual propagation, literal words and the historical gallery graph, historical domain-named words and native-scale propagation, or historical domain-named words and the historical graph. These controls do not select a new primary. The old graph has 16 neighbors, coefficient 0.36 and two updates chosen in earlier Sketchy experiments. The generic study uses none of those numbers to define its primary. Source/target names appear only in the historical ablations; the benchmark adapter alone knows image paths, split names and relevance labels.

Check the matrix implementation against an independent sum over other gallery items, including a one-owner word, a gallery permutation and separate-vs-batch queries. Use the unmodified KD evaluator on each rank list. A mechanism that only works with one prompt bank or the tuned graph has weaker support as a domain-general method. Agreement across these changes would support stability to the tested design choices. Sketchy and TU-Berlin share a domain pair, so their results cannot demonstrate effectiveness on other domain pairs. The user kept the dataset scope fixed. No images from Office-Home or DomainNet are needed or used.

FreeDom provides a vocabulary-mediated baseline for cross-domain retrieval. Relevance models give a probabilistic precedent for estimating word relevance from a collection. The original CLIP provides the public embedding model and its temperature. The new sampling score, component controls and domain-name-free API are derived for this problem. [FreeDom, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Efthymiadis_Composed_Image_Retrieval_for_Training-Free_Domain_Conversion_WACV_2025_paper.html) · [Lavrenko and Croft, SIGIR 2001](https://www.inf.ed.ac.uk/publications/report/1190.html) · [CLIP, ICML 2021](https://proceedings.mlr.press/v139/radford21a.html)
