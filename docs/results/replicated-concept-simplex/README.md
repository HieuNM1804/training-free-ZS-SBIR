# Can a small public vocabulary describe the same visual subjects across domains?

Two complete unseen Sketchy splits selected the configurations. TU-Berlin used each frozen choice and all 30 classes. The Kaggle run used four NVIDIA L4 GPUs, OpenAI CLIP ViT-B/32, the fixed 601 public nouns and seed 42. Ext-1 and TU report mAP@all and P@100. Ext-2 reports mAP@200 and P@200. The official per-query replay used the original KD-SBIR function.

| Method | Sketchy Ext-1 | Sketchy Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Direct word agreement | 75.35 / 77.10 | 72.66 / 71.07 | 73.46 / 73.98 |
| Barycentric coordinates from replicated visual means | 75.88 / 74.80 | 68.98 / 68.00 | 74.08 / 74.66 |
| Estimate anchor reporting rates on independent visual probes | 77.77 / 74.69 | 72.74 / 71.97 | 79.58 / 77.55 |
| Same probes and anchors with raw restricted probabilities | 78.76 / 77.62 | 73.21 / 72.46 | 78.83 / 77.11 |
| Independently tuned RGB neighbor mean | 80.52 / 80.06 | 75.40 / 73.70 | 78.20 / 76.65 |

The first estimator selects an affine basis for the word-conditional means of RGB embeddings under two fixed CLIP text banks. It uses a cross-product between independent gallery halves to choose each pivot. An anchor-word simplex and dissimilarities estimated from independent folds have substantial prior art. Here the question is whether directions that replicate when changing the domain prompt describe subjects reliably enough for retrieval. Both prompt banks observe the same collection of RGB images. [Arora et al., ICML 2013](https://proceedings.mlr.press/v28/arora13.html) · [Walther et al., NeuroImage 2016](https://pubmed.ncbi.nlm.nih.gov/26707889/)

The second estimator represents each noun by its distribution across 1,024 held-out visual probes. It estimates the mixture weights that best reproduce the probe population across the two readers. It divides each anchor's observed name probability by its inferred reporting probability. This assumes that a noun predominantly identifies one subject and that the propensity estimated on photos transfers to drawings. Positive-only learning provides a precedent for the dependence on reporting probability. [Elkan and Noto, KDD 2008](https://cseweb.ucsd.edu/~elkan/posonly.pdf)

Using class labels after ranking, the 32 anchors had mean dominant-class purity of 92.1%, 89.4% and 88.7%. They covered 25 of 25, 21 of 21 and 29 of 30 classes. Correcting the reporting rates fixed and damaged 288/844, 571/681 and 72/62 top-ranked anchor-class decisions. The fitted mechanism improved TU and diminished both Sketchy mAPs relative to the raw matched control. This rules it out as the primary contribution under the agreed selection criterion. [Frozen population configuration](populations/freeze.json) · [Per-query population results](populations/paired-L4.zip) · [Per-query centroid results](paired-L4.zip)
