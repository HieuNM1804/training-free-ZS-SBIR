# Candidate-withheld gallery closure

A high semantic Dot can come from one photo even when no related photos support that match. The testable hypothesis is that support from a small cycle, computed without the candidate's own score, provides relevance information beyond the scores of its neighbors. The study measures whether such support can fix errors that cannot be fixed by choosing a score between the original and flipped-query readings.

## Inference

Let $q$ and $p_i$ be the baseline 601-word posterior vectors for a sketch and gallery photo $i$. Compute $s_i=q^\top p_i$. Build the mutual 16-nearest-neighbor graph $G$ from frozen CLIP photo embeddings. At each score threshold $t$, retain the vertices with $s_j\ge t$ and edges between them to obtain $G_t$.

Let $\mathcal P_i$ contain every simple path of one or two edges in $G\setminus\{i\}$ whose endpoints are distinct neighbors of $i$ in $G$. The certificate is

$$
C_i(q)=\max_{P\in\mathcal P_i}\min_{j\in P}s_j .
$$

It gives the highest threshold at which one such path remains in $G_t\setminus\{i\}$. Adding $i$ back would close a three- or four-vertex cycle. A path through three direct neighbors is already covered by one of its edges at an equal or higher threshold. When the path set is empty, set $C_i=0$. Rank all photos by decreasing $C_i$, using $s_i$ to break ties. Only this tie-break reads the candidate's own semantic score.

Two candidates with the same Dot and neighbor-score multiset can receive different certificates when different pairs of high-scoring neighbors close a path. On a single triangle, $C_i=\min(s_a,s_b)$. Superposition fails because the endpoint signals $(1,0)$ and $(0,1)$ each give zero and their midpoint gives $1/2$. This separates the certificate mathematically from fixed linear diffusion. Gallery-side preprocessing enumerates every triangle and the necessary pairs of edges through a common neighbor. For a gallery with $n$ photos and $M$ indexed paths, applying the certificate to one score vector costs $O(n+M)$ before ranking. This bound excludes construction of the nearest-neighbor graph.

## Prior work

The closest mathematical precedent is cycle convexity. For any positive threshold $t$, set $S_{i,t}=\{j\ne i:s_j\ge t\}$. Araujo et al. characterize admission of $i$ to the one-step cycle interval of $S_{i,t}$ by the existence of two distinct neighbors of $i$ in the same connected component of the induced subgraph. In this implementation $C_i\ge t$ holds exactly when such neighbors can be joined inside that subgraph by a path of at most two edges. Dropping the path-length restriction gives their admission rule at every threshold. Taking the highest qualifying threshold supplies an ordinal retrieval score. The mathematical overlap and the measured failure of the topology controls limit the contribution to an empirical test of an existing graph principle. [Araujo et al., DMTCS 2018](https://dmtcs.episciences.org/4456) · [Full experiment and controls](results/candidate-withheld-gallery-closure/README.md)

The graph uses mutual neighborhoods as in k-reciprocal retrieval. Zhong et al. score neighborhood overlap with a Jaccard distance. Here the RGB graph is fixed and the query selects a prefix of the gallery. The candidate score is held out of the path certificate. [Zhong et al., CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.html)

Iscen et al. propagate evidence on an image manifold by solving a regularized linear system. To compare access to the same gallery structure, the diffusion control here uses the exact same reciprocal edges. The proposed path operator instead asks for the existence of a route between two supporting photos at a given retrieval level. [Iscen et al., CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Iscen_Efficient_Diffusion_on_CVPR_2017_paper.html)

Other connections are level-set persistence, which studies component birth and merge levels, and graph retrieval traversal, which prioritizes graph links to acquire candidates. The proposed contribution is the retrieval criterion with a candidate held out and its associated controls. Cycle closure and max–min paths are standard graph primitives. [Chazal et al., JACM 2013](https://doi.org/10.1145/2535927) · [Chang et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Chang_Explore-Exploit_Graph_Traversal_for_Image_Retrieval_CVPR_2019_paper.html)

### Sketch retrieval and training-free composed retrieval

SpLIP trains shared visual and language prompts on a frozen CLIP backbone, using a sketch–photo triplet objective and a conditional jigsaw task. Its supervision addresses cross-domain alignment during learning. This experiment holds CLIP and the original text prompts fixed and tests whether gallery relationships carry additional evidence at inference. [Singha et al., ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03426.pdf)

FreeDom retrieves proxy images, assigns vocabulary words to them, and combines frequent words with target-domain text. BASIC separately scores image and text agreement with a candidate and fuses the scores. STiTch refines a composed caption with a transition vector and measures transport between distributions representing the caption and candidate image. These methods change the construction or comparison of the composed query. Our fixed 601-word query supplies a score for every photo, and the experiment tests whether connections among other scored photos identify the withheld candidate. Reported performance on composed retrieval answers a different evaluation question from category-level sketch retrieval. [FreeDom, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Efthymiadis_Composed_Image_Retrieval_for_Training-Free_Domain_Conversion_WACV_2025_paper.pdf) · [BASIC, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/6391a02f02dc529cfa3f242fb4343c6a-Abstract-Conference.html) · [STiTch, Li et al.](https://arxiv.org/abs/2605.21261)

## Locked experiment

Use OpenAI CLIP ViT-B/32, the baseline templates `a drawing of a {word}.` and `a photo of the {word}.`, all 601 boxable words, and seed 42. Keep the original and flip aggregation in Dot. Scores for one sketch are computed independently of other sketches. Use full unseen Sketchy Ext-1, Sketchy Ext-2 and TU-Berlin Ext with all 30 classes. Use the original KD-SBIR evaluator. Report mAP@all and P@100 on Ext-1 and TU-Berlin, and mAP@200 and P@200 on Ext-2.

- **Open pair.** Use the second-highest neighbor score with the candidate still held out. This removes the requirement that the two high-scoring neighbors are linked.
- **Permuted endpoints.** Keep every candidate's neighbor score multiset, exact path count by length and interior witness photos. Randomly permute endpoint identities within each half of the visual-neighbor order, once per gallery. This preserves local score supply and the closure operation while removing the identity correspondence between a first-hop position and its alternate path.
- **Same-graph diffusion.** Use reciprocal RGB diffusion with `alpha=0.5` and 16 iterations, alongside the locked Dot baseline.

The primary mechanism test matches the number of photos newly entering the precision cutoff relative to Dot, per query. Compute the class-macro purity difference between closure and each control and bootstrap classes with seed 42. Also count corrected and damaged top-1 results. Use gallery labels after score construction to identify Dot top-1 errors with $\max_{j:y_j=y_q}\max(s^{(1)}_j,s^{(2)}_j)<\max_{j:y_j\ne y_q}\min(s^{(1)}_j,s^{(2)}_j)$. Count how often each arm repairs these errors and how often it breaks a correct Dot top-1 on the full split.

The graph degree, path lengths and all control parameters were fixed before scoring Sketchy. TU-Berlin uses the identical settings. All scoring and checks run in Colab.

```bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m tf_sbir.candidate_withheld_gallery_closure
```
