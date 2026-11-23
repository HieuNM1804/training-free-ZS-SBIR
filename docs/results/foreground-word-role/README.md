# Does a sketch identify which of two candidate concepts is foreground?

Public-word RGB diffusion returns an incumbent photo A. Nominate the first photo B in its original top 200 whose most likely public noun differs from A's. This nomination sees neither class labels nor other sketches. For words a and b, encode four fixed neutral descriptions. One template puts the first word in the foreground and the second in the background. The other places the first word in the background and the second in the foreground. Each template is evaluated with both noun orders.

Orient both swap differences toward a being the foreground noun. Their average is the foreground-role contrast. Half their difference captures mention position. Isolated single-noun descriptions and a role vector orthogonalized to both isolated identity and mention position act as controls. There is no learned prompt. The descriptions are competing hypothetical readings. They are not captions asserting that both named objects appear in each sketch. Abbasi et al. report first-mention bias in multi-object descriptions. Winoground uses minimal caption pairs to probe compositional understanding. The proposed research test changes mention order independently of semantic role on actual cross-domain retrieval competitors. [Abbasi et al., CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Abbasi_CLIP_Under_the_Microscope_A_Fine-Grained_Analysis_of_Multi-Object_Representation_CVPR_2025_paper.pdf) · [Thrush et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Thrush_Winoground_Probing_Vision_and_Language_Models_for_Visio-Linguistic_Compositionality_CVPR_2022_paper.pdf)

Across 1,914 unique noun pairs from the two full split evaluations, the mean cosine between the two aligned swap differences was −0.493. About 75% of the role vector's norm lay outside the span of isolated identity and mention position, averaged across these pairs. A distinct residual direction can exist even when it fails to improve retrieval.

| Complete native evaluation | Ext-1 | Ext-2 |
|---|---:|---:|
| Queries | 14,179 | 12,694 |
| Queries where exactly one of the two nominees has the right class | 344 | 868 |
| Correct on those queries, original incumbent | 139 | 457 |
| Correct, isolated nouns | 190 | 511 |
| Correct, position alone | 171 | 480 |
| Correct, counterbalanced role | 211 | 531 |
| Fixed / damaged top-1 by counterbalanced role | 131 / 59 | 160 / 86 |
| Fixed / damaged top-1 by isolated nouns | 103 / 52 | 214 / 160 |
| Raw role minus isolated nouns, class-balanced gain in correct first result per query | +0.134 pp | +0.158 pp |
| 95% class-bootstrap interval for that gain | [−0.177, +0.416] | [−1.526, +2.246] |

Both candidates have the same relevance status on most queries. Across all queries, role changes top-1 accuracy by +0.51 and +0.58 points versus native. Requiring both candidate photos to confirm the noun assignment under both word orders, and requiring the sketch decisions to agree, leaves just 67 and 99 informative comparisons. On those restricted opportunities, verified role produces 36 fixes / 8 harms and 40 / 13. Isolated nouns produce 33 / 5 and 39 / 14 on the same opportunities. There is no established class-general advantage over the isolated-noun control.

| Swap only the incumbent with its one named rival | Ext-1 mAP@all | P@100 | Ext-2 mAP@200 | P@200 |
|---|---:|---:|---:|---:|
| Two-view words + RGB gallery diffusion | 82.6970 | 82.2704 | 75.1522 | 73.5939 |
| Isolated noun choice | 82.6970 | 82.2704 | 75.1216 | 73.5939 |
| Counterbalanced role | 82.6972 | 82.2704 | 75.1252 | 73.5939 |
| Counterbalanced role verified by the photos | 82.6971 | 82.2704 | 75.1508 | 73.5939 |

A two-photo swap leaves the membership of the first 200 unchanged and can increase top-1 without increasing full ranking quality. The 768-query pilot also tested adjacent pairs throughout the top 64. Among those mixed pairs the role contrast had balanced accuracies of 0.559 / 0.546 versus 0.697 / 0.654 for isolated nouns. The identity of the pair nomination procedure matters.

Pilot at `9a8a2ee4b1ddf8d088bf58670a4a5e6b6b8e58b1`. Complete Sketchy at `a108921631e0fe23a9387b36a2956f23197d1600`. Run `python -m tf_sbir.foreground_word_role --full` on Colab T4 with frozen CLIP ViT-B/32. Both possible rankings for every query were replayed through original KD-SBIR, then selected according to the query's decision. This is the standalone word-plus-diffusion base. The historical strong full-reference values on Ext-1 and Ext-2 are 85.29 and 77.55 mAP.
