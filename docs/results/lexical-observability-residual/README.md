# Does a sketch retain class information outside the part measured by its public words?

All rankings were computed for 768 uniformly sampled unseen sketches per split against the complete original galleries of 17,100 and 12,552 photos. The two splits overlap. Selection on Sketchy is exploratory. Values are percentages in the original KD-SBIR evaluator. Ext-1 reports mAP@all / P@100 and Ext-2 reports mAP@200 / P@200. Encoders are frozen OpenAI CLIP ViT-B/32. The public vocabulary contains the same 20,931 Open Images words in both splits. No other sketch participates in an individual ranking. All inference and evaluation ran on Google Colab Tesla T4. Scoring revision `749520b05c6ddb99380f90a83a3ef541048e10b8`. Equal-angle diagnostic revision `31e08ae78cb4829864d1a4c014e6c0d1fdffae74`. The strong pilot reconstruction repeats all original crops and the frozen regional and hierarchical scores on exactly these query IDs.

A query-specific posterior over public nouns induces a covariance C over their source text vectors. Apply the ridge filter λ(C+λI)⁻¹ to the normalized sketch x, with λ = trace(C)/512. Use its cosine with a gallery donor to test whether the part weakly measured by words adds relevant visual evidence even when that donor already supports the semantic query. Compare against x itself, the complementary component, a rotated dictionary and a uniform-word covariance. The motivating analogy is linear concept removal. The posterior-dependent filter and the same-donor comparison are our own test. [LEACE, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html).

| Readout | Ext-1 | Ext-2 |
|---|---:|---:|
| strong | 85.964 / 85.579 | 75.744 / 74.598 |
| weak/same_donor/0.03125 | 85.918 / 85.500 | 76.067 / 74.630 |
| raw/same_donor/0.03125 | 85.932 / 85.518 | 76.079 / 74.661 |
| rotated/same_donor/0.03125 | 85.932 / 85.496 | 76.139 / 74.710 |

Ext-2 gains are matched or exceeded by the raw sketch and by a rotated vocabulary. The weak component still has a cosine of about 0.965 with the sketch. The original exploratory perturbation normalized x after adding a fixed-length change to it. This gives unequal angles at x. For complementary directions low and x−low, the tangent projections are exactly opposite. At equal angular displacement of 0.05 radians their symmetric KL is 0.35012 for both on Ext-1 and 0.34475 for both on Ext-2. Maximum within-query difference is below 0.000003 on both splits. The earlier KL contrast therefore cannot establish word blindness. Retain the raw visual signal as the appropriate control.

[All ranking outcomes](sketchy-pilot-t4.json) · [Equal-angle test](equal-angle.json).
