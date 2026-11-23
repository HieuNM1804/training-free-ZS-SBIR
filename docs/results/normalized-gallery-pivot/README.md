# Can a normalized visual pivot transmit the sketch's content to a different photograph?

All rankings were computed for 768 uniformly sampled unseen sketches per split against the complete original galleries of 17,100 and 12,552 photos. The two splits overlap. Selection on Sketchy is exploratory. Values are percentages in the original KD-SBIR evaluator. Ext-1 reports mAP@all / P@100 and Ext-2 reports mAP@200 / P@200. Encoders are frozen OpenAI CLIP ViT-B/32. The public vocabulary contains the same 20,931 Open Images words in both splits. No other sketch participates in an individual ranking. All inference and evaluation ran on Google Colab Tesla T4. Scoring revision `ce8c0ca66f520d84e13275abc205a5f84c5106d8`. Reproduce with `python -m tf_sbir.normalized_gallery_pivot` after creating the matching two-view strong pilot caches.

## Hypothesis and control

A gallery image can be popular with many sketches regardless of class. Estimate that image's source-domain partition function from responses to the public nouns under source descriptions. Combine it with the prior probability of its depiction under source versus target prompts. Algebraically the source partition cancels, leaving the joint source and target partition in the denominator. Each candidate receives word evidence from donors in the opposite randomly chosen half of its gallery. Thus a donor cannot return itself as an endpoint. Alternative exposures use target prompts alone, visual densities among gallery images, or retain the raw CLIP cosine. Controls shuffle donors' word identities within narrow style bins, use sketch words directly or apply the same adjustment magnitudes in the visual ranking order. All base scores remain paired by query.

| Readout | Ext-1 | Ext-2 |
|---|---:|---:|
| strong | 85.964 / 85.579 | 75.744 / 74.598 |
| only/raw_pivot | 43.822 / 46.250 | 41.792 / 39.590 |
| only/joint_text_partition | 47.207 / 49.721 | 45.279 / 42.898 |
| only/direct_words | 55.087 / 60.362 | 52.729 / 51.124 |
| joint_text_partition/0.03125 | 85.870 / 85.470 | 75.738 / 74.499 |
| raw_pivot/0.03125 | 85.853 / 85.441 | 75.704 / 74.460 |
| shuffled_word_identity/0.03125 | 85.848 / 85.358 | 75.861 / 74.670 |
| matched_visual/0.03125 | 85.898 / 85.594 | 75.975 / 74.746 |
| matched_smoothed_visual/0.03125 | 85.920 / 85.461 | 75.975 / 74.620 |
| image_style_partition/0.03125 | 85.933 / 85.540 | 75.520 / 74.340 |

A better normalized stand-alone pivot still retrieves worse than direct words on both splits. At the smallest nonzero strength, its attachment to the strong base reduces mAP on both splits. The visual-order control surpasses the proposed ordering on both. On Ext-2 shuffling word identities also surpasses it. Results for the prespecified higher strengths and top-200 gains/losses accompany this report.

The independent diagnostic measured each donor's empirical visual popularity over the class-balanced sketch sample after all rankings were fixed. The source prompt log partition correlates with it at Spearman 0.383 / 0.437. The target prompt estimate gives 0.176 / 0.110 and the proposed joint estimate gives 0.211 / 0.171. Prompt-derived exposure contains information about depiction but the joint normalizer discards part of this information. Correcting exposure alone has insufficient support as a retrieval mechanism. These post-ranking diagnostics do not supply test-query statistics to the retriever.

[Full paired score table and correlations](sketchy-pilot-t4.json).
