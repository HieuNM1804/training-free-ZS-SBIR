# Where does the gallery lose all correct photos?

We scored every unseen sketch independently with three frozen CLIP ViT-B/32 channels. They are visual cosine between the two sketch views and the photo, direct agreement of their public-word posteriors, and RGB-gallery diffusion of the direct word score. After ranking we measured which channels contained any relevant photo at equal depth. The full native galleries were used. Category labels were first used to evaluate the resulting lists.

| Selection at depth 200 | Ext-1 queries with a hit / 14,179 | Ext-1 mean correct photos | Ext-2 queries with a hit / 12,694 | Ext-2 mean correct photos |
|---|---:|---:|---:|---:|
| Native public words + RGB diffusion | 12,186 | 164.20 | 10,518 | 147.19 |
| Direct public words | 13,281 | 159.92 | 11,866 | 142.66 |
| Visual cosine | 13,484 | 84.20 | 12,357 | 67.74 |
| Alternate unique native and visual results | 13,322 | 121.71 | 12,326 | 105.33 |
| Alternate unique native and direct-word results | 13,110 | 161.33 | 11,724 | 144.61 |

Among the 1,993 complete native misses on Ext-1, 501 are covered only by visual cosine, 159 only by direct words, 940 by both and 393 by neither. For the 2,176 native misses on Ext-2, these counts are 686, 57, 1,297 and 136. Visual matching can therefore rescue a positive candidate in 72.3% / 91.1% of native misses. A naive interleave pays a large cost in the number of positives retained. These are candidate-coverage diagnostics. They must not be read as improvements in official mAP or precision.

Among native misses recovered by cosine, the modal predicted class in the native top 200 occupies 94.1% / 94.5% of positions on average. One example is `sketch/dolphin/n02068974_212-5.png`. Native word diffusion ranks a seagull first with no dolphin in the first 200. Visual cosine ranks dolphin `photo/dolphin/ext_353.jpg` first. The interleaved list retains it at rank 2. The example was chosen at the median modal-class concentration among such queries. Classes were never used to route a query.

Run `python -m tf_sbir.unseen_candidate_coverage` at `90077abc4fe0cb17d21404481bb66ea5823e1f63` on Colab T4. The JSON gives depths 1, 16, 64 and 200, per-class miss counts and examples. These measurements refer to the two-view lexical diffusion component. They do not establish that every miss survives the separate hierarchical and regional components of the strong reference.
