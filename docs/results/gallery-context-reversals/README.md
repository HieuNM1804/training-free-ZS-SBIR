# Does changing the gallery reverse the same visual decision?

Run each native Sketchy split independently using its full, original gallery and frozen CLIP ViT-B/32. After scoring, join on identical image paths. Ext-1 has 17,100 gallery photos and Ext-2 has 12,552. They share 2,231 sketches and 2,205 photos in four categories (scissors, tree, wheelchair and windmill). No scores or references from the other split enter a query's native ranking.

Fix a set of difficult decisions without using either graph. For each shared sketch, order the shared photos by direct two-view word score. Retain adjacent pairs among its first 200 in which one photo is relevant and the other is not. This yields 15,149 sketch–pair observations, clustered within sketches and classes. These are conditional diagnostics, with repeated photos and sketches, and cannot be interpreted as percentages of all retrieval errors. Compare the order of those same two photos after each native run.

| Scorer | Fixed pairs reversing | Sketches with changed correctness at rank 1 among shared photos | Correct rank 1 among shared photos, Ext-1 / Ext-2 |
|---|---:|---:|---:|
| Direct public words | 0 / 15,149 | 0 / 2,231 | 1,953 / 1,953 |
| RGB gallery diffusion | 1,489 / 15,149 | 32 / 2,231 | 2,001 / 1,983 |
| Geometric endpoint pooling | 1,284 / 15,149 | 48 / 2,231 | 2,038 / 2,000 |
| Strong reference | 1,464 / 15,149 | 52 / 2,231 | 2,075 / 2,067 |

The strong reference's 1,464 reversals split almost equally by direction. 734 pairs were correct only in Ext-1 and 730 only in Ext-2. Class-level counts are heterogeneous, including 2 of 1,571 for wheelchair and 866 of 7,755 for windmill. Comparing correct first results among the same shared photos shows how gallery context also helps relative to direct word matching. The gallery composition, size and derived graph statistics change together, so this experiment does not identify which of those changes caused an individual reversal. On the direct word readout, the maximum log-score discrepancy across the two independent passes is 0.000022 with zero observed order reversals on the selected pairs.

An example is `scissors/n04148054_17682-4.png`. The same two scored photos are scissors `ext_71.jpg` and wheelchair `n04576002_40991.jpg`. The strong scorer places scissors ahead by 0.5612 in Ext-1 and behind by 0.8212 in Ext-2. The JSON retains median-margin examples in both directions for each class with eligible changes.

This evaluation is motivated by the study of context-dependent choice and by gallery diffusion in image retrieval. It tests a deterministic relative order between two identical candidate items in existing benchmark environments. [Seshadri et al., ICML 2019](https://proceedings.mlr.press/v97/seshadri19a) · [Wang et al., CVPR 2012](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/cvpr12.pdf)

Run `python -m tf_sbir.gallery_context_reversals` on Colab L4, scoring commit `d108a6b1abf4b653c4f327b96ae66ed389155dbc`. Both benchmark galleries serve only as independent scoring environments here. This diagnostic introduces no new split or retrieval metric for official results.
