# Finding semantic rivals using unlabeled reference photos

The incumbent is the strongest frozen two-view ranker. We reserve 4,096 reference photos and remove them from the candidate list. The original ranker still uses the full gallery graph. Among the first 200 results from the word score, keep the first eight photos whose eight strongest words are disjoint from the incumbent's. This yields 7,085 of 14,179 queries on Ext-1 and 7,932 of 12,694 on Ext-2. Every rival-selection control sees the same eight candidates.

Two panels of 2,048 photos record when CLIP assigns any of a candidate's eight words to the top 32 for a reference photo. Compare observed joint recognition to the independence expectation conditional on the marginals. Repeat using each reference photo's eight RGB neighbors. The joint selector maximizes the weakest standardized deficit over both tests and both panels. These events describe model predictions. The measured relation does not establish human judgments of semantic opposition.

| Diagnostic | Ext-1 | Ext-2 |
|---|---:|---:|
| Different-class rival, first eligible | 3,741 / 7,085 | 4,555 / 7,932 |
| Different-class rival, joint selection | 4,632 / 7,085 | 5,904 / 7,932 |
| Different-class rival, all four deficits positive | 3,315 / 3,371 | 3,603 / 3,634 |
| Correct top-1 recovered / lost when switching on mention and omission within this subset | 88 / 689 | 103 / 561 |

The class-balanced lift in different-class rival selection over a control matched on candidate position within query class and depth of the eighth eligible rival is 10.05 points on Ext-1 (class bootstrap 95% interval 8.09–12.07) and 12.53 on Ext-2 (9.36–16.18). Labels enter the analysis after the scores are assigned. Many candidate pairs do not meet the absolute deficit criterion. Counting top-1 on this restricted head-to-head task does not give a full-gallery mAP.

Church and Hanks motivate comparing observed word co-occurrence with the chance expectation. This implementation uses signed standardized covariance. [ACL 1989 paper](https://aclanthology.org/P89-1010/)

Code at scoring time was `6b92d9d3282fdb8b7200d94ea9a7e4380a9bdf0b`. Experiments ran on Colab L4 using the approved frozen CLIP ViT-B/32, public vocabulary and independent sketch queries. The command was `python -m tf_sbir.gallery_concept_exclusion`.
