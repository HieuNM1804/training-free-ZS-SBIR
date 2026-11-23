# Does standardizing word exposure preserve correct choices?

All variants scored the same 2,231 sketches from four overlapping classes and the same 2,205 candidate photos. Each graph was compiled from its entire original gallery, 17,100 photos for Ext-1 and 12,552 for Ext-2. Candidates retain all native endpoints as possible donors. Only after scoring do we use the class labels to identify adjacent relevant–irrelevant pairs in the direct ranking among the shared candidates.

| Variant | Reversals / 15,129 | Correct pairs Ext-1 / Ext-2 | Correct top-1 Ext-1 / Ext-2, out of 2,231 | Mean absolute change in weight on shared photos |
|---|---:|---:|---:|---:|
| Native RGB diffusion | 1,488 | 11,191 / 10,973 | 2,001 / 1,984 | 7.62 pp |
| Standardize to each candidate's public-word posterior | 976 | 10,902 / 10,554 | 1,942 / 1,949 | 3.02 pp |
| Swap target posterior within top-word group | 2,118 | 10,875 / 10,603 | 1,947 / 1,941 | 5.12 pp |
| Match each row's entropy in native donor order | 806 | 11,047 / 10,767 | 1,964 / 1,951 | 2.98 pp |

Standardizing reduced reversals by 512, but also reduced correct pair choices by 289 and 419. The entropy control attained still fewer reversals and more correct decisions on the same pairs than standardization. Native diffusion achieved the most correct pair and top-1 decisions on both galleries. Direct words gave the same top-1 correctness for all sketches across galleries, but only 1,952 correct first results in each. This argues against minimizing context sensitivity alone as the research objective.

The L4 runtime was lost before its last live result could be recovered. This entire comparison was recomputed on a fresh T4 using frozen OpenAI CLIP ViT-B/32 and the identical selection rule, including the resized two-view sketch and complete 20,931-word vocabulary. Re-encoding and near ties selected 15,129 comparisons here versus 15,149 in the previous L4 audit. No claim is made about exact identity of each cross-device pair. Within the T4 test, benchmark, controls and direct scores use exactly the same saved pair IDs. Direct matching had four tiny cross-gallery reversals and five ties on T4.

Reproduce with `python -m tf_sbir.standardized_context_audit` at `feea7c3c63ae33d06f822f898219143bbbefe3be`. The JSON contains the array signature. The archived T4 fixture can be extracted to `/content/sbir/results/gallery-context-reversals/` to replay the identical choices. The high-precision L4 study of how different donor contributions change a native decision remains [available separately](https://github.com/HieuNM1804/training-free-ZS-SBIR/blob/f5ff47e03d77a898d4d8c79926baa8405f35d0b5/docs/results/gallery-context-reversals/context-attribution.md).
