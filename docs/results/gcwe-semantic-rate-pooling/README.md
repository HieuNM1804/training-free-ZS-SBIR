# Geometry-based enrichment shared by nearby words

The 1,536-query test used a new segment of the seeded query permutation on each split. The original GCWE scored 85.728 and 77.958 mAP. Sharing enrichment among nearby words scored 85.754 and 77.968, changes of +0.026 and +0.011. Grouping the same word IDs by gallery frequency scored 85.722 and 78.128; disabling all feedback for the affected words gave 85.634 and 77.799. Nearby-word pooling did not beat the matched frequency control on Ext-2. The small numeric gains over the baseline cannot be attributed specifically to semantic relatedness.

The transfer criterion was fixed before scoring. No TU validation of this variant was run. [Protocol](protocol.md) · [Both metrics on Ext-1](sketchy_1.json) · [Both metrics on Ext-2](sketchy_2.json) · [Paired ranks' metrics](paired-L4.zip)
