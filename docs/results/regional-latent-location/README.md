# Candidate conditioning changes which scale is useful

On a fresh fixed set of 1,536 queries per Sketchy split, the original regional reader scored 80.772 / 72.011 mAP. The proposed latent-position readout scored 80.297 / 71.735; displacing the large-scale maps gave 80.281 / 71.711. Changing just the effective window prior gave 80.788 / 72.110. Averaging overlapping windows into patch positions did not improve the regional module.

A separate control isolated scale and position. Let each candidate choose the scale but use the same within-scale positional attention across photos. Standalone regional mAP then rose to 81.002 / 72.270 (+0.231 / +0.258). Once the global semantic term was included, this simplification fell to 83.944 / 77.749 versus 84.130 / 78.106 for fully candidate-specific attention (−0.186 / −0.357). Restricting candidate information to position and sharing scale scored only 79.015 / 70.799 standalone. This is a concrete interaction in rankings: the same change has opposite signs when measured independently and alongside semantic retrieval. It was observed on an exploratory sample; the effect needs a fresh-query confirmation before any method change.

[Pre-specified experiment](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Paired per-query metrics](paired-L4.zip)
