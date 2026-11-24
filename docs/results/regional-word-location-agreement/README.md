# Does agreement of word location profiles identify a better gallery candidate?

On 3,072 sketches per split with complete galleries and the T4-matched seed-42 reference, the original global-plus-regional score reached 85.405 / 84.775 on Ext-1 and 76.931 / 74.748 on Ext-2 (mAP / precision). Applying the word-location agreement correction gave 85.342 / 84.762 and 76.995 / 74.782. Reassigning the same correction between adjacent-score candidates gave 85.356 / 84.769 and 76.943 / 74.784. Restricting the correction to scale gave 85.396 / 84.782 and 76.946 / 74.773.

The bounded-energy identity passed on every batch. The candidate-specific correction reduced Ext-1 mAP and did not beat its matched rank sham there; it failed the transfer criterion. The opposite scoring direction, allowing each word to use its own position, showed the reverse pattern (85.457 / 76.799 mAP). There is no consistent evidence from these splits for increasing or decreasing this agreement term.

[Derivation and frozen protocol](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Paired query records](paired-T4.zip)
