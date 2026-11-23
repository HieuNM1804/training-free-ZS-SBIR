# Does the noun's local geometric credit improve the retrieval score?

The patchwise decomposition passed its reconstruction identity on every scored real and blank pair. On 1,536 new queries per split the unchanged GCWE reached 86.055 and 78.206 mAP. Reading the local geometric evidence of the noun receiving credit gave 85.945 and 77.925. Permuting photo patch coordinates before reading the noun gave 85.999 and 78.226. Exchanging noun identities gave 85.972 and 77.839. The word-specific intervention failed the prespecified test. These numbers do not show that CLIP's local noun grounding selects the geometrically relevant patches for this retrieval task. The mathematical attribution remains available independently of this negative ranking result.

[Derivation and controls](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Paired per-query metrics](paired-L4.zip)
