# What the selected photos contribute to GCWE

The diagnostic repeats the frozen feedback computation on the same 1,536 queries per split used for confirmation. Gallery donor labels and target labels enter only in the measurements. The target photos used for the word-overlap margin are in the opposite fold from the donors.

| Donor verifier | Ext-1 donor of correct class (%) | Ext-2 donor of correct class (%) | Ext-1 word-margin gain ×10⁴ | Ext-2 word-margin gain ×10⁴ |
|---|---:|---:|---:|---:|
| Uniform nominee | 43.46 | 34.06 | 0.00 | 0.00 |
| Original GCWE | 58.75 | 48.81 | 23.92 | 16.53 |
| Within-part offset concentration | 58.86 | 48.64 | 21.75 | 17.75 |
| Scrambled coordinates | 57.96 | 47.25 | 19.36 | 16.29 |
| Unbinned residuals | 59.35 | 48.52 | 21.55 | 17.02 |
| Raw offset concentration | 57.62 | 47.50 | 19.13 | 16.07 |

Relative to uniformly selecting one of the same nominees, both the original and proposed verifiers increase same-class selection and improve the mean direct word-overlap margin on held-out photos. Relative to original GCWE, the proposed verifier moves donor purity by +0.102 and −0.174 points and the mean word margin by −2.17 and +1.22 units of 10⁻⁴. Both axes favor real versus permuted coordinates. Neither explains the proposed verifier's paired AP improvement on both splits by itself. The within-class descriptive correlations of the donor-purity change with AP change are 0.149 and 0.154. All four class-bootstrap intervals comparing the primary verifier with original GCWE include zero.

[Measurement definitions](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Per-query records](paired-T4.zip)
