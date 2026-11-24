# What can be inferred before scoring the new intervention?

The earlier 1,536-query pilot reported all four factor settings. Taking differences of its published means yields the following descriptive contrasts in mAP percentage points. The "position effect" turns on candidate-specific within-scale positions. Its interaction with scale is the position effect with candidate scale minus the position effect with shared scale.

| Score | Split | Position effect with shared scale | Position effect with candidate scale | Interaction in mAP |
|---|---|---:|---:|---:|
| Regional | Ext-1 | −1.001 | −0.230 | +0.771 |
| Regional | Ext-2 | −0.736 | −0.258 | +0.477 |
| Global plus regional | Ext-1 | +0.214 | +0.186 | −0.028 |
| Global plus regional | Ext-2 | +0.420 | +0.357 | −0.063 |

When global evidence is active, position already helps under shared scale. Thus the archived mAP values do not support the explanation that positive scale–position synergy is required for position to help. In regional-only scoring, scale conditioning partly offsets a loss from using candidate positions. These are descriptive interactions in final metric means. The proposed pilot evaluates the log cross-ratio in individual candidate scores before ranking, a different estimand. Its result cannot be inferred from this table. No confidence interval for this four-way contrast is inferred from marginal summaries.

The separate large confirmation evaluated only original, scale-only and shared gates. It tested the interaction between global scoring and positional specificity, but did not identify the complete scale-by-position interaction. [Original four gates](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/763126d/docs/results/regional-latent-location) · [Independent query confirmation](https://github.com/HieuNM1804/training-free-ZS-SBIR/tree/2ae23a1/docs/results/regional-scale-confirmation)
