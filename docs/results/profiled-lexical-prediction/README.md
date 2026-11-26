# Can a worst-case prediction over plausible visual subjects recover the right semantic match?

Cross-fit two families of visual populations on the unlabeled gallery. Each family predicts every photo in the opposite fold. A population supplies a visual likelihood and a probability distribution over the 601 locked public nouns. With the largest visual likelihood rescaled to one, the first estimator takes the largest product of visual likelihood and word probability separately for each word. Its sum over words is the conditional normalizer. Dividing by this sum minimizes the largest log-score deficit to the family for each word. Averaging population predictions gives the matched Bayesian control. This normalized maximum construction has prior art. [Zhou et al., ICML 2021](https://proceedings.mlr.press/v139/zhou21b.html)

The second experiment models uncertainty about population odds for both the sketch and the photo. Within each observation, it computes the largest conditional probability of every noun over an interval of allowed population-odds changes and normalizes this envelope. Fixing all odds multipliers to one gives the matched Bayesian estimator using the same visual populations and noun evidence. A small enumerated fixture checks the accelerated implementation against every extreme prior. [Frozen odds and numerical check](conditional/protocol.json)

| Full unseen mAP / precision (%) | Ext-1 | Ext-2 | TU-Berlin |
|---|---:|---:|---:|
| Direct semantic Dot | 75.35 / 77.10 | 72.66 / 71.07 | 73.46 / 73.98 |
| Normalized profile on target only | 79.96 / 80.40 | 74.86 / 73.25 | 74.73 / 75.07 |
| Bayesian control on the same target populations | 80.27 / 80.52 | 74.79 / 73.18 | 74.84 / 75.17 |
| Bounded conditional odds on both domains | 78.23 / 77.63 | 74.61 / 72.89 | 74.99 / 75.03 |
| Same conditional model with fixed odds | 78.79 / 78.07 | 74.90 / 73.25 | 75.53 / 75.31 |

Two full unseen Sketchy splits selected all configurations by equal-weight mAP and precision as a tie-break. TU used the resulting locks. All runs used the same OpenAI ViT-B/32, seed 42 and KD-SBIR evaluator on Kaggle L4 GPUs. The bounded-odds operation reduced mAP on all three splits compared with its fixed-odds control. The target-only variant reached the four-point improvement on Ext-1 only. [First lock](freeze.json) · [Second lock](conditional/freeze.json) · [First paired archive](paired-L4.zip) · [Second paired archive](conditional/paired-L4.zip)
