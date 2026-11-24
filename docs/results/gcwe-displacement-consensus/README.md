# Which displacement statistic helps GCWE?

On 1,536 queries per split in the frozen three-component framework, GCWE's native verifier scored 82.808 / 81.757 on Ext-1 and 77.598 / 75.862 on Ext-2. The primary cross-part residual vote agreement scored 82.672 / 81.926 and 77.920 / 75.953. Scoring the concentration of the two residual vote maps individually instead gave 83.464 / 82.854 and 78.025 / 76.137. Cross-part agreement failed its matched-control test. Requiring the *two* patch halves to predict an identical displacement is unsupported in this pilot.

The higher within-part energy is a secondary result, +0.656 and +0.427 mAP relative to native GCWE. It might reflect peaked CLIP affinities rather than coherent spatial votes. A separate seeded query sample and new null conditions are necessary before giving this an algorithmic interpretation. [Pilot design](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Per-query data](paired-T4.zip)
