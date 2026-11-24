# Separating patch eccentricity and relational geometry

On 1,536 independent query evaluations per Sketchy split (with full galleries), the original GCWE reached 84.591 / 83.905 on Ext-1 and 77.579 / 75.279 on Ext-2. Using only the covariance of patch eccentricities gave 84.586 / 83.784 and 77.014 / 74.854. Removing that contribution, keeping the double-centered relational term, gave 83.465 / 82.949 and 77.726 / 75.203. Shuffling the photo coordinates of that term gave 83.900 / 83.192 and 77.578 / 74.826. Normalizing by the residual variances instead did not remove the split discrepancy.

The algebraic identity passed for the real and blank patch couplings. On Ext-1 the original score can nearly be reproduced in aggregate by matching how peripheral patches are, while the strictly relational signal was worse than its coordinate-shuffle control. On Ext-2 the relational signal did outperform both controls in mAP. These results undermine a universal claim that the gain of the original verification comes specifically from structural matching beyond eccentricity. They do not imply that rankings from the eccentricity-only and original systems were identical. The promotion criterion was not met.

[Mechanism and prior art](protocol.md) · [Ext-1](sketchy_1.json) · [Ext-2](sketchy_2.json) · [Per-query observations](paired-L4.zip)
