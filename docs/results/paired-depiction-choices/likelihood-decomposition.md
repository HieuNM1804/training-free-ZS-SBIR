# What can be learned from knowing which two winners belonged to the same slate?

For an eligible word w and two photos i and j, let u_i and u_j be their positive word responses after the same gallery prior has been added. Let d = log u_i − log u_j. Relative to an unbiased choice, the directional log likelihood of choosing i is

`log[2 u_i/(u_i+u_j)] = d/2 − log cosh(d/2)`.

Expectation over real and blank winners separates two quantities. The signed orientation term is half the difference of the real and blank marginal means of log u. It is algebraically identical for paired and independent draws with the same two winner marginals. Pairing changes only the expectation of the symmetric dispersion penalty log cosh(d/2). Reversing the roles of real and blank exactly flips the first term and preserves the second.

This gives a sharper falsification criterion. If pairing raises accuracy over the independent control, its distinctive contribution comes from the relation between the two winners' **magnitudes** of public-word preference. The orienting information itself is already in the marginals. Compare the exact paired log score, independent log score, orientation only, symmetric penalty only, reversed orientation, and a ratio of arithmetic mean word responses. A simpler marginal explanation outranking pairing limits the scope of any novelty claim.

The first pilot grouped words outside the top 512 into a residual class. Those 512 words retained only 65.9% and 64.8% of the original word probability on the two splits. A second pilot evaluates every one of the 20,931 approved words on every sketch without grouping, pruning, pair sampling, or a new fitted coefficient. A fused GPU reduction calculates both dispersion expectations. Its reference test compares every output with a float64 explicit computation on random tensors including non-multiple tile dimensions. The exact shared-slate law is separately checked by enumerating a discrete example with tied scores and ineligible photos. Keep the full-split previous method as the existing reference. No new configuration is selected on TU.
