# Phase 43: analytic confidence-to-alpha panel protocol

## Why this panel is new

Phase 42 exhausted the usable Gazebo sources and showed that another binary
target-cell threshold would not answer the alpha-selection question. Phase 43
therefore freezes a new simulated multi-view panel before the proposed method
is implemented. Each case has a dense analytic reference surface and known
surface Betti numbers, so geometry and topology are identifiable.

## Frozen panel

The three analytic families are a unit sphere, a torus with major/minor radii
1.0/0.35, and two radius-0.65 spheres centered at x = -1.2 and x = +1.2. Their
expected surface Betti triples are respectively `(1,0,1)`, `(1,2,1)`, and
`(2,0,2)`.

Each case contains 72 independently sampled anchor points, 72 independently
sampled target-view points, and 1,536 independently sampled reference points.
The target view receives a deterministic rigid perturbation:

| Profile | Rotation | Translation / characteristic length | Noise sigma |
|---|---:|---:|---:|
| mild | 2 degrees | 0.025 | 0.003 |
| coherent | 6 degrees | 0.075 | 0.005 |

Calibration uses seed 43001, yielding six cases. Held-out validation uses
seeds 43101, 43102, and 43103, yielding 18 cases and three repeats for every
family/profile block. The split seeds are disjoint.

## Frozen method comparison

The proposed method must convert an observed-only point confidence continuously
into Delaunay top-cell filtration scores. It is compared with anchor-only B4,
fused-view B4, fused-view PCA anisotropic B5, and binary confidence deletion.
Continuous penalty strengths are `{0.5, 1, 2, 4}` and binary confidence
thresholds are `{0.25, 0.5, 0.75}`. For each configuration, its score threshold
is selected from pooled calibration critical-score quantiles
`{0.08, 0.12, 0.18, 0.26, 0.36, 0.50, 0.68, 0.84}`.

Calibration minimizes mean normalized Chamfer-squared plus normalized
Hausdorff plus `0.05 * Betti L1 error`. Deterministic ties prefer the lower
complexity parameter and then lower score threshold. Surface sampling uses 768
points and the F-score tolerance is 0.05 times characteristic length.

## Frozen validation gate

The continuous route must have lower held-out mean geometry loss than both
fused B4 and binary deletion, no larger mean Betti error than either, and
objective repeat standard deviation no larger than the better of those two.
PCA anisotropic B5 remains a mandatory reported novelty baseline. Stability is
the mean, over six family/profile blocks, of the standard deviation across the
three held-out seeds.

The analytic reference, expected topology, component labels, and applied
perturbation values are evaluation-only. They may not enter point confidence or
filtration scores. A positive result supports only this bounded
confidence-weighted adaptive filtration. It is not a classical spatially
varying alpha complex and does not mean that PFTF predicts one global alpha.
