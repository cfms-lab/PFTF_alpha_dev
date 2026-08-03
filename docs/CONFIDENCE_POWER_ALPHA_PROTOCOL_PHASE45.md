# Phase 45: confidence-aware regular/power alpha protocol

## Scientifically distinct construction

Phase 44 showed that multiplying fixed Euclidean-Delaunay top-cell scores by a
confidence penalty does not transfer. Phase 45 changes the complex itself. It
starts from M1's proper regular triangulation and weighted-power circumradius,
with the previously frozen density scale 0.375, then lowers the vertex power
weight of uncertain observations:

`w_i = spacing_i^2 * (0.375^2 - penalty_scale^2 * (1 - confidence_i))`.

Confidence is the unchanged observed-only Phase-43 combination of nearest-anchor
distance, anchor-plane residual, and unsigned PCA-normal alignment. Anchor
confidence is one. Reference surfaces, topology labels, stress identity, and
known perturbations never enter confidence, weights, connectivity, or threshold
selection.

The candidate penalty grid is `{0.125, 0.25, 0.375, 0.5}`. A penalty that
submerges any input point on any calibration case is invalid. If the selected
penalty submerges a point on a held-out case, that case fails closed to M1
density weights at scale 0.375 and records the fallback.

## New panel and frozen selection

The Phase-44 density-shift, half-view target-occlusion, and local-nonrigid-warp
generators are reused with entirely new seeds: calibration 45001 and held-out
45101, 45102, and 45103. Crossing three analytic families with three stresses
gives nine calibration and 27 held-out cases. No Phase-43 or Phase-44 endpoint
is reused.

Every method uses the unchanged Phase-44 complete critical-gap selector over
the 50%--98% selected-cell interval. Only the confidence power-penalty scale is
calibrated. Binary deletion remains fixed at threshold 0.25 and fixed-cell
continuous weighting at strength 1.0; M1 remains fixed at density scale 0.375.

## Frozen validation gate

The confidence-power candidate must have lower mean geometry and objective than
anchor B4, fused B4, M1, binary deletion, and fixed-cell continuous weighting;
no larger mean Betti error than all five; no worse repeat stability than M1 and
fixed-cell continuous; and lower objective than B5. It must jointly beat M1 and
fixed-cell continuous in at least 18/27 cases, change M1 connectivity in at
least half the cases, and fall back in no more than 10% of cases.

A positive result supports only a floating-Qhull confidence-aware regular alpha
construction. It is not an exact weighted-alpha complex, a PFTF-trained global
alpha, a local-SPD metric complex, real-scan evidence, correct-topology proof,
or deployment evidence.
