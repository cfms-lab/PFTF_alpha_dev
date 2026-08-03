# Phase 44: confidence-weighted filtration transfer protocol

## Motivation

Phase 43 passed its preregistered comparison against fused B4 and binary
deletion, but every method selected the largest allowed scale quantile, 0.84.
Its continuous objective also failed to beat anchor-only B4. Phase 44 does not
retune those 18 held-out cases. It freezes a new panel and removes the learned
quantile grid from per-case score selection.

## New transfer panel

The analytic sphere, torus, and disconnected-sphere families retain their known
reference surfaces and Betti targets. Each family is crossed with:

| Profile | Anchor / target points | Transfer stress |
|---|---:|---|
| density shift | 48 / 96 | target density is twice anchor density |
| target occlusion | 72 / 72 | target points are selected from the visible half of a random view |
| local non-rigid warp | 72 / 72 | a local Gaussian displacement reaches 0.08 characteristic lengths |

Every case also has 3-degree rotation, 0.035-characteristic-length translation,
and noise sigma 0.005. Calibration uses seed 44001 for nine cases. Held-out
validation uses seeds 44101, 44102, and 44103 for 27 cases. These seeds are
disjoint from each other and from Phase 43.

## Reference-free complete critical-score selection

For each prepared filtration, sort every finite unique top-cell score. Examine
every adjacent pair whose lower endpoint selects between 50% and 98% of cells.
Choose the largest log-score gap and use the geometric midpoint as the
threshold. Ties prefer the lower selected fraction and then the lower
threshold. This consumes no reference geometry, expected topology, family,
profile, or applied perturbation value.

Only continuous penalty strength `{0.5, 1, 2, 4}` and binary confidence
threshold `{0.25, 0.5, 0.75}` are calibrated. Calibration minimizes mean
normalized Chamfer-squared plus normalized Hausdorff plus `0.05 * Betti L1
error`; ties prefer the lower parameter. Anchor B4, fused B4, and B5 have no
calibrated method parameter.

## Frozen transfer gate

Continuous weighting must:

1. have lower mean geometry and objective than anchor B4, fused B4, and binary
   deletion;
2. have no larger mean Betti error than all three;
3. have no larger repeat-stability value than fused B4 and binary deletion;
4. have lower mean objective than PCA B5; and
5. jointly beat anchor B4, fused B4, and binary deletion on per-case objective
   in at least 18 of 27 cases.

Reference points, component labels, profile identities, and perturbation values
are evaluation-only. A positive result supports only transfer of this bounded
confidence-weighted filtration. It cannot establish a classical local-alpha
complex, a PFTF-predicted global alpha, real-scan transfer, correct topology, or
deployment.
