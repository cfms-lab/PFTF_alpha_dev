# Phase 43: continuous confidence-weighted alpha benchmark

## Frozen sequence

The analytic panel and validation gate were committed first as `4936df2`.
The continuous method and all comparators were then committed as `f1eb525`
before the held-out endpoints were evaluated. The protocol artifact SHA-256 is
`a1e9e37b7b1e6bc203a061eabbce3647699d8104dd26c034270e69486b08853d`.

The observed-only confidence combines nearest-anchor distance, anchor local
plane residual, and unsigned anchor/target PCA-normal alignment. Anchor points
have confidence one. A Delaunay tetrahedron receives the geometric mean of its
four vertex confidences, and its B4 score becomes

`B4_score * (1 + strength * (1 - cell_confidence))`.

Thus low-confidence cells enter the filtration later but are not deleted. The
complete closure of selected tetrahedra is still used, so every output is a
global simplicial subcomplex.

## Calibration

All five methods selected the largest preregistered critical-score quantile,
0.84. The binary comparator selected confidence threshold 0.75, while the
continuous route selected penalty strength 2.0 and score threshold
3.8207679454. Because every method reached the scale-grid boundary, this is a
declared calibration-range warning rather than evidence that 0.84 is a global
optimum.

## Held-out result

The 18 held-out cases contain three untouched seeds for every combination of
sphere, torus, or disconnected spheres with mild or coherent target-view
misregistration.

| Method | Geometry loss | Betti L1 error | Objective | F-score | Repeat stability |
|---|---:|---:|---:|---:|---:|
| anchor B4 | 0.136665 | 1.388889 | 0.206109 | 0.864350 | 0.006327 |
| fused B4 | 0.153141 | 2.611111 | 0.283696 | 0.864250 | 0.034078 |
| fused PCA B5 | 0.186358 | 14.500000 | 0.911358 | 0.758991 | 0.206477 |
| binary confidence deletion | 0.145920 | 2.111111 | 0.251475 | 0.866044 | 0.043830 |
| continuous confidence weighting | **0.134240** | **1.444444** | **0.206463** | **0.866905** | **0.021446** |

The continuous route passes all three preregistered comparisons against fused
B4 and binary deletion: lower geometry loss, no larger mean Betti error, and
lower repeat variability. It also beats B5 decisively on this panel. A second
full execution produced the identical result SHA-256
`fbf7e8f010af65bc9addd7a40129009b4a63cb0a9eecbb587cdc6df34e769f63`.

## Claim boundary and next decision

This is bounded positive evidence for a simulated confidence-weighted adaptive
filtration: `bounded_simulated_confidence_filtration_supported=true`. It is not
evidence for a classical spatially varying alpha complex or a single alpha
predicted by PFTF, so `point_local_alpha_field_supported=false` remains.
Topology is not correct overall (`mean Betti error = 1.444444`), and the
continuous objective is slightly worse than anchor-only B4 (0.206463 versus
0.206109), hence `anchor_objective_dominance_supported=false` and
`topology_correctness_supported=false`.

The next scientifically distinct phase should not tune Phase-43 held-out
cases. It should preregister a new calibration/validation panel that widens or
replaces the boundary-hit quantile grid and tests density, occlusion, and
non-rigid/local misregistration shifts. A true local-alpha claim additionally
requires a construction whose spatial parameter has a precise complex-level
definition, rather than this top-cell score penalty.
