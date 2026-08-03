# Phase 45: confidence-aware regular/power alpha result

## Frozen construction

Commit `7f4d1d0` froze the new Phase-45 seeds, confidence-to-power formula,
candidate grid, point-submersion policy, comparators, and validation gates.
Commit `a1b71f6` clarified before candidate implementation that zero power-radius
cells are always selected and counted in occupancy while logarithmic gaps are
computed only between adjacent positive critical scores. Protocol SHA-256:
`554b214bbc664041661634e8315c7bd56d87f10fb3825e4a410ad4e765cbe414`.

Commit `c0601a4` then implemented

`w_i = spacing_i^2 * (0.375^2 - penalty^2 * (1 - confidence_i))`,

regular triangulation through the lower 4D convex hull, and proper weighted
power-circumradius scoring. Penalty zero is exactly M1; confidence one removes
the penalty effect. Calibration rejects any penalty that submerges an input
point, while held-out submersion fails closed to M1.

## Calibration

Penalties 0.125, 0.25, and 0.375 retain every point in all nine calibration
cases and change M1 connectivity in all nine. Their calibration objectives are
0.210033, 0.208807, and 0.208555, respectively, so 0.375 is selected. Penalty
0.5 submerges a point in one calibration case and is invalidated.

## Held-out result

| Method | Geometry loss | Betti L1 error | Objective | Repeat stability |
|---|---:|---:|---:|---:|
| anchor B4 | **0.142943** | 1.592593 | 0.222573 | 0.011498 |
| fused B4 | 0.145457 | **1.333333** | 0.212123 | 0.004130 |
| fused PCA B5 | 0.184113 | 3.629630 | 0.365594 | 0.103321 |
| M1 density power alpha | 0.144752 | **1.333333** | 0.211419 | 0.004307 |
| binary confidence deletion | 0.143579 | **1.333333** | **0.210246** | 0.005087 |
| fixed-cell continuous confidence | 0.144438 | **1.333333** | 0.211105 | 0.004568 |
| confidence power alpha | 0.144874 | **1.333333** | 0.211541 | **0.004049** |

The candidate changes M1 connectivity in 27/27 held-out cases with mean
Jaccard distance 0.2525 and never falls back. It passes topology parity,
stability, B5 novelty, connectivity-change, and fallback gates. It fails the
geometry gate because anchor and binary are better, and fails the objective
gate because binary, fixed-cell continuous, and M1 are better. It jointly beats
M1 and fixed-cell continuous in only 6/27 cases versus the required 18/27.

Across stress profiles, candidate objectives are 0.211461 for density shift,
0.210728 for occlusion, and 0.212433 for local warp. Relative to M1 it improves
density and occlusion slightly but regresses local warp; relative to fixed-cell
continuous it improves density but regresses occlusion and local warp.

Two complete executions produced the identical result SHA-256
`d2ef5dc843ac881402c5874db52c45bf334864dcf3f1b3b664eeb2876e1014ce`.

## Interpretation

`confidence_power_alpha_supported=false`. Connectivity change is necessary for
a genuinely different complex but is not sufficient for endpoint improvement.
The tested observed confidence-to-power mapping is not aligned strongly enough
with reconstruction geometry and topology to beat simpler deletion or M1.

Do not tune the Phase-45 held-out seeds, the selected 0.375 penalty, or the
invalidated 0.5 candidate. Another scalar confidence-to-power formula is not a
distinct next step without a new theoretical or learned relation. Keep
`exact_weighted_alpha_supported=false`, `pftf_trained_alpha_supported=false`,
`point_local_alpha_field_supported=false`,
`topology_correctness_supported=false`, `real_scan_transfer_supported=false`,
and `deployment_supported=false`.

A subsequent alpha project should either construct a globally consistent
local-SPD/anisotropic complex with a precise metric compatibility condition or
learn the confidence-to-power relation from separate ground-truth training data
and validate it once on a new frozen panel. The current scalar heuristic family
has reached its evidence boundary.
