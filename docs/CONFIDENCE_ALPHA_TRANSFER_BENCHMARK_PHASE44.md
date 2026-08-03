# Phase 44: confidence-weighted filtration transfer result

## Frozen sequence

Commit `8574681` froze the density-shift, target-occlusion, and local-nonrigid
transfer panel, disjoint seeds, complete critical-score selector, method grids,
comparators, and validation gates. Protocol SHA-256:
`2bdd309855500e2e0bced3701ae4fffa3741358513d6e0d4a5310ed269a5f5a3`.
Commit `e504afc` then implemented the reference-free selector and benchmark
before any held-out endpoint was evaluated.

For every filtration, the selector scans every unique critical score and picks
the largest adjacent log-score gap whose lower endpoint selects 50%--98% of
top cells. It therefore removes the Phase-43 learned quantile grid. Reference
geometry, expected topology, stress identity, and perturbation values do not
enter threshold selection.

## Calibration and held-out result

The nine calibration cases select continuous penalty strength 1.0 and binary
confidence threshold 0.25. The 27 held-out cases contain three new seeds for
each analytic-family/stress combination.

| Method | Geometry loss | Betti L1 error | Objective | Repeat stability | Selected-cell fraction |
|---|---:|---:|---:|---:|---:|
| anchor B4 | 0.157169 | 1.740741 | 0.244206 | 0.048093 | 0.970242 |
| fused B4 | **0.140571** | **1.333333** | **0.207237** | 0.005539 | 0.957254 |
| fused PCA B5 | 0.170583 | 2.777778 | 0.309472 | 0.071817 | 0.966271 |
| binary confidence deletion | 0.141527 | **1.333333** | 0.208194 | 0.004671 | 0.964757 |
| continuous confidence weighting | 0.141652 | **1.333333** | 0.208318 | **0.004282** | 0.963918 |

Continuous weighting passes topology parity, repeat stability, and the B5
novelty comparison. It fails the primary geometry and objective gates because
fused B4 is better by 0.001081 geometry and 0.001081 objective, while binary
deletion is better by 0.000125 on both. It jointly beats anchor, fused, and
binary objective in only 4/27 cases versus the frozen requirement of 18/27.

By stress profile, continuous objective is 0.209558 for density shift,
0.206791 for occlusion, and 0.208605 for local warp. Corresponding fused-B4
values are 0.208910, 0.203992, and 0.208811. The local-warp advantage is too
small and too isolated to offset the density and occlusion regressions.

The result is deterministic: two complete executions produced SHA-256
`cdf19791161c4c5fd762395cb78fce2293d211b8ed7b0ac855d26b5995cff5ca`.

## Interpretation and next boundary

`bounded_confidence_filtration_transfer_supported=false`. Phase 43 remains a
bounded in-panel result, but its advantage does not transfer to the new stress
panel under reference-free complete critical-gap selection. The selector also
chooses late filtration states (mean selected fractions 0.957--0.970), making
the three density-based routes nearly indistinguishable.

Do not tune Phase-44 held-out cases, strength, binary threshold, or occupancy
bounds. Another confidence penalty on the same Euclidean Delaunay cells is not
a scientifically distinct next step. A future phase should instead define a
complex-level spatial construction, such as a rigorously specified weighted or
local-metric alpha construction, and preregister a new panel. Keep
`point_local_alpha_field_supported=false`,
`topology_correctness_supported=false`, `real_scan_transfer_supported=false`,
and `deployment_supported=false`.
