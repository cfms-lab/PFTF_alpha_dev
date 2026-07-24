# PFTF-alpha first benchmark plan

Status: B0-P2 frozen held-out smoke complete; exact CGAL fallback not
implemented.

## Objective

Determine whether a PFTF-derived local SPD metric and confidence-aware fallback
improve alpha-shape reconstruction beyond established density-scaled and
normal-based anisotropic baselines.

## Minimal reproducible panel

| Family | Expected surface Betti | Variation | Required failure signal |
|---|---:|---|---|
| U/C concavity | (1, 1, 0) | opening width, density | convex-hull overfill |
| opposing sheets | (2, 0, 0) | gap / sample spacing | false bridge |
| torus | (1, 2, 1) | hole radius, noise | topology loss |
| disconnected parts | (2, 0, 2) | separation | false merge |
| sharp crease | (1, 0, 0) | angle, normal noise | webbing or hole |
| missing patch | (1, 0, 1) | missing area | unstable closure |

Every family uses fixed train/calibration/held-out seeds. Held-out conditions
include unseen density ratios, gap-to-spacing ratios, and noise levels.

## Methods

| ID | Method | Purpose |
|---|---|---|
| B0 | convex hull | over-smoothing reference |
| B1 | hand-picked global alpha | common manual workflow |
| B2 | exhaustive critical-alpha scan | global selection oracle |
| B3 | persistence + resampling stability | unlabeled automatic baseline |
| B4 | kNN density-scaled alpha | adaptive-scale baseline |
| B5 | normal/PCA anisotropic alpha | anisotropic prior-art baseline |
| P1 | PFTF local SPD metric | field contribution |
| P2 | P1 + confidence/B4 guard prototype | safety-path prototype |

## Current implementation status

The 2026-07-24 smoke runs all six synthetic families through B0-P2. It is a
pipeline and information-boundary check, not a promotion result:

- B2 exhaustively evaluates every top-simplex critical value and may use the
  dense synthetic reference as the declared global-selection oracle.
- B3 scans the full component/Euler topology sequence, then evaluates a fixed
  budget by input-point fit and 90% resampling stability. It does not use the
  dense reference or expected component count during selection. The terminal
  convex-hull plateau is not treated as a finite persistence interval,
  preventing the largest sliver alpha from winning automatically.
- B4 divides each Euclidean tetrahedral circumradius by the geometric mean kNN
  spacing of its vertices. Thresholded top cells receive complete downward
  closure before boundary extraction.
- B5 estimates a regularized local-PCA SPD metric, combines it with the B4
  density normalization, and applies a planarity-weighted normal penalty. It
  uses the same complete-closure extraction.
- B4/B5 currently score cells of one fixed Euclidean Delaunay triangulation.
  They are scale/rotation-invariant research surrogates, not exact anisotropic
  Delaunay or CGAL alpha-shape constructions.
- P1 forms a signed trace-free relation tensor from directed receiver/source
  kNN scale contrast and receiver-direction imbalance. It maps that relation
  through bounded log-eigenvalues to an SPD metric, then blends uncertain point
  metrics toward the density-scaled identity.
- P1 confidence combines relation strength, neighbor-distance regularity, and
  reciprocal-kNN support.
- P2 accepts an explicit threshold or freezes one without reference geometry.
  The automatic path pools P1 simplex confidence on the calibration panel and
  places the threshold between order statistics for a predeclared target
  fallback fraction, before P2 multiplier selection.
- For each low-confidence top cell P2 uses `max(P1 score, B4 score)`, so the cell
  must pass both tests under one frozen multiplier. Confident cells retain the
  P1 score. This is a conservative fixed-complex guard, not exact CGAL fallback.
- P1 uses the same fixed Euclidean Delaunay top-cell closure as B4/B5, so it is
  not an exact spatially varying anisotropic triangulation.
- `--calibrate-adaptive` first freezes P2 confidence without dense references,
  then pools the six calibration cases to minimize one mean geometry/topology/
  complexity objective per method and freezes one multiplier before evaluating
  the requested split. Its current topology term is component error only. The
  declared surface Betti targets and their errors are evaluation-only and do
  not enter adaptive multiplier or reference-free P2 confidence selection.
- The frozen held-out smoke used 48 input points, 512 reference points, 96
  surface samples, 12 requested calibration candidates, and seed 20260724. It
  selected B4 multiplier 1.1977526569935681, B5 multiplier 2.80293354289327,
  P1 multiplier 1.2076985596095746, and P2 multiplier
  1.2217738660639386.
- All 24 B4/B5/P1/P2 held-out records used the same corresponding frozen
  multiplier, declared `uses_reference_for_selection=false`, and evaluated one
  candidate.
- Across the six smoke cases, mean F-score was 0.15198 for B4, 0.15616 for B5,
  0.17634 for P1, and 0.16777 for P2. Mean normalized squared Chamfer was
  0.00919, 0.00957, 0.00966, and 0.01037 respectively; component-error sums
  were tied at 2. Thus this is not evidence of an overall P1 or P2 win.
- GF(2) surface Betti-error sums were B4 20, B5 27, P1 25, and P2 25. P1/P2
  reconstructed the torus as `(1,2,1)` exactly, but missed the other five
  declared targets; the missing-patch full-sphere target alone had error 14.
  The topology endpoint therefore exposes failures hidden by component count.
- P1's maximum observed metric condition number was 1.64114 under the bound of
  9.0. Median point confidence ranged from 0.34275 to 0.43049 across cases, and
  numeric fallback fraction was zero.
- The reference-free target fallback fraction 0.25 selected threshold
  0.26868716232131434 from 1,099 calibration cells and achieved 0.25842.
  Held-out fallback ranged from 0.07487 to 0.48990 of all cells and 0.11009 to
  0.42308 of selected cells, so the nearly-all-fallback failure was removed.
- Score-level and selected-set B4 guard violations were zero. Every selected
  top-cell set reported complete downward closure and no face incidence above
  two. These invariants do not establish zero false-safe cases because that
  endpoint and exact fallback are still pending.
- Every result records the selected generic parameter, complete candidate
  range, selection objective, reference-use flag, runtime, surface endpoints,
  and method diagnostics. Alpha-specific fields remain populated for B1-B3.
- GF(2) surface Betti numbers and L1 target error are implemented for the
  triangular output complex, with Euler-Poincare consistency checked in tests.
  Exact semantic hole/cavity localization, exact false-bridge/false-safe,
  normal-consistency, volume, and memory endpoints are not implemented yet.
  The existing `false_bridges`/`false_splits` fields are component-count proxies.
- Surface-sampling sliver tetrahedra can produce very large critical values;
  these remain visible in the candidate range and are not silently removed.

The P2 conservative fallback path has run, but the project remains **ToDo**
because the selective fallback smoke does not improve P1 geometry or topology,
and exact/false-safe evaluation is still pending.

## Primary endpoints

1. Exact false bridges per shape (the current component proxy is provisional).
2. Betti/component error.
3. Surface F-score and Hausdorff distance.
4. Stability under 10% subsampling and calibrated noise.
5. Non-manifold and singular-face counts.
6. Runtime, memory, and fallback rate.

## Required ablations

- PFTF relation field versus local PCA covariance.
- Isotropic versus anisotropic metric.
- Density only, normal only, relation only, combined.
- Hard gate versus soft calibration followed by hard exact evaluation.
- Confidence fallback on/off.

## Promotion rule

The initial B0-P2 end-to-end run gate is satisfied. The project remains
**ToDo** until a higher-fidelity frozen held-out evaluation shows value beyond
both B4 and B5 on declared geometry and topology endpoints, and an exact or
validated fallback demonstrates no unreported
false-safe cases. A mean F-score gain in this small smoke alone is insufficient
for promotion.
