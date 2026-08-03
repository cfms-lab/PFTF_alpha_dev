# Observed-data identifiability audit: Phase 14

## Frozen question

Phase 13 showed sample overlap for the Phase-12 peak/support influence family:
one harmful 1% outlier case had lower observed influence than many safe cases.
Phase 14 does not propose another guard or threshold. It asks a diagnostic
question: within a broader, predeclared observed-only signature family, are
accepted harmful outlier cases distinguishable from accepted safe control and
local-bump cases on a new held-out panel?

This audit cannot prove absolute non-identifiability. A negative result is
limited to the declared feature family, distance rule, and synthetic panels. A
positive result would motivate a separately frozen guard phase; it would not
itself support trimming, real scans, or deployment.

## Frozen observed-only signature

For every audited case, compute these 14 values without stress identity,
injected-source labels, or clean references:

1. log observed point count;
2. inferred layer-count imbalance;
3. median within-layer nearest-neighbour distance / observed bounding-box
   diagonal;
4. 95th-percentile within-layer nearest-neighbour distance / diagonal;
5. median cross-layer nearest-neighbour distance / diagonal;
6. inferred layer-centroid gap / diagonal;
7. pooled median thickness along the centroid-gap direction / diagonal;
8. median local insertion influence;
9. 95th-percentile local insertion influence;
10. second-largest local insertion influence;
11. largest local insertion influence;
12. median multiscale quadratic leave-one-out residual score;
13. 95th-percentile multiscale quadratic residual score; and
14. largest multiscale quadratic residual score.

The Phase-12 influence and Phase-11 multiscale residual configurations remain
unchanged. The signature is computed before any evaluation label is consulted.

## Frozen diagnostic classes and distance rule

Only unchanged Phase-7 gate accepts enter the two-class audit:

- **harmful:** an outlier-stress accept with Phase-10 geometry/topology harm;
- **safe-focus:** a control or local-bump accept without geometry/topology harm.

Other stress families remain in the source panel but are excluded from this
narrow identifiability question.

Fit feature-wise scaling on the pooled calibration audit cases. For each
feature, use `1.4826 * MAD`; if zero, use `IQR / 1.349`; if still zero, use
`max(0.05 * abs(median), 1e-9)`. Standardize all signatures with the calibration
median and scale.

For each query, compute Euclidean distance in standardized 14-dimensional
space to the nearest harmful calibration case and nearest safe-focus
calibration case. Predict harmful when the harmful distance is less than or
equal to the safe distance; ties therefore fail closed. Calibration diagnostics
use leave-one-out neighbours. Held-out diagnostics use the full calibration
reference set. No feature weights, neighbour count, or threshold are tuned.

## Frozen two-panel protocol

Both panels use the unchanged nine Phase-8 stresses, `N in {96,160,256}`, eight
repeats, 2048 clean-reference points, and 256 surface endpoint samples.

- Calibration seed: `21600804`.
- Held-out seed: `21700804`.
- Phase-10 geometry/topology harm labels remain evaluation-only.
- The held-out panel is executed once after implementation tests pass. No
  result-dependent feature or distance changes are permitted.

## Predeclared identifiability criterion

The declared signature family is supported only if both calibration
leave-one-out and held-out diagnostics:

1. contain at least one harmful and one safe-focus case;
2. identify every harmful case (`harmful recall = 100%`); and
3. identify at least 90% of safe-focus cases (`safe specificity >= 90%`).

Nearest opposite-class cases and feature differences are reported for every
missed harmful case. Regardless of outcome,
`trimmed_reconstruction_supported=false`, `real_scan_supported=false`, and
`deployment_supported=false` remain fixed.

## Planned run

```powershell
python -m pftf_alpha.observed_identifiability `
  --output benchmark-out/observed_identifiability_phase14.json
```

## Result

The implementation passed the full 207-test suite before the two frozen panels
were opened. The declared signature family **did not pass** the identifiability
criterion.

| Panel | Harmful correct | Harmful recall | Safe-focus correct | Safe specificity | Gate |
|---|---:|---:|---:|---:|---|
| Calibration LOO, seed `21600804` | 51/54 | **94.44%** | 42/42 | **100%** | fail |
| Held-out, seed `21700804` | 47/51 | **92.16%** | 39/40 | **97.50%** | fail |

All seven missed harmful cases are listed below. `d_safe < d_harm` means the
case is closer to a calibration safe-focus case than to any calibration harmful
case. The final column reports the three largest absolute standardized feature
differences from that nearest safe case.

| Panel | Harmful case | Nearest safe seed | `d_safe` | `d_harm` | Largest standardized differences |
|---|---|---:|---:|---:|---|
| calibration | `outliers_01`, N=96, seed `22070865` | `22440856` | 2.426 | 3.887 | median influence 1.22; cross-layer distance 1.22; peak residual 0.95 |
| calibration | `outliers_01`, N=160, seed `23000819` | `22630828` | 2.751 | 3.007 | layer imbalance 2.16; gap thickness 1.47; influence p95 0.45 |
| calibration | `outliers_01`, N=256, seed `24070871` | `23620824` | 1.564 | 1.955 | peak influence 0.79; peak residual 0.74; support influence 0.70 |
| held-out | `outliers_01`, N=160, seed `23100819` | `22610814` | 1.176 | 2.729 | centroid gap 0.78; influence p95 0.43; cross-layer distance 0.41 |
| held-out | `outliers_01`, N=160, seed `23110826` | `22650842` | 2.167 | 3.182 | gap thickness 1.64; centroid gap 0.97; peak influence 0.72 |
| held-out | `outliers_01`, N=160, seed `23170868` | `22660849` | 1.738 | 2.423 | gap thickness 1.23; peak influence 0.69; peak residual 0.64 |
| held-out | `outliers_03`, N=256, seed `24230846` | `23600810` | 1.997 | 2.004 | support influence 1.17; influence p95 0.83; residual p95 0.74 |

Held-out also has one safe false alarm: the 96-point local-bump case at seed
`22520842` is closer to harmful seed `22250857` than to its nearest safe
reference. This is consistent with the earlier observation that coherent local
bumps and sparse contamination can occupy overlapping observed feature regions.

The strongest failure mode is 1% contamination: six of the seven missed
harmful cases are `outliers_01`. Adding global spacing, layer balance, gap, and
thickness features to both local residual and insertion-influence summaries did
not restore zero-harm separability.

`feature_identifiable=false` and `guard_supported=false`. This does not prove
absolute non-identifiability, but it closes the declared 14-feature local/global
signature route. Another guard using the same single-scan coordinates should
not be proposed from these panels. A defensible next study would change the
information set explicitly, for example repeated-scan persistence or sensor
confidence, and test whether that added observation resolves the safe/harmful
twins.

Trimmed reconstruction, real-scan validation, and deployment remain
unsupported and were not started.

Artifact: `benchmark-out/observed_identifiability_phase14.json`.
