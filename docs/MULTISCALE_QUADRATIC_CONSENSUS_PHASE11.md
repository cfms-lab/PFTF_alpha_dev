# Multiscale quadratic surface consensus: Phase 11

## Frozen question

Phase 10 showed that a maximum leave-one-out residual from one local tangent
plane cannot separate harmful outliers from coherent clean and local-bump
surfaces. Phase 11 asks a narrower follow-up question: does a genuinely
different, multiscale local-quadratic representation provide that separation
on a calibration panel and then on an untouched held-out panel?

This remains a synthetic safety audit. It does not establish PFTF-SPD novelty,
trimmed-reconstruction safety, real-scan support, or deployment support.

## Frozen observed-only representation

For each observed point, use only its inferred shared-trend layer and exclude
the point from every fit. At each feasible same-layer neighbourhood size in
`{12, 18, 24}`:

1. estimate a PCA tangent frame from the neighbours;
2. express neighbour positions as tangent coordinates `(u, v)` and normal
   height `h`;
3. fit `h = b0 + b1*u + b2*v + b3*u^2 + b4*u*v + b5*v^2` by least squares;
4. measure the omitted point's absolute prediction residual; and
5. divide by the larger of `1.4826 * MAD` of neighbour fit residuals and
   `0.04 *` the median neighbour tangent radius.

The per-point score is the minimum standardized residual over the feasible
neighbourhood sizes. The case score is the maximum per-point score. This lets
a coherent local feature be supported at one spatial scale while requiring an
isolated off-surface point to be unsupported at every scale.

The route sees only observed coordinates and inferred layer labels. Stress
identity, injected-source labels, and the dense clean reference remain
evaluation-only.

## Frozen calibration rule

The Phase-10 geometry/topology harm endpoint is retained unchanged, including
the `0.025 * characteristic_length` harmful-distance threshold. On the full
calibration panel, consider only cases that the unchanged Phase-7 gate accepts.

- Let `H` be the multiscale case scores of accepted harmful outlier cases.
- If `H` is empty, the calibration is irrelevant and fails.
- Otherwise set the routing threshold to the largest floating-point value
  strictly below `min(H)` (`nextafter(min(H), -infinity)`).

This is the least restrictive scalar threshold that makes calibration harmful
false-safe count exactly zero. It is not adjusted to optimize average accuracy.
Calibration passes only if the resulting threshold also retains at least 90%
of the unchanged gate's safe accepts over the union of control and local-bump
cases. Source-provenance violation accepts remain diagnostic only.

## Frozen two-panel protocol

Both panels use the unchanged nine Phase-8 stresses, `N in {96,160,256}`, eight
repeats, 2048 clean-reference points, and 256 surface endpoint samples.

- Calibration seed: `20900804`.
- Held-out seed: `21000804`.
- The held-out panel is not executed or inspected unless calibration passes.
- If calibration passes, its threshold is frozen and evaluated exactly once on
  the held-out seed. No held-out retuning is permitted.

## Predeclared success gate

Phase 11 is supported only if:

1. calibration reproduces at least one unguarded harmful-outlier false-safe;
2. calibrated harmful-outlier false-safe count is zero;
3. calibration clean/local-bump safe-accept retention is at least 90%;
4. the held-out panel reproduces at least one unguarded harmful-outlier
   false-safe;
5. held-out harmful-outlier false-safe count is zero; and
6. held-out clean/local-bump safe-accept retention is at least 90%.

Even a pass permits only a later, separately predeclared trimmed-reconstruction
study. It does not itself support trimming, real scans, or deployment.

## Planned run

```powershell
python -m pftf_alpha.multiscale_surface_consensus `
  --output benchmark-out/multiscale_surface_consensus_phase11.json
```

## Result

The implementation passed the full 194-test suite before either frozen panel
was opened. Calibration passed at the mechanically selected threshold
`7.203925635649806`, so the held-out panel was executed exactly once.

| Panel | Harmful false-safe, unguarded | Harmful false-safe, guarded | Clean/local-bump safe retention | Gate |
|---|---:|---:|---:|---|
| Calibration, seed `20900804` | 53 | **0** | **36/40 = 90.00%** | pass |
| Held-out, seed `21000804` | 54 | **1** | **41/42 = 97.62%** | fail |

The remaining held-out failure is the 96-point 3% outlier case at seed
`21510826`. Its case score is `7.087467354965811`, below the frozen threshold.
Two harmful source-2 vertices are used by 12 output faces; cross-layer faces,
component error, and Betti error are all zero. The endpoint therefore detects
geometric face harm without requiring a topology change.

The split provenance diagnostic reports accepted source-provenance violations
of 55 to 2 on calibration and 55 to 2 on held-out. These counts are not
substituted for the harm endpoint.

There is no post-hoc scalar-threshold rescue across both panels. A threshold
strictly below the minimum held-out harmful score would retain 40/42 held-out
clean/local-bump safe accepts, but only 35/40 (`87.50%`) on calibration. The
frozen calibration threshold is already the least restrictive threshold that
eliminates all calibration harm and exactly meets its 90% retention gate.

`phase11_supported=false`. The representation improved coherent-shape
retention substantially relative to Phase 10, but the calibration boundary did
not transfer with zero harmful false-safes. Thresholds were not retuned, and
trimmed reconstruction, real-scan validation, and deployment remain
unsupported and were not started.

Artifact: `benchmark-out/multiscale_surface_consensus_phase11.json`.
