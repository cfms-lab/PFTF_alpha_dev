# Studentized paired-scan persistence: Phase 16

## Frozen question

Phase 15 showed that one independent repeat can reject every harmful
calibration case, but its fixed residual scale also rejected too many safe
local-bump cases. Reconstruction of those failures located most peak residuals
at the high-curvature bump and one at a weakly supported domain boundary.

Can replicate-only predictive uncertainty distinguish local-model error and
extrapolation from a genuinely nonpersistent primary point without adding
another repeat or retuning the Phase-15 score?

Phase-15 cohorts are development diagnostics only. Phase 16 uses new seeds and
is a separate synthetic paired-acquisition audit. It does not establish
real-scan registration, PFTF-SPD novelty, trimming safety, or deployment
support.

## Frozen observation model

Each primary case receives one independently sampled replicate from the same
surface/stress family and point count. The replicate seed remains

`primary_case_seed + 400000009`.

Primary and replicate share a coordinate frame, but sampling, observation
noise, occlusion realization, and injected outliers are independent. The
studentized score receives only primary/replicate coordinates and layers
inferred independently from those coordinates. Stress identity, injected
source labels, clean references, and geometry/topology endpoints remain
evaluation-only.

## Frozen studentized score

Align inferred replicate layers to primary layers by the same minimum-centroid
identity/swap assignment used in Phase 15. For every primary point and each
feasible same-layer replicate neighbourhood size in `{12, 18, 24}`:

1. estimate the replicate-neighbour PCA tangent frame and median tangent
   radius;
2. fit the six-term local quadratic height model in radius-normalized tangent
   coordinates;
3. compute the fit residuals, hat-matrix diagonal, and absolute leave-one-out
   prediction errors `abs(e_i / (1 - h_i))`, with the denominator clipped at
   machine epsilon;
4. define the replicate predictive base scale as the maximum of
   `1.4826 * MAD`, the 90th percentile absolute LOO error, `0.04 *` tangent
   radius, and machine epsilon;
5. compute query leverage `h_q = x_q^T (X^T X)^+ x_q`; and
6. divide the primary absolute normal residual by
   `base_scale * sqrt(1 + h_q)`.

The LOO term measures local model misspecification such as the nonquadratic
bump, while leverage expands the uncertainty interval for extrapolation. Both
are computed from the observed replicate design and heights without evaluation
labels. The point score is the minimum across neighbourhood sizes. The case
retains the largest `peak` and second-largest `support` point scores.

No Phase-16 scale, quantile, neighbour count, or aggregation rule may be
changed after either new calibration cohort is observed.

## Frozen dual-cohort calibration

Use the conservative rectangular accept rule

`peak <= peak_threshold AND support <= support_threshold`.

Candidate thresholds and the ranking rule are unchanged from Phase 13/15:
discard every rectangle accepting a harmful case in either calibration cohort,
then maximize worst-cohort control/local-bump retention, total focus retention,
worst-cohort all-safe retention, total all-safe retention, peak threshold, and
support threshold in that order.

Both calibration cohorts must reproduce harm, reduce guarded harmful accepts
to zero, and retain at least 90% of safe control/local-bump accepts before the
final held-out panel is opened.

## Frozen three-panel protocol

Every primary panel uses the unchanged nine Phase-8 stresses,
`N in {96,160,256}`, eight repeats, 2048 clean-reference points, and 256 surface
endpoint samples.

- Calibration cohort A seed: `22100804`.
- Calibration cohort B seed: `22200804`.
- Conditional final held-out seed: `22300804`.
- The final panel is not executed unless both calibrations pass with the same
  frozen rectangle.
- Final-held-out retuning is forbidden.

## Predeclared success gate

Phase 16 is supported only if all three full primary panels reproduce at least
one unguarded harmful-outlier false-safe, the guarded harmful false-safe count
is zero on every panel, and safe control/local-bump retention is at least 90%
on every panel.

Even a pass establishes only that one exactly registered synthetic repeat plus
replicate-derived predictive uncertainty resolves the declared ambiguity. It
would permit a later registration-error and real paired-scan study, not
trimming or deployment.

## Planned run

```powershell
python -m pftf_alpha.studentized_paired_scan `
  --output benchmark-out/studentized_paired_scan_phase16.json
```

## Result

The implementation tests and full regression suite passed before the frozen
panels were executed. The dual-cohort optimizer selected

`peak <= 1.2689290645115756 AND support <= 0.9295682875345601`.

- Calibration A reproduced 55 unguarded harmful-outlier false-safes and
  reduced them to zero, but retained only 18/43 safe control/local-bump accepts
  (`41.86%`). Across all safe stresses it retained 41/120.
- Calibration B reproduced 54 harmful false-safes and reduced them to zero,
  but retained only 11/42 safe control/local-bump accepts (`26.19%`). Across
  all safe stresses it retained 32/115.
- Both cohorts failed the predeclared 90% focus-retention gate, so final
  held-out seed `22300804` was not executed.

The limiting calibration-A case was `outliers_03`, N=96, repeat 6, seed
`22660861`. Its case peak `0.9325999204031549` came from a normal surface point,
while an evaluation-only harmful source point supplied support
`0.9295682875345602`. For that harmful point, raw cross-scan residual
`0.09629559382853346` was divided by predictive scale
`0.10359173728261821`, reducing its studentized score below one. The
replicate-derived uncertainty therefore absorbed not only safe curvature/model
error but also the nonpersistent outlier signal that the guard must preserve.

The exhaustive rectangle search already maximizes worst-cohort focus retention
subject to zero harmful accepts in both cohorts. The much lower retention than
Phase 15 is therefore a representation failure rather than a threshold
tie-break. The LOO quantile, leverage factor, and rectangle must not be retuned
on these cohorts.

Thus `phase16_supported=false` and `paired_synthetic_supported=false`. This
studentized one-repeat representation is closed. Real paired-scan registration,
trimmed reconstruction, and deployment remain unsupported and were not
started.

The reproducible artifact is
`benchmark-out/studentized_paired_scan_phase16.json` (gitignored, SHA-256
`f91c9f7c1dc592202ddea9cdcd4e009fb78b2af9b782c159df3e99bf51b7cd08`).
