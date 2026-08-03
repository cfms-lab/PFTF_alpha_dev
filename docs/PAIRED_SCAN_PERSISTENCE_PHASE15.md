# Paired-scan surface persistence: Phase 15

## Frozen question

Phase 14 found that a broad single-scan observed signature still confused
sparse harmful contamination with safe local geometry. Phase 15 changes the
information set explicitly: can one independent repeat scan provide enough
surface-persistence evidence to reject harmful primary reconstructions while
retaining clean and local-bump cases?

This is a synthetic paired-acquisition audit. It does not establish real-scan
registration, PFTF-SPD novelty, trimming safety, or deployment support.

## Frozen paired observation model

Each primary synthetic case receives one independently sampled replicate with
the same surface/stress family and point count. The replicate seed is

`primary_case_seed + 400000009`.

The primary and replicate are assumed to share a coordinate frame. Sampling,
noise, occlusion choices, and injected-outlier realization are independently
generated from their seeds. Stress identity, injected-source labels, and clean
references are never supplied to the persistence calculation.

This assumption is intentionally stronger than the earlier single-scan model
and weaker than a real acquisition claim: registration error and correlated
sensor artifacts are out of scope.

## Frozen observed-only persistence score

Infer the two shared-trend layers independently in the primary and replicate.
Align replicate layer IDs to primary layer IDs by choosing the identity or swap
assignment with the smaller total centroid distance.

For every primary point and each feasible replicate same-layer neighbourhood
size in `{12, 18, 24}`:

1. find the nearest replicate-layer neighbours;
2. estimate their PCA tangent frame;
3. fit the same six-term local quadratic height model used in Phase 11;
4. predict the primary point's height in that replicate frame; and
5. divide the absolute residual by the larger of `1.4826 * MAD` of replicate
   fit residuals and `0.04 *` the replicate median tangent radius.

The point score is the minimum over neighbourhood sizes. The case retains the
largest `peak` and second-largest `support` scores. A coherent local bump can be
supported by the independent repeat surface; a nonpersistent contamination
point should not be.

## Frozen dual-cohort calibration

Use the same conservative rectangular accept rule as Phase 13:

`peak <= peak_threshold AND support <= support_threshold`.

Candidate thresholds are positive infinity and values immediately below each
harmful calibration coordinate from either cohort. Discard every rectangle
that accepts harmful cases in either cohort. Rank the remainder by:

1. maximum worst-cohort safe control/local-bump retention;
2. maximum total retained safe control/local-bump cases;
3. maximum worst-cohort all-stress safe retention;
4. maximum total retained all-stress safe cases;
5. largest peak threshold; and
6. largest support threshold.

Both calibration cohorts must reproduce harm, remove it completely, and retain
at least 90% of safe control/local-bump accepts before final held-out is opened.

## Frozen three-panel protocol

Every primary panel uses the unchanged nine Phase-8 stresses,
`N in {96,160,256}`, eight repeats, 2048 clean-reference points, and 256 surface
endpoint samples. Each primary case receives the one paired replicate defined
above.

- Calibration cohort A seed: `21800804`.
- Calibration cohort B seed: `21900804`.
- Final held-out seed: `22000804`.
- The final held-out panel is not executed unless both calibration cohorts pass
  with the same frozen rectangle.
- No final-held-out retuning is permitted.

## Predeclared success gate

Phase 15 is supported only if all three full primary panels reproduce at least
one unguarded harmful-outlier false-safe, paired-scan guarded harmful false-safe
count is zero on every panel, and safe control/local-bump retention is at least
90% on every panel.

Even a pass would establish only that independent paired synthetic observations
resolve the declared ambiguity. It would permit a later real paired-scan and
registration study, not trimming or deployment.

## Planned run

```powershell
python -m pftf_alpha.paired_scan_persistence `
  --output benchmark-out/paired_scan_persistence_phase15.json
```

## Result

The implementation tests and full regression suite passed before the frozen
panels were executed. The dual-cohort optimizer selected

`peak <= 3.866421328274521 AND support <= infinity`.

The apparent equality between the serialized peak threshold and the closest
harmful peak is rounding: the candidate threshold is the immediately preceding
binary64 value and therefore rejects that harmful case.

- Calibration A reproduced 56 unguarded harmful-outlier false-safes and
  reduced them to zero, but retained only 36/42 safe control/local-bump accepts
  (`85.71%`).
- Calibration B also reduced 56 harmful false-safes to zero, but retained only
  37/43 safe control/local-bump accepts (`86.05%`).
- Across all safe stresses, retention was 88/118 in A and 97/122 in B.
- Because both calibration panels failed the predeclared 90% focus-retention
  gate, final held-out seed `22000804` was not executed.

The limiting harmful case was a 1% outlier case in calibration A
(`outliers_01`, N=96, repeat 5, seed `22250851`) with peak
`3.8664213282745212` and support `1.7440799339827457`. The conservative
exhaustive rectangle search already maximizes worst-cohort focus retention
subject to zero harmful accepts in both cohorts, so threshold retuning cannot
raise both panels to 90% while preserving zero calibration harm.

Therefore `phase15_supported=false` and `paired_synthetic_supported=false`.
The paired synthetic observation adds useful separation but does not satisfy
the declared safety/retention gate. Real paired-scan registration, correlated
artifacts, trimmed reconstruction, and deployment remain unsupported and were
not started.

The reproducible artifact is
`benchmark-out/paired_scan_persistence_phase15.json` (gitignored, SHA-256
`1a8de74d6d189e2585f70f0e509fb463a654b356c2f1b9c50754eb9bb43dc20c`).
