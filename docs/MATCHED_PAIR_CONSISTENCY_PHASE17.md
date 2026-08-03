# Exact-correspondence matched-repeat consistency: Phase 17

## Frozen question

Phases 15 and 16 showed that one independently sampled repeat does not provide
a safe enough local-surface persistence certificate. The remaining ambiguity
mixes geometric change with sampling correspondence, local model bias, and
extrapolation.

As a deliberately stronger synthetic upper bound, does an externally supplied
one-to-one acquisition pair ID make harmful transient returns identifiable
while preserving safe control and local-bump reconstructions?

This differs from Phase 0. Phase 0 selected extra ROI samples and reran an
unchanged B5 reconstruction; it did not compare matched measurements. Phase 17
does not add points or change the primary reconstruction. It audits whether
exact correspondence would supply discriminating evidence.

## Frozen matched-acquisition simulator

The primary panel remains the unchanged Phase-8 sensor-stress generator. For
each primary point index, the simulator supplies one matched repeat return with
the same declared acquisition ID:

1. retain the primary observed `x,y` as the latent beam/sample location;
2. for a primary surface source, evaluate the same declared analytic stress
   surface and layer at that location;
3. for an injected primary source, use the closer of the two analytic layer
   heights as the repeat surface return;
4. add independently seeded observation noise using the same isotropic or
   anisotropic noise family as the primary stress; and
5. for outlier-stress families, independently choose the declared outlier
   fraction of repeat IDs and replace only their depth with a uniform transient
   return.

The repeat seed is

`primary_case_seed + 500000009`.

Primary source labels and stress identity are used only inside this declared
synthetic simulator and for endpoint evaluation. The guard receives only the
ordered primary coordinates, ordered repeat coordinates, and the assertion
that equal indices are externally matched. It does not receive source labels,
stress identity, inferred layers, analytic surface values, or clean references.

This is an optimistic upper bound. Pair identity is assumed correct and there
is no registration error, missing correspondence, object motion, or calibration
drift.

## Frozen robust matched-displacement score

For every pair, compute displacement `d_i = primary_i - repeat_i`. For each
coordinate axis, estimate

- location as the median displacement; and
- scale as the maximum of `1.4826 * MAD`, `0.002 * L_obs`, and machine epsilon,
  where `L_obs` is the diagonal length of the pooled primary/repeat coordinate
  bounding box.

The point score is the Euclidean norm of the three median-centered,
axis-standardized displacement components. The case retains the largest
`peak` and second-largest `support` scores. Axis-wise robust scaling allows the
declared anisotropic-noise stress without using its identity, while fewer than
half transient pairs cannot control the medians or MADs.

No scale floor, aggregation rule, matched-repeat simulator rule, or score may
be changed after either Phase-17 calibration cohort is observed.

## Frozen dual-cohort calibration

Use the same conservative rectangle

`peak <= peak_threshold AND support <= support_threshold`.

Candidate thresholds and ranking are unchanged from Phases 13, 15, and 16.
Every accepted rectangle must reject all harmful calibration cases in both
cohorts. Among those rectangles, maximize worst-cohort control/local-bump
retention, total focus retention, worst-cohort all-safe retention, total
all-safe retention, peak threshold, and support threshold in that order.

Both cohorts must reproduce harm, reduce guarded harmful accepts to zero, and
retain at least 90% of safe control/local-bump accepts before final held-out is
opened.

## Frozen three-panel protocol

Every primary panel uses the unchanged nine Phase-8 stresses,
`N in {96,160,256}`, eight repeats, 2048 clean-reference points, and 256 surface
endpoint samples.

- Calibration cohort A seed: `22400804`.
- Calibration cohort B seed: `22500804`.
- Conditional final held-out seed: `22600804`.
- The final panel is not executed unless both calibrations pass with the same
  frozen rectangle.
- Final-held-out retuning is forbidden.

## Predeclared success gate

Phase 17 is supported only if all three full primary panels reproduce at least
one unguarded harmful-outlier false-safe, matched-displacement guarded harmful
false-safe count is zero on every panel, and safe control/local-bump retention
is at least 90% on every panel.

Even a pass establishes only an exact-correspondence synthetic upper bound. It
would justify a later correspondence-error and missing-pair stress study. It
does not support real paired scans, trimmed reconstruction, or deployment.

## Planned run

```powershell
python -m pftf_alpha.matched_pair_consistency `
  --output benchmark-out/matched_pair_consistency_phase17.json
```

## Result

The implementation tests and full regression suite passed before the frozen
panels were executed. The dual-cohort optimizer selected

`peak <= 5.493421266362053 AND support <= infinity`.

The serialized peak threshold is the immediately preceding binary64 value
below the closest harmful calibration peak.

- Calibration A reproduced 52 unguarded harmful-outlier false-safes and
  reduced them to zero while retaining 43/43 safe control/local-bump accepts
  (`100%`). It retained 118/121 safe accepts across all stresses.
- Calibration B reproduced 58 harmful false-safes and reduced them to zero
  while retaining 43/43 focus accepts (`100%`). It retained 123/124 safe
  accepts across all stresses.
- Both calibrations passed, so final held-out seed `22600804` was opened with
  the same frozen rectangle.
- Final held-out reproduced 55 harmful false-safes, reduced them to zero, and
  retained 43/43 focus accepts (`100%`) plus 124/125 all-stress safe accepts.

The tightest calibration separation occurred in cohort B: the maximum
safe-focus peak was `5.168295051376339`, while the minimum harmful peak was
`5.493421266362054`. Final held-out had a wider margin, with minimum harmful
peak `15.071688292362838` and maximum safe-focus peak
`4.704902804304362`.

The gate is geometry/topology-harm based, not a strict source-provenance gate.
Final held-out provenance-violation accepts fell from 57 to 1 rather than zero.
The remaining case was `outliers_01`, N=96, repeat 6, seed `23060858`, with
peak `4.403917476825436`; it used one source-outlier vertex but had zero harmful
outlier vertices/faces, zero component error, and zero Betti error. Therefore
the declared gate passes, but Phase 17 does not support a claim that every
transient source is removed.

Thus `phase17_supported=true` and
`exact_correspondence_synthetic_supported=true`. This is positive evidence only
for the declared exact-ID simulator. `real_correspondence_supported=false`,
`real_paired_scan_supported=false`, `trimmed_reconstruction_supported=false`,
and `deployment_supported=false`. A correspondence-error, missing-pair, and
registration-perturbation stress audit is required before any real-scan step.

The reproducible artifact is
`benchmark-out/matched_pair_consistency_phase17.json` (gitignored, SHA-256
`d9595604fa48bbb170433f3e82e2b50de1f0eafbd1160161ca692bcce97f5b9f`).
