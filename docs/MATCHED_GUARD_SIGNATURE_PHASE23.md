# Matched-displacement guard signature: Phase 23

## Frozen question

Phase 22 found a harmful missing-pair case under 100% correct, unchanged
pairing whose standardized peak remained below the frozen Phase-18 threshold.
Can a fixed case-level signature of the complete matched-displacement tail
transfer across fresh cohorts while retaining safe control and local-bump
cases?

This phase isolates the downstream guard. It does not develop or evaluate a
new correspondence algorithm.

## Frozen information boundary and profiles

The route receives only ordered primary/repeat coordinates whose presented
pairing is exact by construction, plus the pre-existing upstream candidate
decision. Source labels, simulator stress identity, clean references,
geometry/topology endpoints, and transient-return indices are unavailable to
feature construction and routing.

Only the three Phase-18 profiles that preserve pair identity are used:

1. `exact`;
2. `registration_0p5deg`; and
3. `missing_10pct`.

Mismatch-bearing profiles are excluded because they test correspondence, not
the isolated guard-transfer question. The matched-repeat simulator, 0.5-degree
rotation, 10% pair deletion, primary reconstruction, and endpoint remain
unchanged.

## Frozen displacement preparation

Reuse Phase 17 without modification:

1. median-center primary-minus-repeat displacement per coordinate;
2. scale each axis by `max(1.4826 * MAD, 0.002 * L_obs, eps)`; and
3. compute Euclidean point scores in the standardized displacement space.

The Phase-18 rectangle is not reused or retuned. Phase 23 consumes the same
evidence plus observed pair counts.

## Frozen 12-coordinate case signature

Let `eps` be machine epsilon except for the declared tail ratio floor `1e-6`.
The signature, in fixed order, is:

1. log retained pair count;
2. retained-pair fraction relative to the presented primary count before the
   declared deletion;
3. `log1p` median standardized displacement;
4. `log1p` 95th-percentile standardized displacement;
5. `log1p` peak standardized displacement;
6. `log1p` support standardized displacement;
7. `log1p(peak / (support + 1e-6))`;
8. `log1p(max(peak - support, 0))`;
9. `log1p(maximum_centered_displacement / L_obs)`;
10. `log1p(max(axis_scale) / min(axis_scale))`;
11. `log1p(median(axis_scale) / L_obs)`; and
12. `log1p(norm(displacement_location) / L_obs)`.

Every quantity is observed from the paired coordinate arrays. No feature
selection, interaction, polynomial expansion, per-profile indicator, stress
identifier, or endpoint-derived input is allowed.

## Frozen training rule

Training uses only cases that the upstream candidate would accept. Endpoint
truth labels those training cases as harmful (`1`) or safe (`0`) for the
synthetic supervised audit.

On training A only:

1. standardize each feature by its training mean and population standard
   deviation, replacing a scale below `1e-12` by one;
2. fit ridge linear least squares with penalty `1.0` on feature coefficients
   and no penalty on the intercept; and
3. freeze the rejection cutoff as the immediately smaller binary64 value below
   the minimum harmful training score.

The fit is valid only if training contains at least one harmful and one safe
candidate accept. At routing time an upstream accept is changed to
`unsupported` when its frozen score is greater than or equal to the cutoff.
The model is never refit after training A is opened.

## Frozen sequential protocol

Every panel uses the unchanged nine Phase-8 stresses, `N in {96,160,256}`,
eight repeats, 2048 clean-reference points, and 256 endpoint surface samples.
The three profiles produce 648 audited rows from 216 primary reconstructions.

- Training A: seed `24400804`.
- Development transfer B: seed `24500804`, opened only if training A passes.
- Validation A: seed `24600804`, opened only if development B passes.
- Validation B: seed `24700804`, opened only if validation A passes.
- Final held-out: seed `24800804`, opened only if validation B passes.
- Prior opened or reserved seeds `23300804` through `24300804` are forbidden.

## Frozen gates

Every profile in every opened panel must:

1. reproduce at least one unguarded harmful false-safe;
2. reduce guarded harmful false-safes to zero; and
3. retain at least 90% of upstream-safe control/local-bump accepts.

Training A must pass before B is opened. Each later panel is opened only after
the preceding panel passes. No feature, label, penalty, cutoff, profile, seed,
or gate may change after training A is observed.

Even a complete pass supports only the declared synthetic exact-pair guard
signature. `real_correspondence_supported`, `real_paired_scan_supported`,
`trimmed_reconstruction_supported`, and `deployment_supported` remain false.

## Planned run

```powershell
python -m pftf_alpha.matched_guard_signature `
  --output benchmark-out/matched_guard_signature_phase23.json
```

## Result

The implementation and full regression suite passed before training seed
`24400804` was opened. Among 519 upstream candidate accepts used for training,
168 were harmful and 351 were safe. The fitted rejection cutoff was
`0.49009791476937975`.

Training A passed every frozen gate:

- all 168 harmful false-safes were rejected;
- all 351 upstream-safe accepts were retained;
- all 120 safe control/local-bump accepts were retained; and
- each of `exact`, `registration_0p5deg`, and `missing_10pct` reduced 56
  harmful false-safes to zero with 40/40 focus retention.

Training A therefore opened development B seed `24500804`. The frozen model
transferred to 158/159 harmful false-safes, retained 348/351 all-stress safe
accepts, and retained all 123/123 focus-safe accepts. The exact and registration
profiles passed, but `missing_10pct` retained one harmful false-safe and failed
the zero-harm gate.

The limiting case is development-B `missing_10pct`, `outliers_01`, N=256,
repeat 0, case seed `26900822`. It retains 230 pairs after 26 declared
deletions, contains two harmful vertices and twelve harmful faces, and has:

- model score `0.4460047071388273`, below the frozen rejection cutoff;
- standardized peak `9.625315359824762`;
- standardized support `8.325654660773582`; and
- maximum centered physical displacement `0.12883464039226636` at observed
  characteristic length `3.1263563112538026`.

This is a calibration-margin transfer failure rather than demonstrated
focus-safe representation overlap. Post-hoc diagnosis only, with no retuning,
finds the minimum harmful score across opened A/B is `0.4460047071388273`,
while the maximum focus-safe score is `0.17394405752649739`, leaving a
`0.2720606496123299` score gap. Development B also rejects three harmless
outlier-stress provenance cases, but none is a control/local-bump focus case.

Development B failed, so validation seeds `24600804`/`24700804` and final seed
`24800804` were not opened. The signature, ridge coefficients, and cutoff were
not changed after either opened panel.

Therefore `phase23_supported=false` and
`matched_guard_signature_synthetic_supported=false`. A follow-up must not lower
this cutoff on development B. If pursued, it should preregister separate fresh
cohorts for score fitting and conservative cutoff calibration before any
validation cohort is opened. `real_correspondence_supported=false`,
`real_paired_scan_supported=false`, `trimmed_reconstruction_supported=false`,
and `deployment_supported=false`.

The reproducible artifact is
`benchmark-out/matched_guard_signature_phase23.json` (gitignored, 5,007,199
bytes, SHA-256
`7564ed416cb1f1ce30e8690f6291eb6de5c6ab8797146261353fae343db0d84a`).
