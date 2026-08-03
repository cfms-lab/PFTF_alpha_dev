# Split-cohort guard calibration: Phase 24

## Question

Phase 23 showed that one ridge score could separate every opened focus-safe
case from every opened harmful case, but a cutoff placed immediately below the
minimum harmful training score did not transfer to development B. Can the same
observed-only score family pass fresh panels when coefficient fitting and
cutoff calibration use separate cohorts and the cutoff reserves an explicit
harm-side margin?

## Frozen scope and information boundary

This phase changes only the cohort roles and cutoff rule. It retains the Phase
23 twelve-coordinate matched-displacement signature, ridge penalty `1.0`,
three profiles (`exact`, `registration_0p5deg`, `missing_10pct`), upstream
candidate decisions, and endpoint definitions.

The deployed route may use only presented retained pairs, observed coordinates,
the frozen signature transform, fitted coefficients, and frozen cutoff. Clean
references and source labels are allowed only to fit the score and cutoff in
their declared cohorts or to evaluate a panel. Acquisition metadata, true
correspondence beyond the presented pair order, and post-hoc trimming remain
unavailable.

## Frozen score-fit and cutoff rule

1. Fit feature centering, scaling, intercept, and twelve ridge coefficients on
   upstream accepts from score-fit seed `24900804` only.
2. Do not use the score-fit cohort to choose the final cutoff.
3. Apply the frozen score to upstream accepts from cutoff-calibration seed
   `25000804`.
4. Let `s_focus_max` be the maximum score among safe control/local-bump accepts
   and `s_harm_min` the minimum score among harmful accepts in that calibration
   cohort.
5. Calibration is valid only if both groups are nonempty and
   `s_harm_min > s_focus_max`.
6. Freeze

   `cutoff = s_focus_max + 0.25 * (s_harm_min - s_focus_max)`.

The route rejects scores greater than or equal to the cutoff. The quarter-gap
placement reserves 75% of the observed calibration separation on the harmful
side and 25% on the focus-retention side. The fraction, features, labels,
penalty, profiles, gates, and seeds may not change after either training cohort
is observed.

## Frozen sequential protocol

Every panel uses the unchanged nine Phase-8 stresses, `N in {96,160,256}`,
eight repeats, 2048 clean-reference points, and 256 endpoint surface samples.
Each panel contains 648 audited rows from 216 primary reconstructions.

- Score fit: seed `24900804`.
- Cutoff calibration: seed `25000804`.
- Validation A: seed `25100804`, opened only if both training-role panels pass.
- Validation B: seed `25200804`, opened only if validation A passes.
- Final held-out: seed `25300804`, opened only if validation B passes.
- Every opened or reserved Phase 20--23 seed from `23300804` through
  `24800804` is forbidden.

## Frozen gates

After the score and cutoff are frozen, both training-role panels and every
opened evaluation panel must pass the same profile-wise gates:

1. reproduce at least one unguarded harmful false-safe;
2. reduce guarded harmful false-safes to zero; and
3. retain at least 90% of upstream-safe control/local-bump accepts.

Validation A is opened only if score fit, cutoff calibration, and cutoff
separation are valid. Each later panel is opened only after the preceding panel
passes. A failed gate closes every later seed. No failed panel may be used for
retuning.

Even a complete pass supports only the declared synthetic exact-presented-pair
guard calibration. `real_correspondence_supported`,
`real_paired_scan_supported`, `trimmed_reconstruction_supported`, and
`deployment_supported` remain false.

## Planned run

```powershell
python -m pftf_alpha.split_cohort_guard_calibration `
  --output benchmark-out/split_cohort_guard_calibration_phase24.json
```

## Result

The implementation, targeted tests, and full 252-test regression suite passed
before score-fit seed `24900804` was opened. The score-fit cohort contained 528
upstream accepts: 168 harmful and 360 safe. Its fitted scalar score separated
the opened focus cases from harmful cases, with an overall minimum harmful
score `0.5934101233628385` and maximum focus-safe score
`0.11929451628276799`.

Cutoff-calibration seed `25000804` was then opened without changing the twelve
features, ridge penalty, or coefficients. Its 516 upstream accepts contained
165 harmful cases, 351 safe cases, and 126 focus-safe cases. The frozen score
did not preserve the required ordering:

- minimum harmful score: `0.028881929557006913`;
- maximum focus-safe score: `0.13806207726534975`; and
- separation gap: `-0.10918014770834283`.

The calibration rule was therefore invalid before a deployable cutoff existed.
The implementation set the cutoff to the finite fail-closed sentinel
`-1.7976931348623157e+308`, rejected every upstream candidate, and failed both
training-role panel gates through zero focus retention. This is intentional
fail-closed behavior, not a candidate threshold.

The limiting harmful case is calibration `missing_10pct`, `outliers_01`,
N=160, repeat 0, case seed `26400819`, replicate seed `526400828`, and
perturbation seed `928400832`. It retains 144 pairs after 16 declared missing
pairs, has one harmful vertex and five harmful faces, and records:

- model score `0.028881929557006913`;
- standardized peak `4.2892190857714`;
- standardized support `3.2903131252038804`; and
- maximum centered displacement `0.04178755027458678` at observed
  characteristic length `3.123826785172041`.

Post-hoc diagnosis only, with no retuning, places all twelve coordinates of
this harmful case inside the coordinate-wise calibration focus-safe ranges.
Its nearest focus-safe case is only `0.536975522904941` feature-scale-normalized
Euclidean units away; the focus-safe within-class nearest-neighbor interquartile
range is `0.364003918813852` to `0.609378173364983`. Thus the fresh cohort
exposes local overlap in the frozen twelve-coordinate representation, not just
the Phase-23 cutoff-margin problem.

`prevalidation_gate_passed=false`, so validation seeds `25100804`/`25200804`
and final seed `25300804` were not opened. No coefficient, cutoff formula,
feature, or gate was changed after either training-role cohort was observed.

Therefore `phase24_supported=false` and
`split_cohort_guard_calibration_synthetic_supported=false`. Recalibrating the
same global twelve-feature linear score is not supported by this result. A
follow-up must either add a preregistered observed local/spatial harm statistic
on entirely fresh seeds or obtain external acquisition/correspondence evidence.
`real_correspondence_supported=false`, `real_paired_scan_supported=false`,
`trimmed_reconstruction_supported=false`, and `deployment_supported=false`.

The reproducible artifact is
`benchmark-out/split_cohort_guard_calibration_phase24.json` (gitignored,
5,027,119 bytes, SHA-256
`1814fe4a4bc56bd872aa2244565abcdfe7f27ead368aac578d0ee127f522ec30`).
