# Tangential pair-confidence filter: Phase 19

## Frozen question

Phase 18 showed that the Phase-17 matched-displacement guard tolerates the
declared 0.5-degree registration and 10% missing-pair stresses, but treats a 2%
pair-ID mismatch like a physical transient return and rejects nearly every safe
focus case. Can an observed-coordinate pair-confidence filter remove incorrect
pair assignments while retaining normal-direction transient evidence, before
the unchanged Phase-18 safety rectangle is applied?

This phase tests a synthetic calibrated correspondence filter. It does not
claim a real correspondence algorithm or permit threshold retuning of the
matched-displacement guard.

## Frozen information boundary

For routing, the filter receives only the presented primary/repeat coordinate
arrays and their asserted row pairing. It receives no source labels, stress
identity, clean surface, analytic height, endpoint, missing-ID truth, or
mismatch truth.

The synthetic pair-source map created by the Phase-18 perturbation generator is
used only to calibrate and evaluate the pair-confidence cutoff. Geometry and
topology endpoints remain evaluation-only. This supervision makes Phase 19 an
optimistic synthetic upper bound, not a deployable unsupervised matcher.

## Frozen observed-coordinate score

For every presented case:

1. robustly align the repeat array to the primary array by three Kabsch
   iterations, retaining the 80% smallest current pair residuals for the next
   fit;
2. estimate a primary local normal from the smallest-eigenvalue eigenvector of
   the covariance of its 12 nearest primary neighbours;
3. define local tangent spacing as the median tangent-projected distance to
   those neighbours, floored at `0.002 * L_obs`, where `L_obs` is the pooled
   bounding-box diagonal;
4. remove the normal component of the aligned pair residual; and
5. define the pair-confidence mismatch score as tangential residual norm divided
   by local tangent spacing.

No source/stress truth enters alignment, local normal estimation, spacing, or
the score. Ties in alignment trimming use stable index order.

## Frozen dual-cohort cutoff calibration

Use the same five profiles as Phase 18: `exact`, `registration_0p5deg`,
`missing_10pct`, `mismatch_02`, and `combined`.

One scalar pair cutoff is selected jointly from calibration cohorts A and B.
Pairs with `score <= cutoff` are retained. The cutoff is the largest binary64
value strictly below the minimum score of every truth-mismatched presented pair
in the two mismatch-bearing profiles. If either mismatch-bearing
cohort/profile group contains no mismatched pair, calibration fails.

The pair filter must remove every truth-mismatched pair and retain at least 99%
of truth-correct presented pairs in every cohort/profile group. At least eight
pairs must remain in every routed case; otherwise the case fails closed.

## Frozen downstream guard

Compute the unchanged Phase-17 robust matched-displacement peak/support evidence
on only the retained pairs. Do not recalibrate it. Reuse the frozen Phase-18
rectangle exactly:

`peak <= 10.922625244331805 AND support <= infinity`.

The primary reconstruction, unguarded decision, geometry/topology endpoint,
matched-repeat simulator, perturbation severity/order, and seed offsets are
unchanged from Phase 18.

## Frozen three-panel protocol

Each primary panel uses the unchanged nine Phase-8 stresses,
`N in {96,160,256}`, eight repeats, 2048 clean-reference points, and 256 surface
endpoint samples. Five profiles produce 1080 audited rows per panel from 216
primary reconstructions.

- Calibration cohort A seed: `23000804`.
- Calibration cohort B seed: `23100804`.
- Conditional final held-out seed: `23200804`.
- Phase-18's unopened seed `22900804` is not reused.
- Final held-out is not executed unless both calibration panels pass with the
  same pair cutoff and the frozen Phase-18 rectangle.
- No cutoff, score, alignment, neighbourhood, or gate may change after either
  fresh calibration cohort is observed.

## Predeclared success gate

Every profile in every opened panel must satisfy all of the following:

1. at least 99% aggregate retention of truth-correct presented pairs;
2. zero retained truth-mismatched pairs;
3. at least one unguarded harmful-outlier false-safe;
4. zero guarded harmful-outlier false-safes; and
5. at least 90% retention of safe control/local-bump accepts.

The two mismatch-bearing profiles must each contain truth-mismatched pairs. Both
calibration panels must pass before final held-out is opened, and final retuning
is forbidden.

Even a pass supports only this declared synthetic supervision, perturbation
family, and severity. `real_correspondence_supported`,
`real_paired_scan_supported`, `trimmed_reconstruction_supported`, and
`deployment_supported` remain false.

## Planned run

```powershell
python -m pftf_alpha.tangential_pair_confidence `
  --output benchmark-out/tangential_pair_confidence_phase19.json
```

## Result

The implementation tests and full regression suite passed before the fresh
calibration cohorts were executed. Joint truth-supervised calibration selected

`pair score <= 0.05577587737222289`.

This is the immediately preceding binary64 value below the minimum calibration
mismatch score, `0.0557758773722229`. It rejected all 2,738 presented
truth-mismatched pairs across A/B, but retained only 251,749/350,926
truth-correct pairs (`71.74%`), far below the predeclared 99% gate.

| Profile | Calibration A correct-pair retention | Calibration B correct-pair retention | Retained mismatch A/B |
|---|---:|---:|---:|
| `exact` | `70.72%` | `70.70%` | 0 / 0 |
| `registration_0p5deg` | `70.72%` | `70.70%` | 0 / 0 |
| `missing_10pct` | `73.50%` | `73.49%` | 0 / 0 |
| `mismatch_02` | `70.71%` | `70.68%` | 0 / 0 |
| `combined` | `73.49%` | `73.40%` | 0 / 0 |

Case-level safe-focus decisions happened to remain 220/220 (`100%`) in
calibration A and 210/210 (`100%`) in calibration B. Safety did not recover:
harmful false-safes fell only from 275 to 246 in A and from 250 to 225 in B.
Every profile retained harmful false-safes. Both the pair-retention gate and the
zero-harm gate therefore failed, so final held-out seed `23200804` was not
opened and no cutoff was retuned.

The limiting mismatched pair occurred in calibration B, `local_bump`, N=160,
repeat 5, seed `24950866`. Its mismatch score was only
`0.0557758773722229`, while a truth-correct pair in the same case reached
`0.11296911439256035`. Thus incorrect correspondence overlaps correct-pair
noise even after robust rigid alignment and tangent/normal decomposition.
Rejecting every wrong pair necessarily removes many correct pairs and much of
the transient evidence needed by the downstream safety guard.

Therefore `phase19_supported=false` and
`tangential_pair_confidence_synthetic_supported=false`. The scalar local
tangential score family is closed for this protocol: changing its cutoff,
neighbour count, or alignment trim on these cohorts would be post-hoc tuning.
`real_correspondence_supported=false`, `real_paired_scan_supported=false`,
`trimmed_reconstruction_supported=false`, and `deployment_supported=false`.

The reproducible artifact is
`benchmark-out/tangential_pair_confidence_phase19.json` (gitignored,
8,211,482 bytes, SHA-256
`a5464935f294975b7219ee81ade20523b741c5ba80f281070f45aa6222babb24`).
