# Risk-localized reacquisition: Phase 0

## Question

Can the existing label-free boundary-risk localizer turn the thin-gap negative
result into an actionable measurement policy?  The first test compares a local
risk-targeted ROI rescan with a uniform rescan under exactly the same added-point
budget.

This is a feasibility experiment, not a next-best-view or sensor-visibility
claim.

## Frozen information boundary

- Reconstruction: B5 with the calibration-frozen multiplier
  `2.80293354289327`.
- Input: the 48-point held-out `opposing_sheets` case.
- Risk policy inputs: observed points, frozen boundary-face risk, and candidate
  return positions only.
- Hidden from selection: component labels and the disjoint evaluation-reference
  subset.
- Candidate returns are split from the dense synthetic reference before
  evaluation.  Both policies use the same pool and exact added-point budget.
- Added returns receive the same observation-noise scale as the base case.

The targeted policy projects flagged face centroids and candidate returns onto
the estimated sheet tangent plane.  It allocates returns round-robin across
flagged anchors.  Ignoring the normal coordinate is intentional: a thin-gap ROI
must acquire both nearby sheets rather than only the closest one.

## Frozen Phase-0 panel

- Added points: `12`, `24`, `36` (1.25x, 1.5x, 1.75x total density).
- Repeats: `8` paired seeds.
- Candidate pool: `4096` points.
- Disjoint evaluation reference: `4096` points.
- Surface samples: `1024`.
- Primary comparison: risk-targeted versus uniform at equal added-point count.

The gate passes at a budget only if risk-targeted reacquisition:

1. reduces component error (the false bridge must actually be resolved more
   often than under uniform reacquisition),
2. reduces labeled bridge edges and faces,
3. does not worsen Betti error, and
4. stays within `0.01` F-score of the uniform control.

Cross-component kNN fraction is recorded as an evaluation-only diagnostic.  It
can show that the added samples supplied missing local information, but it
cannot by itself pass the reconstruction gate.

## Run

```powershell
python -m pftf_alpha.reacquisition `
  --output benchmark-out/risk_targeted_reacquisition_phase0.json
```

The JSON records every selected candidate subset by SHA-256, all paired trial
endpoints, the information boundary, aggregate comparisons, and the final gate.

## Result

### Frozen panel (8 paired repeats)

| Added points | Uniform F | Targeted F | Uniform component error | Targeted component error | Uniform bridge edges | Targeted bridge edges | Uniform cross-kNN | Targeted cross-kNN | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 12 | 0.8045 | 0.7877 | 1.00 | 1.00 | 24.38 | 28.88 | 0.4766 | 0.4830 | fail |
| 24 | 0.8062 | 0.7953 | 1.00 | 1.00 | 27.12 | 33.25 | 0.4729 | 0.4676 | fail |
| 36 | 0.8317 | 0.7975 | 1.00 | 1.00 | 30.25 | 37.25 | 0.4640 | 0.4668 | fail |

Artifact: `benchmark-out/risk_targeted_reacquisition_phase0.json`.

The predeclared 1.25x--1.75x density range does not resolve the false bridge.
The 24-point policy slightly lowers the cross-component kNN fraction, but this
information improvement does not change the connected output and is accompanied
by more labeled bridge edges.  `phase0_supported` is therefore `false`.

### Exploratory density sweep (3 paired repeats)

The initial failure could have been caused by an insufficient budget.  A
separate, explicitly post-hoc sweep tested much larger additions with 256
surface samples:

| Total points | Uniform F | Targeted F | Uniform component error | Targeted component error | Uniform cross-kNN | Targeted cross-kNN | Gate |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 120 | 0.5065 | 0.4720 | 1.00 | 1.00 | 0.4625 | 0.4359 | fail |
| 192 | 0.5089 | 0.4406 | 1.00 | 1.00 | 0.4083 | 0.3409 | fail |
| 336 | 0.5128 | 0.4336 | 1.00 | 0.67 | 0.3526 | 0.1993 | fail |
| 640 | 0.4913 | 0.4062 | 1.00 | 1.00 | 0.2114 | 0.0189 | fail |

Artifact: `benchmark-out/risk_targeted_reacquisition_density_sweep.json`.

At 640 total points, targeted reacquisition reduces cross-component kNN mixing
to 0.0189, so the two sheets are locally distinguishable.  Nevertheless, the
unchanged B5 reconstruction still merges them in all three repeats.  The 336
point row resolves one of three cases, but the result is unstable, loses
F-score, and fails the bridge/topology gate.

## Decision

The broad hypothesis "risk localization + more local samples + unchanged B5"
is falsified on this Phase-0 model.  Missing samples are a real problem, but not
the only problem: once the observations distinguish the sheets, the current
connectivity/selection path can still bridge them.

Do not advance this exact policy to real scans or call it a reconstruction
improvement.  The defensible next experiment is narrower:

1. use reacquisition to drive cross-kNN mixing below a declared sufficiency
   threshold;
2. treat a remaining risky connected output as an algorithmic failure and fail
   closed; and
3. test a connectivity-changing reconstruction only after this gate, rather
   than adding more points indefinitely.
