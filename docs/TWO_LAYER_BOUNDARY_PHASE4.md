# Two-layer operating-boundary sweep: Phase 4

## Frozen question

Phase 4 estimates where the Phase-3 globally separable two-layer claim stops
working. It is a boundary-characterization experiment, not threshold tuning.
The Phase-3 sampling gate and reconstruction settings remain unchanged.

Construction and routing receive observed coordinates and inferred layer IDs
only. Sweep-axis names, severity values, true labels, and dense references are
evaluation-only.

## Frozen axes and levels

Seed `20280804`; eight unseen repeats per level; 160 observed points, 2048
reference points, and 256 evaluation surface samples. Observation noise is
0.01. The panel contains 4 axes x 6 levels x 8 repeats = 192 cases.

1. `curvature`: paired paraboloid coefficient
   `{0.00, 0.12, 0.24, 0.36, 0.48, 0.60}` at gap 0.80.
2. `tilt_span`: separation `0.80 + span*x`, with span
   `{0.00, 0.25, 0.40, 0.55, 0.70, 0.76}`.
3. `overlap_offset`: x-offset of the upper sheet
   `{0.00, 0.50, 1.00, 1.50, 2.00, 2.50}`. Sheet width is 2, so offset 1 is
   50% overlap and offsets >=2 have no areal overlap.
4. `contact_severity`: minimum local gap decreases from 0.80 according to
   severity `{0.00, 0.20, 0.40, 0.60, 0.70, 0.76}`; the other boundary remains
   at gap 0.80. Severity 0.76 therefore has minimum gap 0.04.

All axes are ordered from least to most severe. The observed-only gate remains
k=12, cross-kNN <= 0.05, minimum cluster fraction 0.20, and separation SNR >=3.

## Predeclared summaries and diagnostic gate

For each axis and level, report sampling eligibility, accepted-safe count,
false-safe count, safe acceptance rate, and mean constrained/B5 F-score.

- `last_reliable_level`: greatest severity with at least 75% safe acceptance
  and zero false-safe cases.
- `first_rejection_dominant_level`: first later severity with acceptance at or
  below 25%.
- If no transition occurs inside the grid, report a one-sided bound rather than
  inventing an interpolated threshold.

`phase4_diagnostic_supported=true` only if the complete frozen panel runs,
false-safe count is zero, the Phase-3 anchor levels retain at least 75% safe
acceptance, the near-contact endpoint is accepted at most 25%, and at least one
axis exposes an accept-to-reject transition. Whatever the outcome,
`deployment_supported=false`.

## Run

```powershell
python -m pftf_alpha.two_layer_boundary `
  --output benchmark-out/two_layer_boundary_phase4.json
```

## Result

The complete 192-case sweep ran, but the diagnostic gate failed because the
curvature axis produced 15 silent false-safe accepts.

| Axis | Last reliable level | First actual rejection-dominant level | Interpretation |
|---|---:|---:|---|
| curvature | 0.24 | not observed | unsafe acceptance starts at 0.36 |
| tilt_span | 0.40 | 0.55 | gate becomes conservative before local contact |
| overlap_offset | >=2.50 | not observed | all tested levels, including no overlap, remained safe |
| contact_severity | 0.20 | 0.60 | severity 0.60 means minimum gap 0.20 |

The curvature axis exposes the central failure:

| Curvature | Eligible | Accepted | Accepted safe | False safe | Mean cross-kNN | Mean SNR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 8/8 | 8/8 | 8/8 | 0 | 0.002 | 79.71 |
| 0.12 | 8/8 | 8/8 | 8/8 | 0 | 0.006 | 15.24 |
| 0.24 | 8/8 | 8/8 | 8/8 | 0 | 0.005 | 8.05 |
| 0.36 | 8/8 | 8/8 | 2/8 | **6** | 0.020 | 5.22 |
| 0.48 | 6/8 | 6/8 | 0/8 | **6** | 0.040 | 4.19 |
| 0.60 | 3/8 | 3/8 | 0/8 | **3** | 0.054 | 3.39 |

The gate correctly becomes rejection-dominant at tilt span 0.55 and contact
severity 0.60, and it remains safe through every overlap-offset level. But its
global normal-coordinate two-means model cannot detect when paired paraboloids
overlap in that coordinate. Cross-kNN and separation SNR can still pass while
the inferred clusters mix true layers, so this is a genuine observed-policy
blind spot rather than a geometry-score regression.

Across all axes, 118/192 cases were accepted safely and 15 were false-safe.
Artifact: `benchmark-out/two_layer_boundary_phase4.json`.

`phase4_diagnostic_supported=false` and `deployment_supported=false`. The
positive Phase-3 claim must now be narrowed to the tested curvature <=0.24
range until an observed-only curvature/model-adequacy guard is calibrated on
separate data and validated on a new held-out panel.
