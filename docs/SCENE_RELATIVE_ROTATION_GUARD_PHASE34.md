# Scene-relative rotation-tail guard design: Phase 34

## Question and evidence boundary

Phase 34 redesigns the real-registration observation after the absolute
global/local/tail route failed on both opened 3DMatch scenes. It uses only the
already opened redkitchen and Maryland hotel3 artifacts as design data. It does
not download, read, or label any of the six remaining official scenes.

The official 3DMatch registration benchmark contains eight real-scene fragment
sets, and each fragment is a surface point cloud fused from 50 depth frames:
<https://3dmatch.cs.princeton.edu/>.

Design scenes:

- `7-scenes-redkitchen`;
- `sun3d-hotel_umd-maryland_hotel3`.

Untouched validation scenes, frozen before any new archive access:

- `sun3d-home_at-home_at_scan1_2013_jan_1`;
- `sun3d-home_md-home_md_scan9_2012_sep_30`;
- `sun3d-hotel_uc-scan3`;
- `sun3d-hotel_umd-maryland_hotel1`;
- `sun3d-mit_76_studyroom-76-1studyroom2`;
- `sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika`.

## Design search and rejected directions

This is an explicitly label-informed design phase, not a preregistered
validation. Two initial candidate families were rejected on the opened scenes:

1. reweighting the existing 32 registration/patch summaries with ridge scores;
2. adding raw reciprocal-support extent, voxel occupancy, centroid offset, and
   covariance-entropy features before ridge scoring.

Neither absolute-score family retained at least 90% of correct predictions in
both scene-to-scene directions. That failure identified scene-level score shift,
not a lack of additional continuous features, as the immediate problem.

The promoted design candidate uses only the external predicted rigid transform.
For every eligible prediction, it computes the principal rotation angle in
`[0, pi]`. Within each scene, all angles are converted to deterministic empirical
midrank percentiles:

`percentile = (midrank - 0.5) / prediction_count`.

The candidate accepts a prediction exactly when its scene-relative rotation
percentile is below `0.90`. Thus the feature is label-free at execution but
batch-relative: a pair cannot be classified without the scene's complete
external prediction set. It is spatial-transform evidence, not correspondence
identity, local surface evidence, or PFTF-alpha reconstruction evidence.

All rotation angles, midranks, and accept/reject decisions for both design
scenes are materialized before either scene's official correctness labels are
joined.

## Frozen design gate

Each design scene must independently satisfy all three conditions:

- guarded precision is strictly greater than its unguarded precision;
- correct-prediction retention is at least 90%;
- incorrect-prediction rejection is at least 10%.

These are design gates only. Passing them does not establish held-out or
cross-scene support.

## Results

| Scene | Route | Accepted | Correct | Incorrect | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| redkitchen | base | 531 | 383 | 148 | 72.13% | 85.30% |
| redkitchen | rotation `< p90` | 478 | 346 | 132 | 72.38% | 77.06% |
| Maryland hotel3 | base | 61 | 15 | 46 | 24.59% | 57.69% |
| Maryland hotel3 | rotation `< p90` | 55 | 14 | 41 | 25.45% | 53.85% |

| Scene | Correct retention | Incorrect rejection | Precision gain | Design gate |
|---|---:|---:|---:|---|
| redkitchen | 90.34% | 10.81% | +0.26 percentage points | pass |
| Maryland hotel3 | 93.33% | 10.87% | +0.86 percentage points | pass |

`phase34_design_supported=true`, but the margins are deliberately described as
small. The rule rejects 37 correct and 16 incorrect predictions on redkitchen,
and one correct plus five incorrect predictions on Maryland hotel3. A nominal
design pass is not evidence that large predicted rotations are generally
incorrect.

## Decision and frozen validation contract

- `phase34_design_supported=true`;
- `held_out_validation_artifacts_accessed=false`;
- `held_out_validation_supported=false`;
- `cross_scene_real_registration_supported=false`;
- correspondence identity, reconstruction, and deployment remain unsupported.

Phase 35, if run, must use the exact `0.90` midrank rule without changing the
feature, cutoff, tie handling, or gates. It should open the six frozen scenes as
one validation panel and report every scene separately. Any failed scene fails
the cross-scene claim; validation results must not be used to revise the rule.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pftf_alpha.scene_relative_rotation_guard `
  --phase32-artifact benchmark-out/threedmatch_registration_guard_phase32.json `
  --phase33-artifact benchmark-out/threedmatch_transfer_audit_phase33.json `
  --output benchmark-out/scene_relative_rotation_guard_phase34.json
```

Artifact SHA-256:
`805f056fdf50c80aa89fd74d1bba67968ab8405279b481bf9051494f655ea9d8`.
