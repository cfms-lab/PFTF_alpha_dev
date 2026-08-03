# Frozen scene-relative rotation validation: Phase 35

## Validation question and frozen boundary

Phase 35 opens the six scenes reserved by Phase 34 and applies the exact
unchanged `scene_relative_prediction_rotation_midrank_percentile` rule. The
official 3DMatch geometric-registration benchmark provides eight real-scene
fragment sets and external registration predictions:
<https://3dmatch.cs.princeton.edu/>.

The frozen rule computes the principal rotation angle of every eligible
external prediction, assigns deterministic within-scene midrank percentiles,
and accepts percentile `< 0.90`. Every scene must independently satisfy:

- guarded precision strictly greater than baseline precision;
- correct-prediction retention at least 90%;
- incorrect-prediction rejection at least 10%.

Any one failed scene fails the panel. The feature, cutoff, exact-value tie
handling, and gates were not changed after validation access.

## Minimum-information intake

The rotation-only feature does not use fragment coordinates. Phase 35 therefore
downloads only the six official evaluation ZIP files, not the approximately
310 MB of fragment PLY archives. Each local ZIP is ignored by Git and is not
redistributed. The official 3DMatch and SUN3D pages request dataset citation but
the accessed pages do not state an explicit SUN3D data license.

| Scene | Bytes | MD5 | SHA-256 |
|---|---:|---|---|
| home_at scan1 | 53,414 | `ce04d973763ea01a86c042dffc356724` | `0645890e675444a7b6a49e1e3ac1c443a55a761b65fb275dd5308980a200f757` |
| home_md scan9 | 71,399 | `657fbdd5d3f75313b017ffee505cb4d8` | `272b83fcb74cb0faedad4c9f614eb9eedac83bc000de60591422d3daa332cecf` |
| hotel_uc scan3 | 62,513 | `9d6dc696247d5f462ac08b6cbc3a479e` | `cc07c279a355756167dcd850184031f59578b56ed916c732020c47ef7420e957` |
| Maryland hotel1 | 31,922 | `f5a4488f41ec5ce2004d77063e8bf5e5` | `1a7822280320c5b6652f30584735ce518af4f3bfd311f2e0a98696d2bbf93c70` |
| MIT studyroom | 102,848 | `bcdd48d415d54b9e51b3f1998ea7b2f9` | `f3f49cb14224e777dbb64244a6efae9e43a2d07fc69469e04fd650832535b14e` |
| MIT lab | 27,637 | `d11b00d76ae0b253abb4435ec31c62ea` | `4dce89e4d24ba883422cf6b782497889a9ed02f01e5ea0c7f21d19590e128660` |

Every ZIP has the exact directory plus `3dmatch.log`, `gt.log`, and `gt.info`
allowlist. The evaluator first verifies the exact Phase-34 artifact SHA-256,
then all archive hashes and central directories. It reads all six prediction
logs and materializes all six angle, midrank, and accept/reject sets before any
`gt.log` or `gt.info` member is decompressed or decoded. Exact archive hashing
does read the ZIP bytes before this point, but no label member content is
interpreted. Phase-34 artifact SHA-256:
`805f056fdf50c80aa89fd74d1bba67968ab8405279b481bf9051494f655ea9d8`.

## Scene-level results

The official non-consecutive-pair filter leaves 1,550 predictions in total.
The gate is evaluated per scene, not on pooled totals.

| Scene | Eligible | Baseline correct | Guard accepted | Guard correct | Precision | Guard precision | Correct retention | Incorrect rejection | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| home_at scan1 | 236 | 83 | 212 | 82 | 35.17% | 38.68% | 98.80% | 15.03% | pass |
| home_md scan9 | 339 | 97 | 305 | 97 | 28.61% | 31.80% | 100.00% | 14.05% | pass |
| hotel_uc scan3 | 199 | 143 | 179 | 136 | 71.86% | 75.98% | 95.10% | 23.21% | pass |
| Maryland hotel1 | 111 | 46 | 100 | 46 | 41.44% | 46.00% | 100.00% | 16.92% | pass |
| MIT studyroom | 550 | 148 | 495 | 148 | 26.91% | 29.90% | 100.00% | 13.68% | pass |
| MIT lab | 115 | 23 | 103 | 23 | 20.00% | 22.33% | 100.00% | 13.04% | pass |

Precision gains are respectively `+3.51`, `+3.19`, `+4.12`, `+4.56`, `+2.99`,
and `+2.33` percentage points. Descriptively pooled, the guard retains 532/540
correct predictions and rejects 148/1,010 incorrect predictions; pooled
precision changes from 34.84% to 38.16%. These pooled values are not substituted
for the preregistered per-scene gates.

## Decision and claim boundary

- `phase34_design_supported=true`;
- `held_out_validation_artifacts_accessed=true`;
- `phase35_validation_supported=true`;
- `held_out_validation_supported=true`;
- `cross_scene_real_registration_supported=true`;
- `real_registration_labels_supported=true`;
- correspondence identity, trimmed reconstruction, and deployment remain
  unsupported.

The positive claim is narrow: on the fixed external 3DMatch prediction logs,
large scene-relative predicted rotations are enriched for official registration
errors across all six held-out scenes. The rule rejects roughly 10% of each
batch by construction, so the 10% rejection and 90% retention gates are modest;
the evidence comes from consistent per-scene precision improvement and the
observed margins above those gates. This does not show that the rule transfers
to another registration algorithm, operates pairwise without a complete batch,
identifies physical correspondences, or improves alpha-shape reconstruction.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pftf_alpha.scene_relative_rotation_validation `
  --data-root benchmark-data\3dmatch_phase35_evaluation `
  --phase34-artifact benchmark-out\scene_relative_rotation_guard_phase34.json `
  --output benchmark-out\scene_relative_rotation_validation_phase35.json
```

Artifact SHA-256:
`c07cb04e82ef597f5c7480fad1181fd3d8141e2d7fbc2ab8d0a3c4e644179372`.
