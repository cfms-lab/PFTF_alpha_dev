# Phase 37: independent Open3D pipeline transfer audit

## Purpose and evidence status

Phase 36 showed that the unchanged scene-relative rotation p90 guard transfers
to FPFH and Spin-Images logs, but those logs were precomputed by one shared
3DMatch toolbox pipeline. Phase 37 closes that generation gap by executing a
separate Open3D 0.19.0 FPFH plus Fast Global Registration (FGR) pipeline
locally on the official 3DMatch fragment PLY files.

This is a **fixed-parameter transfer audit, not fresh held-out validation**.
Both redkitchen and Maryland hotel3 labels were opened in Phases 32--34.
Phase-37 generation therefore proves only that the result is reproducible from
raw fragments under a different executable pipeline. It cannot erase the
earlier label exposure.

## Frozen pre-result contract

The following contract was fixed before the Phase-37 Open3D prediction artifact
was generated. No item may be changed after the official labels are joined.

1. Software: Open3D 0.19.0 in an isolated Python 3.12 environment.
2. Input scenes: all 60 redkitchen fragments and all 37 Maryland hotel3
   fragments from the already frozen official archives.
3. Pair universe: every `source < target` pair with `target - source > 1`;
   1,711 redkitchen pairs and 630 Maryland pairs. There is no overlap, label,
   fitness, or output-quality filtering.
4. Preprocessing follows the official Open3D global-registration tutorial:
   0.05 m voxel downsampling; normals at 0.10 m with 30 neighbors; FPFH at
   0.25 m with 100 neighbors.
5. FGR uses maximum correspondence distance 0.025 m, division factor 1.4,
   relative scale, 64 iterations, tuple scale 0.95, maximum 1,000 tuples, and
   tuple testing. A deterministic per-pair seed derived from base seed 370803
   is set before each call.
6. The generator accepts fragment roots only. It verifies each fragment
   archive and compares every extracted PLY SHA-256 with its archive member.
   It has no evaluation-archive argument and cannot read `3dmatch.log`,
   `gt.log`, or `gt.info`.
7. Both complete prediction sets must be written before evaluation begins.
   The evaluator then materializes all rotation angles, within-scene midrank
   percentiles, and `< 0.90` decisions before decoding either scene's labels.
8. Every scene independently must improve precision, retain at least 90% of
   correct predictions, and reject at least 10% of incorrect predictions. One
   failed scene fails the Phase-37 panel. Zero correct predictions is a failed
   gate, not an exception or an omitted scene.

### Matrix-direction implementation correction

A preliminary, uncommitted v1 execution passed fragment 1 as Open3D's moving
`source` and fragment 2 as its fixed `target`. Its audit produced zero correct
registrations in both scenes. This output is invalidated as a convention error,
not retained as a scientific result. The official 3DMatch code states that the
stored transform aligns fragment 2 to fragment 1: `register2Fragments.m`
estimates fragment-2-to-fragment-1 and `getGtInfoLog.m` writes the same
direction under the `(fragment1, fragment2)` header. Open3D's returned matrix
aligns its moving source to its fixed target. Final schema v2 therefore calls
Open3D with fragment 2 as moving source and fragment 1 as fixed target.

This correction changes only API argument direction. It does not change the
scene panel, pair universe, features, parameters, random seeds, p90 guard, or
gates. The correction and the invalidated-v1 history are recorded in the
prediction artifact itself.

Parameter sources:

- <https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html>
- <https://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.FastGlobalRegistrationOption.html>

## Implementation

- label-free generator: `src/pftf_alpha/open3d_fgr_pipeline.py`
- fixed-protocol evaluator:
  `src/pftf_alpha/independent_pipeline_rotation_audit.py`
- tests: `tests/test_independent_pipeline_rotation_audit.py`

## Result

The corrected schema-v2 pipeline generated all 2,341 declared predictions.
Four representative registrations, two per scene, were regenerated with their
recorded pair seeds; all four stored transformation matrices reproduced with
zero maximum absolute difference. Fragment manifests were also verified
against every member of the two official archives.

| Scene | Predictions | Correct | Base precision | Guarded precision | Correct retention | Incorrect rejection | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| redkitchen | 1,711 | 178 | 10.40% | 11.56% | 100.00% | 11.15% | pass |
| Maryland hotel3 | 630 | 8 | 1.27% | 1.41% | 100.00% | 10.13% | pass |

The pooled result changes precision from 186/2,341 = 7.95% to 186/2,107 =
8.83%. All 186 correct predictions are retained and 234/2,155 incorrect
predictions are rejected (10.86%). Both scenes independently pass all three
unchanged gates. Therefore:

- `external_method_generation_reproduced=true`;
- `independently_generated_prediction_artifact_supported=true`;
- `phase37_fixed_parameter_audit_supported=true`;
- `independent_end_to_end_pipeline_transfer_supported=true`.

The result does **not** support `fresh_label_blind_validation`: the two labels
were opened in earlier phases. It also does not show that the all-pair Open3D
pipeline is a strong standalone detector; its base precision is deliberately
low because every nonconsecutive pair is emitted without an overlap filter.
`independent_algorithm_implementation_supported=false` because Open3D supplies
FPFH and FGR. Physical correspondence identity, alpha-shape reconstruction,
and deployment remain unsupported.

Artifacts:

- `benchmark-out/open3d_fgr_predictions_phase37.json`, 2,285,690 bytes,
  SHA-256
  `7276bed58266349a536101d0d95d825c3084226e656065f498b9700347414515`;
- `benchmark-out/independent_pipeline_rotation_audit_phase37.json`, 1,280,563
  bytes, SHA-256
  `7eece8dacc94e051760ddaea4dd6c7c9b77f2fb532f608881bf4977bc8f36db9`.

The exact extracted-fragment manifest hashes are
`bd01f863a710f5a3a9cd212a4ca27665141e4f4767132fe9e3111d89c01d435d`
for redkitchen and
`519244d8e631a7acbdeccca05ba26c13aaf02d6707b09edd6b59406caad3cc2c`
for Maryland hotel3.
