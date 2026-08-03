# ETH Gazebo Summer fresh validation: Phase 39

## Result

Phase 39 succeeds on a once-only, label-value-blind ETH Gazebo Summer
validation after predictor calibration on the separate, opened Mountain Plain
scene.

| Metric | Base | p90 accepted |
|---|---:|---:|
| Predictions | 465 | 418 |
| Frozen-correct | 135 | 135 |
| Frozen-incorrect | 330 | 283 |
| Precision | 29.03% | 32.30% |
| Correct retention | -- | 100.00% |
| Incorrect rejection | -- | 14.24% |

The unchanged scene-relative rotation p90 guard removes 47 incorrect
predictions and no correct prediction. Precision increases by 3.26 percentage
points. The scene contains both correct and incorrect base predictions, and
all preregistered precision, retention, and rejection gates pass.

Therefore:

- `fresh_label_blind_validation_supported=true`;
- `calibrated_external_pipeline_transfer_supported=true`;
- `fresh_external_rotation_guard_transfer_supported=true`.

## Provenance and label boundary

1. Mountain Plain calibration candidates and selection rules were committed as
   `556efa0` before calibration. The selected `fgr_icp_v050` result was frozen
   in artifact SHA-256
   `1001b214f6b69be4bfe21bade1a0100a7bb89e357b58796edb00031c077228d5`.
2. Gazebo archive identity, 465 pairs, selected parameters, correctness rule,
   and unchanged p90 gates were committed as `d4174a1` before any Gazebo scan
   or pose member content was opened.
3. The generator opened exactly the 32 scan members and wrote all predictions,
   SHA-256
   `ed25ac05393d3a9270bef04e99bf79870b8eddd4c0ba6cb0e45d7bff2931900e`.
4. A separate program wrote all 465 p90 decisions before labels, accepting 418
   and rejecting 47. Decision SHA-256:
   `20dcacaed83575d7c997657d61de2a5e797cfb7d5a3fd3d2ecaea0e070a5f6fb`.
5. Both programs and hashes were committed as `032a3c3`. Only afterward did
   the evaluator open the single frozen Leica pose member.

Final audit artifact:
`benchmark-out/eth_gazebo_rotation_audit_phase39.json`, SHA-256
`5703052834ad3f5c04dd3381f4153a47dc718e15c35b9f489ae04ce0a629f9f0`.

## Claim boundary

Phase 39 resolves the Phase-38 zero-success predictor problem without changing
the Phase-38 result. It supports a calibrated Open3D FPFH+FGR+ICP pipeline and
fresh transfer of the fixed rotation guard to a different ETH outdoor scene.
It does not independently implement FPFH, FGR, or ICP. It also does not yet
identify physical correspondence identity, run an alpha-shape reconstruction
from the guarded observations, validate topology/geometry improvement, or
support deployment.

The next scientifically distinct step should stop adding registration scenes
and connect the validated local/spatial observation to the alpha-field or
reconstruction pipeline in a real-data shadow experiment. That experiment
must compare reconstruction outcomes with and without the observation while
keeping the current registration evidence labels separate.

Primary source: [ETH registration dataset](https://doi.org/10.3929/ethz-b-000721626).
