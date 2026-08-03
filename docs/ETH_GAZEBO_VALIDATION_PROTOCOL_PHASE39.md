# ETH Gazebo Summer validation preregistration: Phase 39

## Untouched input

The once-only validation scene is ETH Gazebo Summer. The official archive is
frozen as:

- bitstream UUID `20e4fcd9-42d2-470e-8f90-598690fe65e2`;
- filename `gazebo_summer_04-Aug-2011-16_13_22.zip`;
- 1,332,460,435 bytes;
- MD5 `94f59356d881a67d2ce74937133c3246`;
- SHA-256
  `614052861d6b599c576209965e504682ca78767ac0d4acc112565cf467acb579`.

The outer ZIP was downloaded and hashed. Its central-directory names confirm
32 local `Hokuyo_0.csv` through `Hokuyo_31.csv` members and the separate
`leica/pose_scanner_leica.csv` member. No pose member has been opened,
decompressed, decoded, or numerically inspected. As in Phase 38, the archive
is physically present, so the precise boundary is label-value blindness.

## Frozen predictor and audit

- calibration input: Phase-39 Mountain Plain artifact SHA-256
  `1001b214f6b69be4bfe21bade1a0100a7bb89e357b58796edb00031c077228d5`;
- predictor: `fgr_icp_v050` with the exact selected parameter dictionary;
- pair universe: all nonconsecutive source<target pairs, exactly 465;
- matrix direction: target-index local coordinates to source-index;
- correctness: strict RRE < 15 degrees and strict RTE < 0.30 m;
- guard: unchanged scene-relative rotation midrank percentile < 0.90;
- gates: both correct and incorrect predictions, improved precision, at least
  90% correct retention, and at least 10% incorrect rejection.

The generator may open only the 32 Hokuyo members. It must materialize all 465
predictions and a separate program must materialize all 465 p90 decisions.
Those programs, artifacts, and hashes must be committed before a separate
evaluator opens the Leica pose member. No result may change these choices.

Preregistration artifact:
`benchmark-out/eth_gazebo_validation_protocol_phase39.json`, SHA-256
`1711e23cdb29f0f305950c3eb3015309d8dea4f686c4b79a4d1a80c0af335059`.

## Completed pre-label execution

After preregistration commit `d4174a1`, the calibrated generator opened exactly
the 32 Hokuyo members and materialized all 465 FGR+ICP predictions. It did not
open a pose member. Prediction artifact:

- `benchmark-out/eth_gazebo_predictions_phase39.json`;
- SHA-256
  `ed25ac05393d3a9270bef04e99bf79870b8eddd4c0ba6cb0e45d7bff2931900e`;
- `complete_prediction_set_materialized=true`;
- `validation_label_member_opened=false`.

The separate hash-locked decision materializer then accepted 418 and rejected
47 predictions under unchanged p90, still without label access. Decision
artifact:

- `benchmark-out/eth_gazebo_rotation_decisions_phase39.json`;
- SHA-256
  `20dcacaed83575d7c997657d61de2a5e797cfb7d5a3fd3d2ecaea0e070a5f6fb`;
- `complete_decision_set_materialized=true`;
- `validation_label_values_accessed=false`.

These programs and hashes must now be committed. Only after that commit may the
post-label evaluator open the single frozen Leica pose member.
