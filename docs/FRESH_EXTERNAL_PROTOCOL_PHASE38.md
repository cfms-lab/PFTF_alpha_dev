# Fresh external validation preregistration: Phase 38

## Outcome before labels

Phase 38 preregisters one fresh real-scan validation scene before reading its
pose values. The selected scene is **ETH Mountain Plain**, published by ETH
Zurich under CC BY 4.0. It contains 31 laser scans and theodolite-derived
scanner poses. The official archive is pinned by repository item, bitstream
UUID, filename, byte count, MD5, and locally computed SHA-256.

The outer ZIP has been downloaded and byte-hashed. Its central-directory names
were enumerated to establish that `csv_local/Hokuyo_0.csv` through
`Hokuyo_30.csv` and `leica/pose_scanner_leica.csv` exist. The label member has
**not** been opened, decompressed, decoded, or numerically inspected. Hashing
the outer container necessarily reads its compressed bytes, so the precise
claim is label-value blindness rather than physical non-possession.

## Frozen protocol

- input scans: all 31 local-coordinate `Hokuyo_<index>.csv` members in integer
  suffix order;
- pair universe: all `source < target` pairs with `target - source > 1`, for
  exactly 435 predictions, with no overlap, pose, fitness, or label filtering;
- prediction pipeline: unchanged Phase-37 Open3D 0.19.0 FPFH+FGR parameters,
  including 5 cm voxel size and 2.5 cm maximum FGR correspondence distance;
- transform convention: each prediction maps target-index local coordinates
  into source-index local coordinates;
- correctness: relative rotation error strictly below 15 degrees **and**
  relative translation error strictly below 0.30 m;
- guard: unchanged within-scene midrank rotation percentile, accepting values
  strictly below p90;
- evidence gate: both correct and incorrect base predictions, improved guarded
  precision, at least 90% correct retention, and at least 10% incorrect
  rejection.

The predictor must produce and hash the complete 435-row artifact before the
evaluator may open the frozen Leica pose member. Neither the correctness
thresholds nor the p90 rule may change after that point, even if the result is
negative. The preregistration artifact is
`benchmark-out/fresh_external_protocol_phase38.json`, SHA-256
`1d183f5b6c8dd7eaeb35a6950ac3fdb16e3306f21e187489e5d0272279973649`.

## Completed pre-label execution

The preregistration was committed as `3528297` before any label value was
opened. The unchanged predictor then read exactly the 31 frozen Hokuyo members,
downsampled 81,982--105,323 raw points per scan to 19,281--26,393 points, and
materialized all 435 FGR predictions in about 17 minutes 26 seconds. Prediction
artifact:

- `benchmark-out/eth_open3d_fgr_predictions_phase38.json`;
- SHA-256
  `71dc13f8ef8702dc54cc9787ddefd537aaf6de82cf5faab448679abd52bff708`;
- `complete_prediction_set_materialized=true`;
- `ground_truth_label_member_opened=false`.

A separate label-free program then used only that hash-locked JSON to
materialize the complete midrank p90 decision set. It accepts 391 and rejects
44 predictions before labels. Decision artifact:

- `benchmark-out/eth_rotation_decisions_phase38.json`;
- SHA-256
  `26f069fa77841dfb446185d01809a062b242af3f1517e605d68105aab43850c0`;
- `complete_decision_set_materialized=true`;
- `label_values_accessed=false`.

Only after these two artifacts and their code are committed may the evaluator
open `leica/pose_scanner_leica.csv` and join the frozen correctness labels.

## Evidence boundary

This phase can test fresh label-blind transfer of the fixed rotation guard to a
different real sensor, scene scale, and benchmark. It does not implement FPFH
or FGR independently of Open3D, identify physical correspondences, reconstruct
an alpha shape, or support deployment.

Primary sources: [ETH dataset record](https://doi.org/10.3929/ethz-b-000721626)
and [Open3D global registration tutorial](https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html).
