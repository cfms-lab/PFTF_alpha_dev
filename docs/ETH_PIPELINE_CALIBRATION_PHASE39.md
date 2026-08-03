# ETH predictor calibration result: Phase 39

## Selected predictor

The preregistered eight-candidate calibration completed on the already-opened
ETH Mountain Plain scene. Candidate selection used only the number satisfying
strict RRE < 15 degrees and RTE < 0.30 m; p90 guard performance was not
computed or used for selection.

| Candidate | Voxel | ICP | Correct / 435 |
|---|---:|:---:|---:|
| fgr_v010 | 0.10 m | no | 0 |
| fgr_icp_v010 | 0.10 m | yes | 13 |
| fgr_v020 | 0.20 m | no | 4 |
| fgr_icp_v020 | 0.20 m | yes | 51 |
| fgr_v030 | 0.30 m | no | 7 |
| fgr_icp_v030 | 0.30 m | yes | 77 |
| fgr_v050 | 0.50 m | no | 13 |
| **fgr_icp_v050** | **0.50 m** | **yes** | **83** |

The selected predictor uses 0.50 m voxels, 1.0 m/30-NN normals, 2.5 m/100-NN
FPFH, FGR at 0.25 m, and point-to-plane ICP at 0.75 m for at most 50
iterations. It produces 83 correct and 352 incorrect calibration predictions.
This is a calibration result, not fresh evidence for the rotation guard.

Calibration artifact:
`benchmark-out/eth_pipeline_calibration_phase39.json`, SHA-256
`1001b214f6b69be4bfe21bade1a0100a7bb89e357b58796edb00031c077228d5`.

## Frozen validation transition

The selected parameter dictionary is now frozen. Phase 39 will use ETH Gazebo
Summer as a separate untouched outdoor validation scene because it uses the
same dataset family and sensor protocol but a different semi-structured scene.
The full prediction set and unchanged p90 decisions must exist and be committed
before `pose_scanner_leica.csv` is opened. Mountain Plain must not participate
in the validation result.

The fixed correctness and guard gates remain unchanged from Phase 38. A
negative Gazebo result will be retained as negative evidence rather than used
to tune this pipeline again.
