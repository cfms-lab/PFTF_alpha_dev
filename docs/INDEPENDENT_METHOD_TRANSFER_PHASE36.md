# Independent-method and cross-benchmark transfer: Phase 36

## Frozen protocol before label decoding

Phase 36 tests the unchanged Phase-34/35 scene-relative rotation p90 rule on
registration predictions produced by two different methods and evaluated on a
different benchmark. The source is the official 3DMatch toolbox at commit
`4c6b2f613adb8bdcc9a62cb04134b7e1379b1a36`:
<https://github.com/andyzeng/3dmatch-toolbox>.

The toolbox evaluation script explicitly supports descriptor names `3dmatch`,
`spin`, and `fpfh`, and lists four ICL-NUIM synthetic benchmark scenes. Phase 36
uses only the independent `spin.log` and `fpfh.log` predictions:

- `iclnuim-livingroom1-evaluation`;
- `iclnuim-livingroom2-evaluation`;
- `iclnuim-office1-evaluation`;
- `iclnuim-office2-evaluation`.

This gives eight method-scene blocks. For each block independently, compute the
principal predicted rotation angle, deterministic within-block empirical
midrank percentile, and accept exactly percentile `< 0.90`. The unchanged gate
requires all three:

- guarded precision strictly greater than baseline precision;
- correct-prediction retention at least 90%;
- incorrect-prediction rejection at least 10%.

Every one of the eight blocks must pass. One failure makes both
`independent_method_transfer_supported=false` and
`cross_benchmark_transfer_supported=false`. No feature, cutoff, tie rule, gate,
scene, or method may be changed after any ground-truth content is decoded.

## Frozen file identities

The clone is kept under ignored `benchmark-data/`; the source logs are not
redistributed. SHA-256 identities were computed before any `gt.log` or
`gt.info` content was decoded.

| Scene | File | Bytes | SHA-256 |
|---|---|---:|---|
| livingroom1 | `fpfh.log` | 141,925 | `3ccc001ea46a4e37ab52e8df9bcd1521523b7c23b5d412094ff185197ee174ea` |
| livingroom1 | `spin.log` | 125,398 | `3beb87f061fcd1c82b821943f1652a6a06ccab23b8785198258ed9f7bd4e5d42` |
| livingroom1 | `gt.log` | 59,507 | `5bee3ead05841d418b6e5533efc269c34ecc7c22e2e33e82de60c288e4907b3f` |
| livingroom1 | `gt.info` | 168,730 | `830154126ba81e58e99092b5084669adcd1fe811b45a84346e89dbebbfffcd28` |
| livingroom2 | `fpfh.log` | 82,106 | `90e824fe015807038a7af1a232d57efc4a6d22bc3b3b1f28225f6c9bf99ca8ab` |
| livingroom2 | `spin.log` | 90,146 | `bb8c9052389f39fe638f1560b6cca443a1549104ebc4cd2a74ffd8ba18742e75` |
| livingroom2 | `gt.log` | 36,197 | `111ea5894862bd14b91ac3fcda28e6cb6325f77e4fef79f7cb757071a919022c` |
| livingroom2 | `gt.info` | 100,990 | `2be9e746e262099a5cf0c4b13aac984ec3675f6cef0864e89106c229ee143d15` |
| office1 | `fpfh.log` | 103,132 | `a05ccbf35ef28caa6eae81f3821360616c1b54883c095cc3d9bbf241abea0d33` |
| office1 | `spin.log` | 96,949 | `8e16c5de61a381c0dacb13a44c337e9609c67e0cab88ebe78cd3f5b9ee29dd7f` |
| office1 | `gt.log` | 42,164 | `6404bcc429483bc499f9961f0536a50c9d1a70726bc75e756e476bfed5051626` |
| office1 | `gt.info` | 118,816 | `ffd354c2ece1e094ce797ad520dd82d9328dd24f1cd96519bc835038e764c39e` |
| office2 | `fpfh.log` | 115,400 | `b695d47f657dc61b277ef88c2cd7d4f126136f0295f37061e9c357971d111631` |
| office2 | `spin.log` | 116,169 | `3e15a45d664554096a358d181a5405ebc49c923e91bbadfd9ffa5a98d000dfdb` |
| office2 | `gt.log` | 32,988 | `f586d4d93ac9f05edd7a58a1ba90a4ef7e5bd8017657637bc77a25ac1e49a509` |
| office2 | `gt.info` | 92,998 | `1ec3f8fadf73e96bee9b6a69a11b8bf366ced268a2a8c48204468b5f6a242c2c` |

The Phase-35 predecessor artifact is frozen at SHA-256
`c07cb04e82ef597f5c7480fad1181fd3d8141e2d7fbc2ab8d0a3c4e644179372`.
The evaluator must verify every file before decoding predictions, decode all
eight prediction logs and materialize all eight decision sets before decoding
any label file, and only then join `gt.log` and `gt.info` with the official
error threshold squared `0.04`.

## Results

All eight method-scene blocks pass the frozen gate.

| Scene | Method | Eligible | Baseline correct | Guard accepted | Guard correct | Precision | Guard precision | Correct retention | Incorrect rejection | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| livingroom1 | FPFH | 567 | 107 | 510 | 107 | 18.87% | 20.98% | 100.00% | 12.39% | pass |
| livingroom1 | Spin | 495 | 116 | 445 | 116 | 23.43% | 26.07% | 100.00% | 13.19% | pass |
| livingroom2 | FPFH | 313 | 59 | 282 | 59 | 18.85% | 20.92% | 100.00% | 12.20% | pass |
| livingroom2 | Spin | 349 | 65 | 314 | 65 | 18.62% | 20.70% | 100.00% | 12.32% | pass |
| office1 | FPFH | 400 | 86 | 360 | 86 | 21.50% | 23.89% | 100.00% | 12.74% | pass |
| office1 | Spin | 376 | 86 | 338 | 86 | 22.87% | 25.44% | 100.00% | 13.10% | pass |
| office2 | FPFH | 455 | 79 | 409 | 79 | 17.36% | 19.32% | 100.00% | 12.23% | pass |
| office2 | Spin | 460 | 101 | 414 | 101 | 21.96% | 24.40% | 100.00% | 12.81% | pass |

FPFH pools 1,735 eligible predictions: precision changes from 19.08% to
21.20%, all 331 correct predictions are retained, and 12.39% of incorrect
predictions are rejected. Spin pools 1,680 predictions: precision changes from
21.90% to 24.35%, all 368 correct predictions are retained, and 12.88% of
incorrect predictions are rejected. Across both methods, all 699 correct
predictions are retained and 343/2,716 incorrect predictions are rejected;
descriptive pooled precision changes from 20.47% to 22.75%. The frozen decision
is based on all eight block gates, not these pooled summaries.

## Decision and claim boundary

- `phase36_panel_supported=true`;
- `independent_method_transfer_supported=true`;
- `cross_benchmark_transfer_supported=true`;
- `prediction_log_provenance_verified=true`;
- `external_method_generation_reproduced=false`;
- `independent_end_to_end_pipeline_transfer_supported=false`;
- `synthetic_registration_labels_supported=true`;
- correspondence identity, alpha-shape reconstruction, and deployment remain
  unsupported.

The positive transfer claim is limited to the committed FPFH and Spin-Images
prediction logs. They are distinct descriptor methods, but both logs were
generated by the 3DMatch toolbox's shared RANSAC registration and log-generation
pipeline. Therefore this is not evidence for an independently implemented
end-to-end registration system. It does show that the rotation-tail ordering is
not unique to the learned 3DMatch descriptor or to the six real held-out scenes.
The complete prediction batch is still required, and the external method
generation was not reproduced locally.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pftf_alpha.independent_method_rotation_transfer `
  --fragments-root benchmark-data\3dmatch-toolbox-phase36\data\fragments `
  --phase35-artifact benchmark-out\scene_relative_rotation_validation_phase35.json `
  --output benchmark-out\independent_method_rotation_transfer_phase36.json
```

Artifact SHA-256:
`9157e15adccdf8dea98e14f96124f826389d3c35bc3ddf04bbcc51e0a00ec24d`.
