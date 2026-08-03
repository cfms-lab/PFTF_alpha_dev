# Fresh external rotation audit: Phase 38

## Result

Phase 38 completed the first genuinely fresh, label-value-blind external test
of the fixed scene-relative rotation p90 guard. The result is **negative** for
transfer on ETH Mountain Plain:

| Metric | Base | p90 accepted |
|---|---:|---:|
| Predictions | 435 | 391 |
| Frozen-correct | 0 | 0 |
| Frozen-incorrect | 435 | 391 |
| Precision | 0.00% | 0.00% |
| Correct retention | -- | 0.00% |
| Incorrect rejection | -- | 10.11% |

The p90 guard rejects 44/435 incorrect predictions, so the 10% incorrect
rejection sub-gate passes. The underlying fixed Phase-37 FPFH+FGR predictor,
however, produces no registration satisfying the preregistered conjunction of
RRE < 15 degrees and RTE < 0.30 m. Therefore the panel lacks both correct and
incorrect predictions, precision cannot improve, correct retention fails, and
the overall transfer gate fails:

- `fresh_label_blind_validation_supported=false`;
- `fresh_external_pipeline_transfer_supported=false`.

This is a model/pipeline-domain failure, not a failure of the label-blind
execution boundary. The full predictions and all p90 decisions were committed
before the Leica pose values were opened.

## Frozen execution order

1. Commit `3528297` preregistered the ETH archive identity, all 31 scans, all
   435 pairs, unchanged Open3D parameters, transform direction, RRE/RTE rule,
   p90 cutoff, and evidence gates.
2. The generator opened exactly the 31 Hokuyo scan members and wrote all 435
   predictions. Prediction SHA-256:
   `71dc13f8ef8702dc54cc9787ddefd537aaf6de82cf5faab448679abd52bff708`.
3. Before labels, the decision materializer wrote all 435 midrank p90
   decisions. Decision SHA-256:
   `26f069fa77841dfb446185d01809a062b242af3f1517e605d68105aab43850c0`.
4. Commit `dffe72d` fixed both programs, artifacts, and hashes.
5. Only then did the evaluator open
   `leica/pose_scanner_leica.csv`, construct ground-truth target-to-source as
   `inv(global_from_source) @ global_from_target`, and join frozen labels.

Final audit artifact:
`benchmark-out/fresh_external_rotation_audit_phase38.json`, SHA-256
`0e27568e7b7e9e1dfe98708b7984a437635836fde1b51693aca4daa992496f2b`.

## Diagnostic boundary

Seventy-five predictions meet RRE < 15 degrees, but none meet RTE < 0.30 m;
the minimum observed RTE is 0.6124 m. This identifies translation failure of
the fixed cross-domain registration pipeline as the immediate bottleneck. It
does not justify changing the Phase-38 threshold or rerunning this now-opened
scene as fresh validation.

A scientifically distinct next phase should use Mountain Plain only as an
explicit calibration/diagnostic scene, freeze an ETH-compatible predictor, and
validate it once on a different untouched ETH scene. Phase 37's positive
3DMatch result remains valid within its fixed benchmark scope, but Phase 38
does not support extending that claim to ETH outdoor LiDAR.

The evidence still does not implement FPFH/FGR independently of Open3D,
identify physical correspondences, reconstruct an alpha shape, or support
deployment.

Primary sources: [ETH dataset record](https://doi.org/10.3929/ethz-b-000721626)
and [Open3D global registration tutorial](https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html).
