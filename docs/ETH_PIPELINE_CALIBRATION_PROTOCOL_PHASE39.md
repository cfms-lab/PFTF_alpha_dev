# ETH predictor calibration protocol: Phase 39

## Purpose and evidence split

Phase 38 showed that the unchanged 3DMatch-scale predictor produced zero
registrations within RRE < 15 degrees and RTE < 0.30 m on ETH Mountain Plain.
The Mountain Plain pose values are now open, so Phase 39 uses that scene only
for explicit **predictor calibration**. It cannot become fresh validation
again.

The scene-relative rotation p90 guard is not a calibration objective. Its
cutoff, midrank tie handling, retention gate, and rejection gate remain frozen.

## Frozen candidate grid

The bounded grid contains four voxel sizes and two pipeline endings:

- voxel size: 0.10, 0.20, 0.30, or 0.50 m;
- FPFH normal radius: 2x voxel, maximum 30 neighbors;
- FPFH feature radius: 5x voxel, maximum 100 neighbors;
- FGR correspondence distance: 0.5x voxel, 64 iterations and the existing
  Phase-37 option values;
- either stop after FGR or refine its transform with point-to-plane ICP;
- ICP correspondence distance: 1.5x voxel, relative fitness/RMSE tolerances
  `1e-6`, maximum 50 iterations.

Every candidate predicts the same 435 nonconsecutive pairs. The selected
candidate maximizes the count satisfying the unchanged strict conjunction
RRE < 15 degrees and RTE < 0.30 m. A tie selects the smaller voxel and then the
pipeline without ICP. Guard performance, prediction rotation percentiles, and
any future validation labels do not participate in selection. If every
candidate has zero correct predictions, calibration is non-viable and Phase 39
must stop without opening another scene's labels.

## Next boundary

If calibration is viable, the selected parameter dictionary will be hash
frozen. A different ETH scene will then be chosen and downloaded. Its scan
members may be read to generate the complete prediction and p90 decision
artifacts, but its Leica pose values must remain unopened until those artifacts
and their code are committed.

This phase calibrates an Open3D-based predictor. It does not independently
implement FPFH, FGR, or ICP; identify physical correspondences; reconstruct an
alpha shape; or support deployment.
