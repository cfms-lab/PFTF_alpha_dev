# Phase 40: ETH Gazebo real alpha-reconstruction shadow protocol

## Purpose

Phase 39 established that the frozen within-scene p90 rotation observation
improves registration precision on fresh ETH Gazebo Summer data while retaining
all correct registrations. Phase 40 asks the next, narrower question: does using
that observation to route real scan inputs improve a fixed-alpha reconstruction
shadow?

This is not another registration benchmark. Registration labels are excluded
from construction and evaluation. The endpoint is source-view heldout geometric
consistency plus descriptive mesh topology.

## Development/validation separation

Source index 0 was used once to verify that the data path, ROI, Open3D alpha
construction, and endpoint calculations run. Its result was observed before
this protocol was frozen, so source 0 is a development case and is excluded
from every Phase-40 validation summary.

The validation panel contains every other source index that has at least one
frozen Phase-39 p90 accept and at least one frozen p90 reject. This rule uses
only the hash-locked pre-label decision artifact. It yields 17 sources:

`2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20`.

No validation source/reference endpoint was evaluated before this protocol was
materialized.

## Frozen inputs

- Phase-39 prediction SHA-256:
  `ed25ac05393d3a9270bef04e99bf79870b8eddd4c0ba6cb0e45d7bff2931900e`
- Phase-39 pre-label decision SHA-256:
  `20dcacaed83575d7c997657d61de2a5e797cfb7d5a3fd3d2ecaea0e070a5f6fb`
- ETH Gazebo archive SHA-256:
  `614052861d6b599c576209965e504682ca78767ac0d4acc112565cf467acb579`
- Phase-40 protocol SHA-256:
  `cd829c3e1c1d9585ccef5c6fa98311e6a62507d9f57fb55e960405dd53ba635b`

The evaluator may open exactly the 32 `csv_local/Hokuyo_*.csv` members. It must
not open `leica/pose_scanner_leica.csv` or consume the Phase-39 post-label audit.

## Frozen construction

For each validation source:

1. Original row indices satisfying `index mod 5 == 0` form an evaluation-only
   heldout reference. Other source rows form the observed anchor.
2. The observed anchor is voxel-downsampled at 0.50 m; the heldout reference is
   voxel-downsampled at 0.25 m.
3. The ROI is the per-axis `[0.005, 0.995]` observed-anchor quantile box expanded
   by 1 m. Reference and reconstructed inputs use this same ROI.
4. The unguarded baseline fuses every frozen direct target-to-source prediction.
   The guarded input fuses only predictions accepted by the frozen p90 decision.
5. Both fused inputs are voxel-downsampled at 0.75 m.
6. Both meshes use Open3D alpha shape with alpha fixed at 1.00 m, exactly twice
   the Phase-39 registration voxel. There is no validation-reference alpha scan.

This experiment tests observation-based input routing at one frozen alpha. It
does not yet implement a pointwise PFTF alpha field or optimize alpha on real
validation references.

## Frozen endpoints and gate

Each mesh is sampled at 4,096 points with matched deterministic random seeds.
Geometry is measured against the heldout source-view points using normalized
symmetric squared Chamfer, normalized Hausdorff, thresholded precision, recall,
and F-score. The primary loss is normalized Chamfer plus normalized Hausdorff.

`geometry_shadow_supported=true` requires all three conditions:

1. mean guarded geometry loss is lower than the unguarded mean;
2. mean guarded F-score is not lower;
3. mean guarded recall is no more than 0.01 below the baseline.

Connected components, Betti numbers, Euler characteristic, boundary edges, and
nonmanifold edges are reported for both meshes. They are descriptive only:
there is no full-scene mesh or topology ground truth, so
`topology_correctness_supported` remains false regardless of direction.

## Claim boundary

A positive result supports only a real-data fixed-alpha reconstruction shadow
with source-view heldout consistency. It does not establish full-scene surface
accuracy, physical correspondence identity, correct topology, a deployed trim,
an adaptive alpha-selection rule, or deployment safety.
