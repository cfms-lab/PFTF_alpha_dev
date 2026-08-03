# Phase 41: point-local multi-scan support calibration

## Question

Phase 40 showed that the frozen scan-level p90 observation improves aggregate
geometry at one fixed alpha, but its topology result is mixed. Phase 41 tests a
new observed-only spatial signal before changing alpha: can target-only voxels
be accepted only when multiple transformed scans agree locally and have low
within-voxel dispersion?

This is a point-local fail-closed input route. It is not yet a spatially varying
alpha complex.

## Information boundary

Calibration uses only Gazebo source index 0, which was already declared and
opened as the Phase-40 development/runtime case. It may use that source's
heldout reference to select a candidate. It must not use:

- any of the 17 Phase-40 validation endpoints;
- the seven reserved Phase-41 validation source references;
- the Gazebo Leica poses or any registration-correctness label.

The hash-locked Phase-39 pre-label predictions and p90 decisions supply only
target-to-source transforms and scan-level accepts.

## Local observation

The source observation and every p90-accepted transformed target are placed on
the same 0.75 m voxel grid, whose origin is the frozen source ROI lower corner.
The route always retains the source-anchor cell centroid. A target cell that
overlaps an anchor cell adds nothing, preventing transformed data from moving a
trusted anchor cell.

For each target-only cell, two observed quantities are computed:

1. `support`: number of distinct target scans contributing at least one point;
2. `dispersion`: RMS Euclidean displacement of contributing points from their
   cell centroid.

The target-only centroid is retained only if support is at least the candidate
threshold and dispersion is no larger than the candidate threshold. Otherwise
the cell fails closed. Reference points, topology, registration correctness,
fitness, and ICP residuals are not inputs to this decision.

## Development grid and selection

The nine candidates are the Cartesian product:

- minimum distinct-scan support: `2, 3, 4`;
- maximum dispersion: `0.15, 0.20, 0.25 m`.

All candidates reconstruct with the unchanged alpha `1.00 m`. They are compared
with two development baselines on the same voxel grid:

- anchor-only fallback;
- all p90-accepted target-only cells plus the anchor.

A candidate is eligible only if it adds at least one target-only cell, has lower
geometry loss than both baselines, has F-score no more than `0.025` below the
anchor, and has recall no more than `0.01` below the anchor. Among eligible
candidates, choose minimum geometry loss; ties prefer higher support and then
lower dispersion. Topology is recorded but not used for selection.

If no candidate is eligible, Phase 41 stops without evaluating reserved
references. If one is selected, freeze its artifact hash and preregister a
validation on sources `1, 17, 18, 21, 22, 23, 24`. These sources were excluded
from Phase 40 and each has at least six direct target scans.
