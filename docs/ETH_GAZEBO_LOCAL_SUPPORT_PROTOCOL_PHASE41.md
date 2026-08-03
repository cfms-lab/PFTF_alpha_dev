# Phase 41: reserved-source local-support validation protocol

## Frozen candidate

The source-0 calibration selects `support02_dispersion0150mm`:

- retain all source-anchor voxel centroids;
- ignore transformed target cells overlapping an anchor cell;
- add a target-only 0.75 m cell only when at least two distinct accepted scans
  contribute and their RMS within-cell dispersion is at most 0.15 m;
- reconstruct with the unchanged global alpha 1.00 m.

Calibration artifact SHA-256:
`dfc37f2bbc011e89646bd5a9a89744b9e065d78026b1dcdb58f900f64b18ecae`.

## Reserved validation panel

The 17 Phase-40 validation sources and development source 0 are excluded. The
reserved panel is defined without reading reconstruction references as every
remaining source with at least six direct target predictions. The frozen source
indices and pair counts are:

| Source | Direct pairs | p90 accepted |
|---:|---:|---:|
| 1 | 29 | 29 |
| 17 | 13 | 13 |
| 18 | 12 | 12 |
| 21 | 9 | 9 |
| 22 | 8 | 8 |
| 23 | 7 | 7 |
| 24 | 6 | 6 |

All 84 predictions are p90-accepted. Phase 41 therefore tests added point-local
spatial evidence rather than another scan-level reject decision.

## Frozen baselines and endpoints

Each source uses the same row-index heldout split, ROI, voxel sizes, alpha, and
4,096-point matched surface evaluation as Phase 40. Three meshes are compared:

1. anchor-only fail-closed baseline;
2. all p90-accepted target-only cells plus anchor baseline;
3. selected local-support route.

The primary gate requires:

1. all 21 meshes materialize;
2. every validation case adds at least one corroborated target-only cell;
3. mean local geometry loss is below both baseline means;
4. mean local F-score is at least mean anchor F-score minus 0.025;
5. mean local recall is at least mean anchor recall minus 0.01.

Topology is reported but excluded from support because there is no full-scene
topology ground truth.

## Information boundary

Before the protocol artifact was generated, none of the seven reserved source
heldout endpoints had been evaluated. The evaluator may open exactly the 32
Hokuyo point-cloud members after protocol commit. It must not consume:

- the 17 Phase-40 validation endpoints;
- the Gazebo Leica pose member;
- any registration correctness label.

Protocol artifact SHA-256:
`f3c48a60657a8877f4cde330a8f917b99bd1a3521ed5ffd7682565145fce60a8`.

A positive result supports only observed-only point-local input routing at a
fixed alpha. It does not establish a spatially varying alpha complex, correct
topology, full-scene truth, deployed trimming, or deployment.
