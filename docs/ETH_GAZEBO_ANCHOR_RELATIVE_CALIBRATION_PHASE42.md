# Phase 42: anchor-relative local geometry calibration

## Motivation

Phase 41 showed that target scans can agree within a voxel while forming a
coherent cluster that is misplaced relative to the trusted source anchor.
Phase 42 therefore keeps the frozen Phase-41 support/dispersion gate and adds
three target-to-anchor observations before a target-only cell is reconstructed.

This remains fail-closed point routing at global alpha 1.00 m. It is not a
spatially varying alpha complex.

## Development information boundary

Only two opened development references may influence calibration:

- source 0, the original Phase-40/41 development source;
- source 17, the opened Phase-41 Hausdorff failure.

Sources `25, 26, 27` are reserved for validation and their reconstruction
references must remain unevaluated until a protocol commit. No Phase-40 source,
other Phase-41 source, Leica pose, or registration-correctness label may enter
selection.

## Anchor-relative observations

For every target-only cell already satisfying Phase 41 (`support >= 2`, RMS
dispersion `<= 0.15 m`):

1. query the 12 nearest anchor-cell centroids;
2. estimate an anchor PCA normal from those centroids;
3. query the 12 nearest transformed target points and estimate a target PCA
   normal;
4. record nearest-anchor Euclidean distance;
5. record absolute displacement from the anchor neighborhood tangent plane;
6. record unsigned normal alignment `abs(n_anchor dot n_target)`.

The cell is added only when all three candidate thresholds pass. Normal signs
are intentionally ignored.

## Frozen development grid

The 18 candidates are the Cartesian product:

- maximum nearest-anchor distance: `0.75, 1.00, 1.50 m`;
- maximum anchor-plane residual: `0.15, 0.30, 0.50 m`;
- minimum normal alignment: `0.00, 0.75`.

Every candidate must add at least three cells in both development sources. It
is eligible when its two-source mean geometry loss is lower than both the
anchor-only and Phase-41 means, mean F-score is within 0.025 of anchor, and mean
recall is within 0.01 of anchor. Selection minimizes mean geometry loss; ties
prefer stronger normal alignment, shorter distance, and smaller plane residual.
Topology is descriptive and excluded.

If no candidate is eligible, stop without opening sources 25-27. If calibration
is viable, hash-lock the result and preregister those three sources. They have
five, four, and three direct target scans respectively and were excluded from
all previous reconstruction endpoint panels.

## Frozen development result

The grid was committed as `6c04859` before execution. Two candidates are
eligible. The frozen selection rule chooses `anchor_d150_p050_n075`:

- maximum nearest-anchor distance: 1.50 m;
- maximum anchor-plane residual: 0.50 m;
- minimum unsigned normal alignment: 0.75.

| Two-source mean endpoint | Anchor | Phase 41 | Selected Phase 42 |
|---|---:|---:|---:|
| Geometry loss | 0.161002 | 0.180936 | 0.160472 |
| F-score | 0.706666 | — | 0.705575 |
| Recall | 0.734173 | — | 0.741501 |

The selected route adds six cells in source 0 and nine in source 17. Calibration
artifact SHA-256:
`ddf119166d119e376acc21b0d73ba078616d6356a9a0f2093944dd7b1fb2f16f`.

This remains development evidence. It does not support a Phase-42 validation
claim until sources 25-27 are preregistered and evaluated unchanged.
