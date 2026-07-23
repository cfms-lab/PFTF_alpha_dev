# PFTF-alpha first benchmark plan

Status: proposed; no benchmark has run.

## Objective

Determine whether a PFTF-derived local SPD metric and confidence-aware fallback
improve alpha-shape reconstruction beyond established density-scaled and
normal-based anisotropic baselines.

## Minimal reproducible panel

| Family | Variation | Required failure signal |
|---|---|---|
| U/C concavity | opening width, density | convex-hull overfill |
| opposing sheets | gap / sample spacing | false bridge |
| torus | hole radius, noise | topology loss |
| disconnected parts | separation | false merge |
| sharp crease | angle, normal noise | webbing or hole |
| missing patch | missing area | unstable closure |

Every family uses fixed train/calibration/held-out seeds. Held-out conditions
include unseen density ratios, gap-to-spacing ratios, and noise levels.

## Methods

| ID | Method | Purpose |
|---|---|---|
| B0 | convex hull | over-smoothing reference |
| B1 | hand-picked global alpha | common manual workflow |
| B2 | exhaustive critical-alpha scan | global selection oracle |
| B3 | persistence + resampling stability | unlabeled automatic baseline |
| B4 | kNN density-scaled alpha | adaptive-scale baseline |
| B5 | normal/PCA anisotropic alpha | anisotropic prior-art baseline |
| P1 | PFTF local SPD metric | field contribution |
| P2 | P1 + confidence/exact fallback | safety contribution |

## Primary endpoints

1. False bridges per shape.
2. Betti/component error.
3. Surface F-score and Hausdorff distance.
4. Stability under 10% subsampling and calibrated noise.
5. Non-manifold and singular-face counts.
6. Runtime, memory, and fallback rate.

## Required ablations

- PFTF relation field versus local PCA covariance.
- Isotropic versus anisotropic metric.
- Density only, normal only, relation only, combined.
- Hard gate versus soft calibration followed by hard exact evaluation.
- Confidence fallback on/off.

## Promotion rule

The project remains **ToDo** until B0-B5 and at least P1 run end to end.
It can be promoted to a paper-quality tier only after frozen held-out results
show value beyond both B4 and B5 with no unreported false-safe cases.
