# Robust residual outlier guard: Phase 9

## Calibration

Phase 8 is calibration-only. For every candidate-accepted case, the guard fits
the same shared quadratic trend plus inferred binary layer offset. It computes
MAD-scaled, leverage-studentized residuals and multiplies each by the square
root of its local kNN-density ratio. The case score is the maximum point score.

Density-specific thresholds were frozen between the largest retained safe score
and smallest false-safe score in Phase 8:

| Observed N | Maximum joint score | Calibration safe retained |
|---:|---:|---:|
| <=96 | 9.20 | 21/21 |
| 97-160 | 7.80 | 44/48 |
| >160 | 9.65 | 47/48 |

Together these remove all 56 calibration false-safe accepts while retaining
112/117 safe accepts (95.7%). The guard never removes suspected points or emits
a trimmed reconstruction. It changes an otherwise accepted case to
`unsupported_geometry`, preserving a fail-closed policy.

## Frozen held-out

- Phase-8 panel unchanged: nine stresses x three densities x eight repeats.
- New seed `20700804`, unseen by Phases 0-8.
- 216 cases, 2048 references, and 256 surface samples.
- Phase-7 reconstruction and all sampling thresholds remain unchanged.

## Predeclared success gate

Phase 9 passes only if:

1. unguarded shared-trend reconstruction reproduces false-safe accepts;
2. the guarded route has zero false-safe accepts and removes all unguarded ones;
3. overall safe-accept retention is at least 90%;
4. every stress and density group has zero guarded false-safe accepts; and
5. every group with at least eight unguarded safe accepts retains at least 85%.

Even a pass remains synthetic and non-deployable. This conventional robust
regression diagnostic is not PFTF-SPD novelty.

## Run

```powershell
python -m pftf_alpha.outlier_guard `
  --output benchmark-out/outlier_guard_phase9.json
```

## Result

The frozen held-out **did not pass**:

| Metric | Unguarded | Guarded | Required |
|---|---:|---:|---:|
| Safe accepts | 115 | 102 | retention >=90% |
| Safe-accept retention | - | **88.70%** | >=90% |
| False-safe accepts | 58 | **4** | 0 |
| Removed false-safe accepts | - | 54/58 (93.10%) | 58/58 |

The guard removes every accepted 3% and 5% outlier case. Four 1% cases remain:
three at N=96 and one at N=256. Their maximum joint scores are `3.80-6.71`,
well below the frozen thresholds. Their injected points lie only `0.029-0.109`
from the nearest generating surface, so a residual-only provenance test cannot
reliably distinguish them from noisy valid samples.

The safe-retention loss is concentrated in the nonquadratic local-bump group:
only 9/22 unguarded safe accepts remain (`40.91%`). Control, occlusion,
imbalance, anisotropic noise, and sinusoidal safe accepts all retain 100%.
Thus the shared-quadratic residual score confounds genuine localized shape with
contamination.

Density results are:

| Observed N | Safe retention | False-safe before -> after |
|---:|---:|---:|
| 96 | 15/19 = 78.95% | 10 -> 3 |
| 160 | 43/48 = 89.58% | 24 -> 0 |
| 256 | 44/48 = 91.67% | 24 -> 1 |

`phase9_supported=false`. The guard is useful negative evidence and catches
moderate contamination, but it is neither complete nor shape-agnostic. No
threshold is retuned. The next method must separate local surface consensus
from source provenance, and the evaluation should report both harmful geometric
outliers and provenance-only near-surface samples rather than silently treating
them as equivalent.

Artifact:
`benchmark-out/outlier_guard_phase9.json`.
