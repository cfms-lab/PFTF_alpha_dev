# Shared-trend two-layer inference: Phase 7

## Calibration finding

Phase 6 showed that observed-only rejection scores could not reliably separate
the two remaining sparse false-safe cases from safe sparse cases. Tangent-plane
overlap, mesh-normal compatibility, and normalized edge-risk distributions all
overlapped. The failure occurs earlier: global-normal two-means assigns one or
two points to the wrong layer.

Phases 5 and 6 are calibration-only for this new candidate. In their 494
base-accepted cases, a shared quadratic trend followed by residual two-means
recovered the evaluation labels in all 494 cases, including both Phase-6
survivors. True labels were used only to score this calibration result.

## Frozen algorithm

1. Estimate the global normal/tangent frame from observed kNN-PCA normals.
2. In tangent coordinates `(u,v)`, fit one shared trend with terms
   `1,u,v,u2,uv,v2` plus a binary layer offset.
3. Alternate the shared least-squares trend and residual two-means for at most
   32 iterations.
4. Recompute the existing cluster-balance, residual separation-SNR, and 3D
   cross-kNN sampling gates from the new labels.
5. Triangulate each inferred layer independently with the existing 2D Delaunay
   construction.

The method uses observed coordinates only. Generator identity, profile, true
labels, and dense references remain evaluation-only. This is conventional
mixture regression for an explicitly restricted two-layer surface model, not a
PFTF-SPD contribution.

## Frozen held-out

- Seed `20500804`, unseen by Phases 0-6.
- The same nine density/noise profiles and five curvature shapes.
- Eight repeats per cell: 360 cases.
- 2048 reference points and 256 evaluation surface samples.

## Predeclared success gate

Phase 7 passes only if:

1. the global-normal base route reproduces at least one false-safe accept;
2. shared-trend reconstruction has zero false-safe accepts;
3. at least 90% of base false-safe accepts become accepted safe outputs;
4. at least 90% of base-safe accepts remain accepted and safe; and
5. every point-count group with at least eight base-safe accepts retains at
   least 85%, with zero candidate false-safe accepts.

Even a pass remains synthetic, generator-matched, and non-deployable.

## Run

```powershell
python -m pftf_alpha.shared_trend_inference `
  --output benchmark-out/shared_trend_inference_phase7.json
```

## Result

The frozen held-out **passed** without changing the configuration:

| Metric | Global-normal base | Shared-trend candidate | Required |
|---|---:|---:|---:|
| Base-safe accepts retained | 186 | **186** | >=90% |
| Base-safe retention | - | **100%** | >=90% |
| False-safe accepts | 60 | **0** | 0 |
| Base false-safe repaired to safe accept | - | **58/60 (96.67%)** | >=90% |
| Total candidate safe accepts | - | **245** | - |

Every density gate passed:

| Observed N | Base safe retained | Base false-safe repaired | Candidate false-safe |
|---:|---:|---:|---:|
| 96 | 26/26 | 1/1 | 0 |
| 160 | 82/82 | 15/17 | 0 |
| 256 | 78/78 | 42/42 | 0 |

The two unrepaired N=160 asymmetric-converging cases produced geometrically
safe candidate meshes, but the recomputed cross-kNN sampling gate returned
`rescan_required`. This is the intended fail-closed behavior, not a silent
failure. All 360 alternating fits converged within four iterations.

`phase7_supported=true` for the declared synthetic shared-quadratic two-layer
regime. `deployment_supported=false`: the five test shapes are well matched by
the quadratic trend family, there is no sensor occlusion or outlier process,
and no real scan has been evaluated. The result supports a practical model-based
layer-inference baseline, not PFTF-SPD novelty or arbitrary multilayer geometry.

Artifact:
`benchmark-out/shared_trend_inference_phase7.json`.
