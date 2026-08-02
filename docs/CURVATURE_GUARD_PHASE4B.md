# Observed-only curvature/model-adequacy guard: Phase 4b

## Calibration boundary

Phase 4 (`seed=20280804`) is calibration-only. The guard uses the sign-invariant
orientation tensor of observed kNN PCA normals,

\[
Q_n = \frac{1}{N}\sum_i n_i n_i^T,
\qquad c_n = \lambda_{\max}(Q_n).
\]

No true labels, axis names, or reference surface enter `c_n`. On Phase 4, the
largest coherence among false-safe curved cases was `0.7799`. The smallest
coherence among the Phase-3 anchor regimes that must retain coverage was
`0.8294`. The threshold is frozen at `c_n >= 0.82`; it is not optimized on the
held-out panel. A base `accept` below the threshold is changed to
`unsupported_geometry_fail_closed`. Other base decisions are unchanged.

This is a conventional local-normal adequacy guard, not evidence for PFTF local
SPD superiority.

## Frozen held-out

- Seed: `20290804`, unseen by Phases 0-4.
- Same four axes and six levels as Phase 4.
- Eight repeats per level, 160 observed points, 2048 reference points.
- 192 cases total; surface evaluation samples: 256.
- Original Phase-4 router and guarded router are evaluated on identical meshes.

## Predeclared success gate

Phase 4b passes only if:

1. the unguarded router still produces at least one false-safe case, proving the
   held-out challenge retained the known failure mode;
2. the guarded router produces zero false-safe cases;
3. every unguarded false-safe accept is removed;
4. at least 90% of unguarded safe accepts are retained; and
5. curvature 0.12/0.24, tilt 0.25/0.40, overlap offset 1.00, and contact severity
   0.20 each retain at least 75% guarded safe acceptance.

Even a pass remains synthetic and sets `deployment_supported=false`.

## Run

```powershell
python -m pftf_alpha.curvature_guard `
  --output benchmark-out/curvature_guard_phase4b.json
```

## Result

The frozen held-out panel passed every predeclared gate.

| Endpoint | Base router | Guarded router |
|---|---:|---:|
| Safe accepts | 117 | 116 |
| False-safe accepts | 14 | **0** |
| Safe-accept retention | - | **99.15%** |

All 14 held-out false-safe accepts were removed. The only lost safe accept was
one curvature-0.36 case; that level changed from 1 safe + 7 false-safe accepts
to 8 fail-closed rejections.

| Curvature | Mean coherence | Base accept | Guarded accept | Base false safe | Guarded false safe |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.9850 | 8/8 | 8/8 | 0 | 0 |
| 0.12 | 0.9594 | 8/8 | 8/8 | 0 | 0 |
| 0.24 | 0.8798 | 8/8 | 8/8 | 0 | 0 |
| 0.36 | 0.7610 | 8/8 | 0/8 | 7 | 0 |
| 0.48 | 0.6395 | 5/8 | 0/8 | 5 | 0 |
| 0.60 | 0.5552 | 2/8 | 0/8 | 2 | 0 |

The required anchors retained coverage: curvature 0.12/0.24 and tilt 0.25 were
8/8, tilt 0.40 was 7/8 because the base sampling gate rejected one case,
overlap offset 1.00 was 8/8, and contact severity 0.20 was 8/8. The guard also
retained all four base-safe accepts at contact severity 0.40.

Artifact: `benchmark-out/curvature_guard_phase4b.json`.

`phase4b_supported=true`, while `deployment_supported=false`. The result supports
the normal-coherence threshold as a low-cost fail-closed certificate for this
frozen synthetic generator and point density. It does not establish a universal
curvature threshold, real-scan calibration, arbitrary surface support, or PFTF
local-SPD superiority.
